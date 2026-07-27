# -*- coding: utf-8 -*-
"""
PhysTrap_xch main entry script.

Functionality:
  - Reads the seed list from a JSON seed file (see --seed-file).
  - Runs the full PhysTrap v3 pipeline in parallel across N workers.
  - Live progress bar via tqdm.
  - Checkpoint/resume support (a checkpoint file automatically skips
    already-completed seeds).
  - Each seed produces a structured JSON record containing the seed's
    basic info, the synthesized candidate question, the ground truth, etc.

Usage:
  python run_pipeline.py --app-key YOUR_APP_KEY [options]

Options:
  --app-key           API gateway app key (can also be set via the
                       PHYSTRAP_APP_KEY environment variable)
  --seed-file         Path to the input seed file (default: ./seeds/scaling_final.json)
  --output-dir        Output directory (default: ./outputs)
  --workers           Number of parallel worker threads (default: 20)
  --max-seeds         Max number of seeds to process (debug only, 0 = no limit)
  --generator-rounds  Max rounds for the Generator (default: 2)
  --anneal-rounds     Max rounds for annealing (default: 3)
  --max-candidates    Per-seed candidate cap (default: 12)
  --resume            Whether to resume from checkpoint (default: True)
  --seed-offset       Start processing from this seed index (0-indexed, default: 0)
  --seed-limit        Number of seeds to process per run (0 = no limit)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from tqdm import tqdm
except ImportError:
    # Minimal fallback if tqdm is not installed.
    class tqdm:  # type: ignore
        def __init__(self, total=0, desc="", unit="", **kwargs):
            self.total = total
            self.n = 0
            self.desc = desc
        def update(self, n=1):
            self.n += n
            print(f"\r{self.desc}: {self.n}/{self.total}", end="", flush=True)
        def set_postfix_str(self, s):
            pass
        def close(self):
            print()
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.close()


# ===================== path configuration =====================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR  # PhysTrap_xch/

V3_PIPELINE_PATH = PROJECT_ROOT / "v3_registry" / "phystrap_v3_candidate_registry.py"
API_CLIENT_PATH = PROJECT_ROOT / "api_client.py"

DEFAULT_SEED_FILE = PROJECT_ROOT / "seeds" / "scaling_final.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"


# ===================== dynamic module loading =====================
def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_api_client_module():
    """Load the api_client module, returning (PhysTrapLLMClient, ApiExhaustedError)."""
    module = load_module("api_client_phystrap_main", API_CLIENT_PATH)
    return module.PhysTrapLLMClient, module.ApiExhaustedError


def load_api_client_class():
    """Backward-compatible helper that returns only PhysTrapLLMClient."""
    return load_api_client_module()[0]


# ===================== checkpointing =====================
_checkpoint_lock = threading.Lock()


def load_checkpoint(checkpoint_file: Path) -> Dict[str, Any]:
    """Load the checkpoint file, returning the set of completed seed_ids and existing results."""
    if not checkpoint_file.exists():
        return {"done_ids": set(), "results": []}
    with open(checkpoint_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    done_ids = set(data.get("done_ids", []))
    results = data.get("results", [])
    return {"done_ids": done_ids, "results": results}


def save_checkpoint(checkpoint_file: Path, done_ids: set, results: list):
    """Write the current progress to the checkpoint file (thread-safe)."""
    with _checkpoint_lock:
        tmp_file = checkpoint_file.with_suffix(".tmp")
        payload = {
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "done_count": len(done_ids),
            "done_ids": sorted(done_ids),
            "results": results,
        }
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        tmp_file.replace(checkpoint_file)


# ===================== structured output record =====================
def build_output_record(
    seed: Dict[str, Any],
    v3_result: Optional[Dict[str, Any]],
    error: Optional[str] = None,
    error_type: str = "error",
) -> Dict[str, Any]:
    """
    Build a standardized output record. Includes:
      - basic seed info (id, name, dimension_tag, initial_prompt, trap_core,
        injected_numbers, ground_truth)
      - the pipeline's best synthesized candidate
      - Judge label
      - pipeline status

    error_type:
      - "error"          generic runtime error
      - "api_exhausted"  all models rate-limited / retries exhausted; this seed is skipped
    """
    record: Dict[str, Any] = {
        "seed_id": seed.get("id", ""),
        "seed_name": seed.get("name", ""),
        "dimension_tag": seed.get("dimension_tag", ""),
        "initial_prompt": seed.get("initial_prompt", ""),
        "trap_core": seed.get("trap_core", ""),
        "injected_numbers": seed.get("injected_numbers", []),
        "ground_truth": seed.get("ground_truth", ""),
        "generation_hint": seed.get("generation_hint", ""),
        "annealing_hint": seed.get("annealing_hint", ""),
    }

    if error:
        record["status"] = error_type  # "error" or "api_exhausted"
        record["error"] = error
        record["best_candidate"] = None
        record["all_certified_candidates"] = []
        return record

    if v3_result is None:
        record["status"] = "skipped"
        record["best_candidate"] = None
        record["all_certified_candidates"] = []
        return record

    # Extract the best candidate from the ranked results.
    seed_id = seed.get("id", "")
    ranked_info = (v3_result.get("ranked") or {}).get(seed_id, {})
    top_all = ranked_info.get("top_all", [])
    registry_data = v3_result.get("_registry_data") or []

    def find_candidate(cand_id):
        for c in registry_data:
            if c.get("candidate_id") == cand_id:
                return c
        return None

    # Build the list of certified candidates.
    certified = []
    for top_cand in top_all:
        cid = top_cand.get("candidate_id")
        full_cand = find_candidate(cid) or {}
        evaluations = full_cand.get("evaluations") or {}
        naturalness = evaluations.get("naturalness") or {}
        alignment = evaluations.get("alignment") or {}
        solver_msg = full_cand.get("solver") or {}
        judge_info = full_cand.get("judge") or {}
        certified.append({
            "candidate_id": cid,
            "source_stage": top_cand.get("source_stage", ""),
            "parent_id": top_cand.get("parent_id"),
            "synthesized_prompt": top_cand.get("prompt", ""),
            "judge_label": top_cand.get("label", ""),
            "attack_tier": top_cand.get("attack_tier", ""),
            "naturalness_score": top_cand.get("naturalness_score"),
            "score": top_cand.get("score", 0.0),
            "eraser_test_pass": alignment.get("eraser_test_pass"),
            "target_alignment_pass": alignment.get("target_alignment_pass"),
            "normal_subtask_escape": alignment.get("normal_subtask_escape"),
            "phystrap_valid": alignment.get("phystrap_valid"),
            "solver_content": (solver_msg.get("message") or {}).get("content", ""),
            "solver_model": solver_msg.get("model", ""),
            "judge_reason": judge_info.get("reason", ""),
            "judge_awareness_quote": judge_info.get("trap_awareness_quote", ""),
        })

    best = certified[0] if certified else None
    record["status"] = "success" if best else "no_certified_candidate"
    record["best_candidate"] = best
    record["all_certified_candidates"] = certified
    record["global_top_candidate_id"] = (v3_result.get("global_top") or {}).get("candidate_id")

    # Seed feasibility result.
    seed_results = v3_result.get("seed_results") or {}
    seed_result_info = seed_results.get(seed_id) or {}
    viability = seed_result_info.get("seed_viability") or {}
    record["seed_viability"] = {
        "seed_grade": viability.get("seed_grade"),
        "reason": viability.get("reason"),
        "eraser_test_pass": viability.get("eraser_test_pass"),
        "target_binding_pass": viability.get("target_binding_pass"),
        "no_normal_subtask_escape": viability.get("no_normal_subtask_escape"),
    }

    return record


# ===================== per-seed processing =====================

# Thread-local storage so each worker thread uses its own v3 module instance.
_thread_local = threading.local()


def _get_v3_module_for_thread(app_key: str, args):
    """Give each thread its own independent v3 module instance (avoids shared global state)."""
    if not hasattr(_thread_local, "v3") or _thread_local.v3 is None:
        # Each thread loads its own copy of the v3 module.
        thread_id = threading.get_ident()
        v3 = load_module(f"phystrap_v3_{thread_id}", V3_PIPELINE_PATH)

        # Create an independent LLM client.
        PhysTrapLLMClient = load_api_client_class()
        client = PhysTrapLLMClient(app_key=app_key)

        # Inject the client into v2.
        v2 = v3.load_module(f"phystrap_v2_{thread_id}", v3.V2_PIPELINE_PATH)
        v2.set_api_client(client)

        # Inject the client into the stability-retest module.
        stability = v3.load_module(f"phystrap_stability_{thread_id}", v3.STABILITY_PATH)
        stability.set_api_client(client)

        _thread_local.v3 = v3
        _thread_local.v2 = v2
        _thread_local.stability = stability
        _thread_local.client = client

    return _thread_local.v3, _thread_local.v2, _thread_local.stability


def process_seed(seed: Dict[str, Any], app_key: str, args, run_id: str, log_dir: Path) -> Dict[str, Any]:
    """
    Run the full PhysTrap v3 pipeline on a single seed and return a
    standardized output record.
    """
    seed_id = seed.get("id", "unknown")
    seed_name = seed.get("name", "unknown")

    try:
        v3, v2, stability = _get_v3_module_for_thread(app_key, args)

        # Configure v2/stability log files (all threads share the same run_id).
        thread_id = threading.get_ident()
        v2.RUN_ID = run_id
        v2.LOG_DIR = str(log_dir)
        v2.LOG_FILE = str(log_dir / f"v2_trace_{run_id}_{thread_id}.md")
        v2.RESULTS_FILE = str(log_dir / f"v2_results_{run_id}_{thread_id}.json")
        v2.ALIGNMENT_CACHE = {}  # clear cache for each seed

        stability.RUN_ID = run_id
        stability.LOG_DIR = str(log_dir)
        stability.LOG_FILE = str(log_dir / f"stability_{run_id}_{thread_id}.md")
        stability.RESULTS_FILE = str(log_dir / f"stability_results_{run_id}_{thread_id}.json")

        # Load the experience bank if present.
        experience_bank = v3.read_text(v3.EXPERIENCE_BANK_PATH, "")

        # Create an independent Registry instance.
        log_messages = []

        def seed_log(text=""):
            log_messages.append(str(text))

        registry = v3.CandidateRegistry(seed_log)

        # === Seed viability pre-screening ===
        seed_viability = v2.call_seed_viability_checker(seed)

        # === Base candidate: start directly from seed.initial_prompt ===
        base = registry.add_candidate(seed, seed["initial_prompt"], source_stage="seed", note="original seed")
        ok = v3.evaluate_basic(v2, seed_log, base, seed, "v3_seed_baseline")
        if ok:
            v3.solver_judge(v2, seed_log, base, seed)

        # === High-temperature exploration ===
        generation_parent = base
        effective_round = 0
        generator_attempt = 0
        max_generator_attempts = args.generator_rounds * (args.generator_duplicate_retries + 1)

        while effective_round < args.generator_rounds and generator_attempt < max_generator_attempts:
            current_count = len(registry.active_candidates(seed["id"]))
            if current_count >= args.max_candidates_per_seed:
                break
            generator_attempt += 1
            candidate_round = effective_round + 1
            child = v3.generate_high_temp_candidate(
                v2, seed_log, registry, seed, generation_parent, candidate_round, experience_bank
            )
            if not child:
                effective_round += 1
                continue
            if child["status"] == "exact_duplicate":
                continue
            else:
                effective_round += 1
                generation_parent = child

        # === Route queue (annealing, light enhancement) ===
        v3.process_route_queue(v2, seed_log, registry, seed, experience_bank, args)

        # === Ranking ===
        ranked = v3.rank_candidates(registry)

        # === Build v3_result ===
        seed_results_info = {
            seed_id: {
                "seed_name": seed_name,
                "seed_viability": seed_viability,
                "candidate_ids": [c["candidate_id"] for c in registry.candidates if c["seed_id"] == seed_id],
            }
        }

        ranked_json = {}
        global_top = None
        info = ranked.get(seed_id, {})
        top_all = info.get("top_all", [])
        ranked_json[seed_id] = {
            "seed_name": seed_name,
            "top_all": [
                {
                    "candidate_id": c["candidate_id"],
                    "source_stage": c["source_stage"],
                    "parent_id": c.get("parent_id"),
                    "label": c.get("judge", {}).get("label"),
                    "attack_tier": c["attack_tier"],
                    "naturalness_score": c.get("evaluations", {}).get("naturalness", {}).get("average_score"),
                    "score": c["score"],
                    "prompt": c["prompt"],
                }
                for c in top_all
            ],
            "top_strong": info.get("top_strong", {}).get("candidate_id") if info.get("top_strong") else None,
            "top_soft": info.get("top_soft", {}).get("candidate_id") if info.get("top_soft") else None,
        }
        for candidate in top_all:
            if global_top is None or candidate["score"] > global_top["score"]:
                global_top = candidate

        v3_result = {
            "run_id": run_id,
            "seed_results": seed_results_info,
            "ranked": ranked_json,
            "global_top": {
                "candidate_id": global_top["candidate_id"],
                "seed_name": global_top["seed_name"],
                "source_stage": global_top["source_stage"],
                "parent_id": global_top.get("parent_id"),
                "label": global_top.get("judge", {}).get("label"),
                "attack_tier": global_top["attack_tier"],
                "score": global_top["score"],
                "prompt": global_top["prompt"],
            } if global_top else None,
            "_registry_data": registry.to_json(),
        }

        output_record = build_output_record(seed, v3_result)
        return output_record

    except Exception as e:
        error_msg = traceback.format_exc()
        # Detect API exhaustion (all models rate-limited) - a soft failure
        # that should not abort the overall run.
        # Note: api_client is loaded dynamically, so multiple module
        # instances may exist; isinstance may fail across different module
        # origins, so we check by class name + module name instead.
        e_type = type(e)
        is_exhausted = (
            e_type.__name__ == "ApiExhaustedError"
            or "ApiExhaustedError" in str(type(e))
            or "api_exhausted" in str(error_msg).lower()
        )
        if is_exhausted:
            print(
                f"[api_exhausted] seed={seed_id}({seed_name}) "
                f"API retries exhausted, skipping this seed: {repr(e)}",
                flush=True,
            )
            return build_output_record(
                seed, None,
                error=f"{repr(e)}\n{error_msg}",
                error_type="api_exhausted",
            )
        return build_output_record(seed, None, error=f"{repr(e)}\n{error_msg}")


# ===================== main =====================
def main():
    parser = argparse.ArgumentParser(description="PhysTrap_xch main pipeline entry (reads seeds from a JSON file)")
    parser.add_argument("--app-key", default=os.getenv("PHYSTRAP_APP_KEY", ""),
                        help="API gateway app key (or set via the PHYSTRAP_APP_KEY environment variable)")
    parser.add_argument("--seed-file", type=Path, default=DEFAULT_SEED_FILE,
                        help="Path to the input seed file (JSON, containing a 'seeds' list)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory (result JSON and checkpoint are written here)")
    parser.add_argument("--workers", type=int, default=20, help="Number of parallel worker threads (default 20)")
    parser.add_argument("--max-seeds", type=int, default=0, help="Debug: max number of seeds to process (0 = no limit)")
    parser.add_argument("--seed-offset", type=int, default=0, help="Start from this seed index (0-indexed)")
    parser.add_argument("--seed-limit", type=int, default=0, help="Number of seeds to process this run (0 = no limit)")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from checkpoint (enabled by default)")
    parser.add_argument("--no-resume", action="store_false", dest="resume", help="Disable checkpoint resume")
    # pipeline parameters
    parser.add_argument("--generator-rounds", type=int, default=2)
    parser.add_argument("--generator-duplicate-retries", type=int, default=2)
    parser.add_argument("--anneal-rounds", type=int, default=3)
    parser.add_argument("--enhance-rounds", type=int, default=2)
    parser.add_argument("--max-route-queue-steps", type=int, default=18)
    parser.add_argument("--route-patience", type=int, default=2)
    parser.add_argument("--max-light-strict-chain-depth", type=int, default=2)
    parser.add_argument("--max-candidates-per-seed", type=int, default=12)
    parser.add_argument("--stability-repeats", type=int, default=0)
    args = parser.parse_args()

    # ---- check app-key ----
    if not args.app_key:
        print("[ERROR] --app-key or the PHYSTRAP_APP_KEY environment variable must be set", flush=True)
        sys.exit(1)

    # ---- output directory ----
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs" / run_id
    log_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "results.jsonl"  # append mode, one seed result per line
    checkpoint_file = output_dir / "checkpoint.json"

    print(f"[PhysTrap_xch] run_id={run_id}", flush=True)
    print(f"[PhysTrap_xch] output_dir={output_dir}", flush=True)
    print(f"[PhysTrap_xch] seed_file={args.seed_file}", flush=True)
    print(f"[PhysTrap_xch] workers={args.workers}", flush=True)

    # ---- load seeds ----
    if not args.seed_file.exists():
        print(f"[ERROR] seed file does not exist: {args.seed_file}", flush=True)
        sys.exit(1)

    with open(args.seed_file, "r", encoding="utf-8") as f:
        seed_data = json.load(f)
    all_seeds: List[Dict[str, Any]] = seed_data.get("seeds", [])
    print(f"[PhysTrap_xch] loaded {len(all_seeds)} seeds", flush=True)

    # ---- offset / slice ----
    seeds = all_seeds[args.seed_offset:]
    if args.seed_limit > 0:
        seeds = seeds[:args.seed_limit]
    if args.max_seeds > 0:
        seeds = seeds[:args.max_seeds]
    print(f"[PhysTrap_xch] processing {len(seeds)} seeds this run (offset={args.seed_offset}, limit={args.seed_limit}, max={args.max_seeds})", flush=True)

    # ---- checkpoint resume ----
    checkpoint_data = load_checkpoint(checkpoint_file) if args.resume else {"done_ids": set(), "results": []}
    done_ids: set = checkpoint_data["done_ids"]
    accumulated_results: List[Dict[str, Any]] = checkpoint_data["results"]

    pending_seeds = [s for s in seeds if s.get("id") not in done_ids]
    skipped_count = len(seeds) - len(pending_seeds)
    if skipped_count > 0:
        print(f"[PhysTrap_xch] checkpoint resume: skipping {skipped_count} already-completed seeds", flush=True)
    print(f"[PhysTrap_xch] pending: {len(pending_seeds)} seeds", flush=True)

    if not pending_seeds:
        print("[PhysTrap_xch] all seeds already completed, nothing to do.", flush=True)
        _write_final_summary(output_dir, accumulated_results, run_id)
        return

    # ---- parallel processing ----
    results_lock = threading.Lock()

    with tqdm(total=len(pending_seeds), desc="Processing seeds", unit="seed") as pbar:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_seed = {
                executor.submit(process_seed, seed, args.app_key, args, run_id, log_dir): seed
                for seed in pending_seeds
            }

            for future in as_completed(future_to_seed):
                seed = future_to_seed[future]
                seed_id = seed.get("id", "unknown")
                try:
                    result = future.result()
                except Exception as e:
                    result = build_output_record(seed, None, error=repr(e))

                status = result.get("status", "unknown")
                label = (result.get("best_candidate") or {}).get("judge_label", "-")
                pbar.set_postfix_str(f"seed={seed_id} status={status} label={label}")

                # append to results.jsonl (thread-safe)
                with results_lock:
                    done_ids.add(seed_id)
                    accumulated_results.append(result)
                    with open(results_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")

                # save checkpoint after each completed seed
                save_checkpoint(checkpoint_file, done_ids, accumulated_results)

                pbar.update(1)

    # ---- write summary file ----
    _write_final_summary(output_dir, accumulated_results, run_id)

    print(f"\n[PhysTrap_xch] Done! processed {len(accumulated_results)} seeds", flush=True)
    print(f"  results (one JSON per line): {results_file}", flush=True)
    print(f"  summary: {output_dir / 'summary.json'}", flush=True)


def _write_final_summary(output_dir: Path, results: List[Dict[str, Any]], run_id: str):
    """Write the summary statistics file."""
    total = len(results)
    success = sum(1 for r in results if r.get("status") == "success")
    no_cert = sum(1 for r in results if r.get("status") == "no_certified_candidate")
    error = sum(1 for r in results if r.get("status") == "error")
    skipped = sum(1 for r in results if r.get("status") == "skipped")

    # statistics grouped by dimension_tag
    dim_stats: Dict[str, Dict[str, int]] = {}
    for r in results:
        dim = r.get("dimension_tag", "unknown")
        s = r.get("status", "unknown")
        dim_stats.setdefault(dim, {})
        dim_stats[dim][s] = dim_stats[dim].get(s, 0) + 1

    # label distribution
    label_counts: Dict[str, int] = {}
    for r in results:
        best = r.get("best_candidate")
        if best:
            label = best.get("judge_label", "unknown")
            label_counts[label] = label_counts.get(label, 0) + 1

    summary = {
        "run_id": run_id,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total": total,
        "success": success,
        "no_certified_candidate": no_cert,
        "error": error,
        "skipped": skipped,
        "success_rate": round(success / total, 4) if total else 0.0,
        "label_distribution": label_counts,
        "dimension_stats": dim_stats,
    }

    summary_file = output_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # also write the full results to results_full.json for convenient inspection
    full_results_file = output_dir / "results_full.json"
    with open(full_results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
