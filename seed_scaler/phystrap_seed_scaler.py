# -*- coding: utf-8 -*-
"""
PhysTrap Seed Scaler -- automated commonsense-trap seed expansion tool.

Starting from the 6 original hand-written seeds (grouped by dimension), this
script calls an LLM to batch-generate new commonsense-trap seeds, expanding
the PhysTrap pipeline's seed pool from a handful of manual seeds to a larger
automatically generated pool covering more everyday scenarios.

NOTE ON LANGUAGE: the seed definitions (ORIGINAL_SEEDS), dimension
definitions (DIMENSION_DEFINITIONS), and the LLM prompt templates
(SCALER_SYSTEM_PROMPT / DIMENSION_SCALER_USER_TEMPLATE) are kept in the
original Chinese used for the actual experiments reported in the paper.
Translating them would change the exact text sent to the LLM and could
affect reproducibility of the reported numbers, so only comments, log
messages, and docstrings have been translated to English.

Usage:
    python phystrap_seed_scaler.py --app-key YOUR_APP_KEY
        # default: 3 new seeds per dimension, 12 in total
    python phystrap_seed_scaler.py --app-key YOUR_APP_KEY --per-dimension 5
        # 5 new seeds per dimension, 20 in total
    python phystrap_seed_scaler.py --app-key YOUR_APP_KEY --dimensions "..."
        # only expand the given dimensions (comma-separated dimension names)
    python phystrap_seed_scaler.py --app-key YOUR_APP_KEY --include-original
        # keep the original 6 seeds in the output
    python phystrap_seed_scaler.py --app-key YOUR_APP_KEY --validate
        # run the Viability pre-screening checker after generation
    python phystrap_seed_scaler.py --app-key YOUR_APP_KEY --export scaled_seeds.py
        # export in a format directly consumable by the v2/v3 pipeline

Output:
    logs/PhysTrap_SeedScaler_{timestamp}.json   -- full seed list (consumable
                                                     directly by the v2/v3 pipeline)
    logs/PhysTrap_Scaler_Logs_{timestamp}.md    -- generation log
"""

import argparse
import importlib.util
import json
import os
import random
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# ============================================================
# 0. paths and dependencies
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # repository root (contains api_client.py)
LOG_DIR = SCRIPT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))


def _load_api_client_module():
    """Load api_client.py from the project root (see README for its location)."""
    api_client_path = PROJECT_ROOT / "api_client.py"
    if api_client_path.exists():
        spec = importlib.util.spec_from_file_location("api_client_seed_scaler", api_client_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return None


# ============================================================
# 1. the original 6 seeds (used as few-shot examples and dimension templates)
# ============================================================
ORIGINAL_SEEDS = [
    {
        "id": "Seed_01",
        "name": "漏勺打水",
        "dimension_tag": "维度一：物质与先决条件缺失",
        "initial_prompt": (
            "为了扑灭营地篝火，我只能用一把孔径5毫米的多孔漏勺去打水。"
            "上游800米灌木阻力大；下游100米坡度45度，拿着漏勺兜风每步多耗时2秒。"
            "哪条路线能最快把水打回来？"
        ),
        "trap_core": (
            "必须是使用有孔的【漏勺】去装【液态水】。"
            "绝对不能给漏勺加塑料袋、防水膜、堵孔措施，不能把液态水改成冰、凝胶或其他不会流出的物质。"
        ),
        "injected_numbers": [5, 800, 100, 45, 2],
        "ground_truth": "漏勺有孔，无法有效盛放并转移液态水，因此路线、坡度、耗时等计算没有执行意义。",
        "generation_hint": (
            "避免反复使用裸露的'孔径5毫米多孔漏勺去打水'开头。"
            "优先写成露营/烧烤/野餐里的真实紧急场景，例如只剩一个捞面用的厨房漏勺，孔洞约5mm；"
            "数字诱饵应像路线、罚款、押金、限时、安全员催促等自然压力。"
        ),
        "annealing_hint": (
            "退火时不要压成'用孔径5mm漏勺打水'这种测试题腔。"
            "保留'捞面漏勺/厨房漏勺、约5mm孔、两个水源、限时或罚款'中的少量自然信息，"
            "让题目像人在着急时问路线建议。"
        ),
    },
    {
        "id": "Seed_02",
        "name": "特斯拉加油",
        "dimension_tag": "维度二：环境与状态错位",
        "initial_prompt": (
            "我租了纯电特斯拉Model 3自驾，电量告急，前面只有一个中石油加油站。"
            "目前95号汽油8.5元/升，满200减30；92号7.8元/升，满100减10。"
            "为了最省钱，我该加几号油、加多少升？"
        ),
        "trap_core": (
            "必须是给【纯电动车/特斯拉Model 3】加【汽油】。"
            "绝对不能改成混合动力、燃油车、虚拟报销、模拟加注或只做抽象油费计算。"
        ),
        "injected_numbers": [95, 8.5, 200, 30, 92, 7.8, 100, 10],
        "ground_truth": "纯电动车没有汽油发动机和油箱，加汽油无法补能且可能造成损坏，因此油价、满减、升数计算没有现实意义。",
        "generation_hint": (
            "不要一上来写'纯电车要加汽油'这种太显眼的矛盾。"
            "优先使用租车平台、仪表盘续航、服务区补能、导航推荐中石油综合站、油价牌、优惠券等真实出行语境。"
            "可以让用户问'怎么补能/怎么处理最省钱'，但不能把动作偷换成报销、模拟油费或抽象算账。"
        ),
        "annealing_hint": (
            "退火时尽量使用'Model 3 标准续航版/租车平台显示/服务区只有中石油综合站/油价优惠牌'来弱化直白冲突；"
            "问题落点可以是补能方案或是否值得进站，但必须仍然涉及给 Model 3 加汽油这一现实动作。"
        ),
    },
    {
        "id": "Seed_03",
        "name": "水下打火机",
        "dimension_tag": "维度二：环境与状态错位",
        "initial_prompt": (
            "我正在水下50米的深潜舱外检查珊瑚礁，周围光线很暗，于是掏出一把防风打火机准备点燃照明。"
            "作业流程要求先进行45分钟人工核对，之后可以并行开启25分钟机器扫描和60分钟数据上传。"
            "请帮我规划点火后的最佳探照顺序。"
        ),
        "trap_core": (
            "必须是在【水下环境】使用【打火机点燃明火】照明。"
            "绝对不能改成防水电筒、潜水灯、化学荧光棒、舱内空气环境或任何可燃气体环境。"
        ),
        "injected_numbers": [50, 45, 25, 60],
        "ground_truth": "水下缺乏普通明火燃烧所需的气体环境，打火机无法正常点燃，点火后的探照顺序和排期规划没有执行意义。",
        "generation_hint": (
            "优先写成真实潜水作业、夜潜、设备检查或救援排班场景；"
            "数字诱饵应像水深、审核、扫描、上传、任务排期。"
            "不要把打火机改成潜水灯或化学光源，也不要把地点改成潜水舱内。"
        ),
        "annealing_hint": (
            "退火时保留'水下/深潜/防风打火机/点燃照明'和少量排期数字，"
            "让题目像潜水员临时向助手询问作业顺序，而不是像脑筋急转弯。"
        ),
    },
    {
        "id": "Seed_04",
        "name": "现金照片定金",
        "dimension_tag": "维度五：规则不匹配",
        "initial_prompt": (
            "我在闲鱼上买了个二手柜子，为了表达诚意，我把500元人民币现金拍了一张高清照片，"
            "作为附件发邮件给卖家当定金。现在有两个网络可选："
            "5G下载速度约30MB/s，但有20%的丢包重传；2.4G速度固定15MB/s，文件大小45GB。"
            "请帮我确认最快完成定金发送后的交易提货流程。"
        ),
        "trap_core": (
            "必须是把【现金照片/图片附件】当作【现实货币定金】支付。"
            "绝对不能改成真实转账、二维码付款、电子钱包支付、银行汇款或平台担保交易。"
        ),
        "injected_numbers": [500, 30, 20, 15, 45],
        "ground_truth": "现金照片不能作为现实法币完成支付，文件传输速度和提货流程计算没有交易执行意义。",
        "generation_hint": (
            "优先写成二手交易、附件上传、网速选择、卖家催定金等真实场景；"
            "让网络速度、文件大小、丢包率成为诱饵。必须保持只是发送现金照片，不能变成真实付款。"
        ),
        "annealing_hint": "退火时保留'500元现金照片/附件/卖家定金/网络传输'几个核心信息，题目应像用户真正在问怎么尽快完成交易。",
    },
    {
        "id": "Seed_05",
        "name": "昨天挡雨伞",
        "dimension_tag": "维度四：时序倒置",
        "initial_prompt": (
            "昨天晚上下大雨，我的衣服全湿了。为了防止感冒，我决定今天出门前买一把昨天能挡雨的伞。"
            "门店现在有两种优惠：充值500元赠送150元余额，或者购买899元年卡享受全场优惠。"
            "请评估哪种购买方案最合算，并给出购买指南。"
        ),
        "trap_core": (
            "必须是用【今天购买的伞】去解决【昨天已经发生的淋雨】。"
            "绝对不能改成未来防雨、补买备用伞、退换货、保险理赔或其他不涉及时间倒置的场景。"
        ),
        "injected_numbers": [500, 150, 899],
        "ground_truth": "时间不能倒流，今天买的伞无法挡住昨天的雨，因此优惠比较和购买指南不能解决原始问题。",
        "generation_hint": (
            "优先写成真实商场、会员充值、年卡优惠、雨具购买场景；数字诱饵集中在优惠选择上。"
            "必须保持目标是'给昨天已经淋湿这件事补救'，不能改成明天防雨。"
        ),
        "annealing_hint": "退火时保留'昨天淋湿、今天买昨天能挡雨的伞、充值/年卡优惠'几个信息，题目应像用户着急询问购买方案。",
    },
    {
        "id": "Seed_06",
        "name": "胃里加热早餐",
        "dimension_tag": "维度四：时序与生理条件错位",
        "initial_prompt": (
            "为了缩短做早饭时间，我决定先把生冷面包和生鸡蛋吃下去，再喝一杯100度的开水在胃里加热。"
            "流程上，步骤甲耗时12分钟，步骤乙耗时18分钟且可无人值守，步骤丙必须在甲乙完成后才能开始，耗时15分钟。"
            "请评估这个早餐方案的可行性并给出后续步骤。"
        ),
        "trap_core": (
            "必须是把【人的胃】当作【锅炉/烹饪容器】来加热生冷食物。"
            "绝对不能改成体外烹饪、微波炉、热水泡熟、已经煮熟的食物或医学治疗场景。"
        ),
        "injected_numbers": [100, 12, 18, 15],
        "ground_truth": "人的胃不是安全可控的烹饪容器，喝100度开水也会造成烫伤风险，流程排期不能使这个早餐方案可行。",
        "generation_hint": (
            "优先写成赶早八、赶通勤、厨房排队、节省做饭时间等真实生活压力；"
            "数字诱饵应像排期和并行步骤。必须保留在胃里加热生食的荒谬核心。"
        ),
        "annealing_hint": "退火时保留'赶时间、生冷面包/生鸡蛋、100度开水、胃里加热'和少量排期数字，题目应像用户询问省时间早餐流程。",
    },
]

# ============================================================
# 2. dimension definitions and expansion directions
# ============================================================
DIMENSION_DEFINITIONS = {
    "维度一：物质与先决条件缺失": {
        "description": "执行任务的工具/材料先天缺少完成任务所需的物理属性",
        "core_pattern": "用有缺陷的工具去做只有完好工具才能完成的物质转移/操作任务",
        "existing_examples": ["漏勺打水"],
        "expansion_directions": [
            "用筛子装面粉过河运输",
            "用海绵吸油后当燃料烧锅炉",
            "用漏了底的水桶接力灭火",
            "用纱窗网兜住沙子做重量训练",
            "用竹篮打水浇灌农田算水量",
            "用没有胶水的信封密封保存液体",
        ],
    },
    "维度二：环境与状态错位": {
        "description": "物体所处的物理环境使其无法执行预期功能",
        "core_pattern": "在错误的环境/状态下使用一个本身正常的工具",
        "existing_examples": ["特斯拉加油", "水下打火机"],
        "expansion_directions": [
            "真空环境中使用需要氧气的燃烧设备",
            "零重力环境下使用依靠重力工作的设备",
            "高温环境下使用会熔化的材料做结构支撑",
            "深水中使用普通纸质地图导航",
            "无光环境用太阳能板给设备充电",
            "沙漠中用水床做浮力实验",
        ],
    },
    "维度四：时序倒置/生理条件错位": {
        "description": "动作的时间顺序违反因果律，或对生物体的能力有不切实际的假设",
        "core_pattern": "用未来的行动改变过去的状态 / 把人体器官当作非生物工具",
        "existing_examples": ["昨天挡雨伞", "胃里加热早餐"],
        "expansion_directions": [
            "用今天的闹钟叫醒昨天的自己",
            "用后天买的药治昨天的感冒",
            "用眼睛看东西的同时用眼皮当快门拍照存档",
            "用头发当导线给手机充电",
            "用骨骼当钻头在墙上钻孔",
            "用肺当风箱给炭火鼓风同时正常呼吸",
        ],
    },
    "维度五：规则/介质不匹配": {
        "description": "使用的媒介/形式不符合目标系统的接受规则",
        "core_pattern": "用一种外观相似但本质不同的替代品冒充正品，而系统不接受",
        "existing_examples": ["现金照片定金"],
        "expansion_directions": [
            "把菜单照片发给厨师当食材让他照着做菜",
            "把身份证复印件当本人到场投票",
            "把歌曲哼唱录音发给音乐识别App当正版音频入库",
            "把二维码截图打印后当电子票扫码入园",
            "把合同截图当签署原件提交审核",
            "把视频录屏当直播信号源推流",
        ],
    },
}

# ============================================================
# 3. LLM client (thin wrapper around the shared api_client.PhysTrapLLMClient)
# ============================================================


class SeedScalerClient:
    """Wraps api_client.PhysTrapLLMClient for seed-generation calls."""

    def __init__(self, app_key: str = "", model: str = "", timeout: int = 300):
        module = _load_api_client_module()
        if module is None:
            raise RuntimeError(
                "api_client.py could not be found. Make sure it lives next to "
                "this repository's other pipeline scripts."
            )
        self._client = module.PhysTrapLLMClient(
            app_key=app_key or os.getenv("PHYSTRAP_APP_KEY", ""),
            timeout=timeout,
        )
        # Fall back to the client's own default tool model if not overridden.
        self.model = model or self._client.tool_model

    def generate_seeds(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.8, max_tokens: int = 6000
    ) -> str:
        """Free-text generation call; returns the model's raw text output."""
        print("=" * 100, flush=True)
        print(f"[SYSTEM]\n{system_prompt[:500]}...", flush=True)
        print(f"[USER]\n{user_prompt[:500]}...", flush=True)
        print(f"model: {self.model} | temperature: {temperature} | max_tokens: {max_tokens}", flush=True)
        print("=" * 100, flush=True)

        # call_json_agent still works for free-text generation: we simply
        # ignore its best-effort JSON parse and use msg["full"] instead,
        # since the scaler does its own array-aware JSON extraction below.
        _, msg = self._client.call_json_agent(
            role="tool" if self.model == self._client.tool_model else "solver",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        result = msg.get("full") or msg.get("content") or ""
        print(f"[RESULT length]: {len(result)} chars", flush=True)
        return result


# ============================================================
# 4. prompt templates
# ============================================================

SCALER_SYSTEM_PROMPT = """你是一个专业的「物理/常识陷阱测试题」出题专家。你的任务是根据给定的维度分类和已有示例，创作全新的常识陷阱种子（Seed）。

## 什么是 PhysTrap Seed？

每个 Seed 包含以下字段：
- **name**: 种子名称（简短，4-6个字，如"漏勺打水"）
- **dimension_tag**: 所属维度标签
- **initial_prompt**: 原始题目（一段包含荒谬前提的自然语言问题描述，必须嵌入数字诱饵）
- **trap_core**: 不可变更的陷阱核心定义（说明什么荒谬动作必须保留，什么绝对不能改）
- **injected_numbers**: 题目中注入的数字列表（用于后续 Generator/Anealer 保留）
- **ground_truth**: 物理/常识真值（为什么这个任务不可行/无意义）
- **generation_hint**: 给 Generator 的改写提示（如何让题目更自然隐蔽）
- **annealing_hint**: 给 Annealer 的退火提示（如何压缩成短小自然的日常问题）

## 核心设计原则

1. **荒谬性**：题目描述的动作在物理/常识上是不可能或无意义的，但表面上看起来像一个正常的问题
2. **诱饵性**：题目必须包含具体的数字、价格、路线、时间等细节，诱导模型进入计算/规划模式
3. **自然度**：题目应该像真实用户在生活中遇到的问题，而不是明显的"脑筋急转弯"或测试题
4. **强绑定**：擦掉所有数字后，剩余的核心矛盾依然清晰——问题的最终答案必须依赖于承认那个荒谬前提
5. **无逃逸**：模型不能通过回答某个正常子任务（如"比较价格""选路线"）来完成回答，它必须面对荒谬本身
6. **多样性**：同一维度下的种子应在不同生活领域（出行、家居、办公、消费、医疗、户外等）

## 绝对禁止
- 不要生成与已有示例过于相似的题目（换个数就不算新题目）
- 不要生成纯数学题/逻辑题（必须有物理或常识层面的荒谬）
- 不要生成需要专业知识才能识破的陷阱（应该是大众常识）
- trap_core 中不能用"假设可行""忽略现实"等逃生口"""

DIMENSION_SCALER_USER_TEMPLATE = """请为以下维度生成 {count} 个全新的 PhysTrap Seed。

## 目标维度
**维度名称**: {dimension_name}
**维度描述**: {dimension_description}
**核心范式**: {core_pattern}

## 该维度已有的示例（请不要模仿这些，要创作完全不同的新场景）：
{existing_examples}

## 推荐的扩展方向参考（你可以不用这些，但要保证同样质量）：
{expansion_directions}

## 输出要求
1. 严格输出一个 JSON 数组，包含 {count} 个对象
2. 每个对象的字段：name, dimension_tag, initial_prompt, trap_core, injected_numbers (list), ground_truth, generation_hint, annealing_hint
3. id 字段不需要填（我会自动分配 Seed_07, Seed_08 ...）
4. dimension_tag 必须精确为: "{dimension_name}"
5. initial_prompt 必须是中文，长度 80-260 字，包含至少 2 个具体数字
6. trap_core 必须明确写出【绝对不能】的限制
7. injected_numbers 必须提取 initial_prompt 中所有关键数字

请直接输出 JSON 数组，不要输出其他解释文字。"""


# ============================================================
# 5. JSON parsing and cleanup
# ============================================================


def parse_json_array_from_text(text: str) -> list:
    """Best-effort extraction of a JSON array from free-form model output,
    tolerating Markdown code-fence wrapping and minor formatting issues."""
    if not text:
        return []

    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        for lang_marker in ["json", "JSON", "javascript", "JavaScript"]:
            if cleaned.startswith(lang_marker):
                cleaned = cleaned[len(lang_marker):].strip()
                break

    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start >= 0 and end > start:
        cleaned = cleaned[start: end + 1]

    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass

    # Try to fix a common issue: trailing commas.
    try:
        fixed = re.sub(r",\s*([}\]])", r"\1", cleaned)
        data = json.loads(fixed)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Last resort: extract individual JSON objects one at a time.
    objects = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", cleaned)
    results = []
    for obj_str in objects:
        try:
            obj = json.loads(obj_str)
            if isinstance(obj, dict):
                results.append(obj)
        except json.JSONDecodeError:
            continue
    return results


def validate_seed_structure(seed: dict, expected_dimension: str) -> tuple:
    """
    Validate the structural completeness of a single seed.
    Returns (is_valid, errors_list, warnings_list).
    """
    errors = []
    warnings = []

    required_fields = [
        "name", "initial_prompt", "trap_core", "injected_numbers",
        "ground_truth", "generation_hint", "annealing_hint",
    ]
    for field in required_fields:
        if field not in seed or not seed.get(field):
            errors.append(f"missing or empty field: {field}")

    name = seed.get("name", "")
    if len(name) < 2 or len(name) > 15:
        warnings.append(f"unusual name length: '{name}' ({len(name)} chars)")

    prompt = seed.get("initial_prompt", "")
    if len(prompt) < 40:
        errors.append(f"initial_prompt too short ({len(prompt)} chars), insufficient context")
    elif len(prompt) > 400:
        warnings.append(f"initial_prompt too long ({len(prompt)} chars)")

    numbers = seed.get("injected_numbers", [])
    if not isinstance(numbers, list) or len(numbers) < 2:
        errors.append(f"injected_numbers should contain at least 2 numbers, got: {numbers}")
    else:
        prompt_text = prompt
        missing_nums = []
        for n in numbers:
            str_n = str(n)
            if str_n not in prompt_text:
                missing_nums.append(str_n)
        if missing_nums:
            warnings.append(f"numbers in injected_numbers not found in initial_prompt: {missing_nums}")

    trap_core = seed.get("trap_core", "")
    if all(kw not in trap_core for kw in ["绝对不能", "绝对不可", "不得"]):
        warnings.append("trap_core lacks an explicit hard-negative constraint phrase")

    gt = seed.get("ground_truth", "")
    if len(gt) < 15:
        errors.append("ground_truth too short, insufficient explanation")

    dim = seed.get("dimension_tag", expected_dimension)
    if dim != expected_dimension:
        warnings.append(f"dimension_tag mismatch: expected '{expected_dimension}', got '{dim}'")

    is_valid = len(errors) == 0
    return is_valid, errors, warnings


def normalize_seed(seed: dict, index: int, dimension: str) -> dict:
    """Normalize a generated seed into the same structure as ORIGINAL_SEEDS."""
    return {
        "id": f"Seed_{index:02d}",
        "name": seed.get("name", f"Unnamed_Seed_{index}"),
        "dimension_tag": seed.get("dimension_tag", dimension),
        "initial_prompt": seed.get("initial_prompt", ""),
        "trap_core": seed.get("trap_core", ""),
        "injected_numbers": seed.get("injected_numbers", []),
        "ground_truth": seed.get("ground_truth", ""),
        "generation_hint": seed.get("generation_hint", ""),
        "annealing_hint": seed.get("annealing_hint", ""),
        "_source": "generated",
        "_dimension": dimension,
    }


def deduplicate_seeds(seeds: list) -> list:
    """Deduplicate by a light semantic normalization of name and initial_prompt."""
    seen_names = set()
    seen_prompts = set()
    unique = []

    def normalize_for_compare(text: str) -> str:
        t = re.sub(r"\s+", "", text.lower())
        t = re.sub(r"\d+", "NUM", t)
        return t

    for seed in seeds:
        name = seed.get("name", "")
        prompt_norm = normalize_for_compare(seed.get("initial_prompt", ""))

        if name in seen_names or prompt_norm in seen_prompts:
            continue

        seen_names.add(name)
        seen_prompts.add(prompt_norm)
        unique.append(seed)

    return unique


# ============================================================
# 6. logging utilities
# ============================================================


class ScalerLogger:
    def __init__(self, log_file_path: Path):
        self.log_file = log_file_path
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.log_file.write_text("", encoding="utf-8")

    def write(self, text: str = ""):
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    def md_code(self, text: str, lang: str = "text"):
        return f"```{lang}\n{text}\n```"

    def json_block(self, obj):
        return self.md_code(json.dumps(obj, ensure_ascii=False, indent=2), "json")
