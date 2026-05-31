from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from evaluator_utils import get_judge_llm, safe_judge_with_schema, to_langsmith_result

# ===========================================================================
# E1: Must-Have Points Recall —— 关键点召回率
#
# 目的：对 agent 生成的会议总结打一个"关键点覆盖率"的分,告诉你 summary 漏没漏掉那些标注的"必须涵盖的要点"。
# 输入：agent 生成的 MD 总结和 dataset 标注的关键点列表
# 输出：score核心指标召回率，comment人类可读的简短结论（覆盖程度），point_detials每个店的逐条诊断（是否覆盖+judge给的理由），用于失败归因

#must_have_points = ["要点A", "要点B", "要点C"]
#                          ↓
#         对每一个 point,独立问一次 LLM judge:
#         "summary 里覆盖了这个 point 吗?(yes/no + 理由)"
#                          ↓
#         统计 yes 的数量 / 总数 = 召回率
#                          ↓
#         返回给 LangSmith,框架在 UI 上画图、做聚合
# ===========================================================================

# ======================数据契约（Pydantic Schema），强制LLM输出格式================
class PointJudgement(BaseModel):
    coverage_level: Literal["full", "partial", "none"] = Field(
        description="该关键点在总结中的覆盖程度，三档之一。"
                    "full = 总结明确、完整地陈述了该要点的核心信息；若要点含具体内容（数值/方法名/前提条件/结论方向），这些具体点也都说到了。"
                    "partial = 只覆盖了要点的一部分，或提到主题却没点出核心信息，或表述含糊、需读者脑补才能对应。"
                    "none = 完全没提到，或仅主题相关而未触及该要点本身。",
    )
    reasoning: str = Field(
        description="给出判断的简短理由（≤80字），指出总结中对应或缺失的部分。",
    )

# ================== 单点评估 =====================

_COVERAGE_SYSTEM = (
    "你是一位严格的学术内容评估员，判断时宁严勿宽。\n"
    "你的任务：判断给定的【关键点】在【会议总结】中的覆盖程度，分 full / partial / none 三档。\n"
    "判断标准（从严）：\n"
    "- full：总结明确、完整地陈述了该关键点的核心信息；如果关键点包含具体内容（数值、方法名、前提条件、结论方向），总结必须把这些具体点也说出来才算 full。\n"
    "- partial：只覆盖了关键点的一部分，或只提到主题而没点出核心信息，或表述含糊、需要读者脑补才能对应上。\n"
    "- none：完全没提到，或只是主题相关而未触及该关键点本身。\n"
    "重要原则：只要核心信息没有被清楚地说出来就不要给 full；不要因为总结整体写得好或主题相关就放松对这一条的要求；只依据总结的实际文字判断，不要替它补充它没有写出来的内容。"
)

# full / partial / none → 分数（从严：partial 大幅折扣，可在此调松紧）
_COVERAGE_SCORE = {"full": 1.0, "partial": 0.4, "none": 0.0}


def _judge_single_point(judge_llm, point: str, summary: str) -> PointJudgement:
    human_prompt = (
        f"## 关键点（需要被覆盖的要点）\n{point}\n\n"
        f"## 会议总结\n{summary or '（总结为空）'}\n\n"
        "请判断该关键点是否在总结中被覆盖，并给出简短理由。"
    )
    fallback = PointJudgement(
        coverage_level="none",
        reasoning="[judge 调用失败，默认判为未覆盖]",
    )
    return safe_judge_with_schema(
        PointJudgement, _COVERAGE_SYSTEM, human_prompt,
        fallback=fallback, llm=judge_llm,
    )


# ================== LangSmith evaluator 主函数 ==========================

def must_have_points_recall(run, example) -> dict[str, Any]:
    summary = (getattr(run, "outputs", None) or {}).get("summary", "") or ""
    must_have_points = (getattr(example, "outputs", None) or {}).get("must_have_points", []) or []

    # 边界情况
    if not must_have_points:
        return {
            "key": "must_have_recall",
            "score": None,
            "comment": "该 case 缺少 must_have_points 标注，跳过 E1。",
        }
    if not summary.strip():
        return to_langsmith_result({
            "key": "must_have_recall",
            "score": 0.0,
            "value": 0.0,
            "comment": f"summary 为空，{len(must_have_points)} 个关键点全部未覆盖。",
            "point_details": [
                {"point": p, "coverage_level": "none", "point_score": 0.0, "reasoning": "summary 为空"}
                for p in must_have_points
            ],
        })

    # 逐点判断（full=1 / partial=0.4 / none=0，从严加权）
    judge_llm = get_judge_llm()
    details: list[dict[str, Any]] = []
    total = 0.0

    for idx, point in enumerate(must_have_points, start=1):
        j = _judge_single_point(judge_llm, point, summary)
        pts = _COVERAGE_SCORE.get(j.coverage_level, 0.0)
        total += pts
        details.append({
            "point_index": idx,
            "point": point,
            "coverage_level": j.coverage_level,
            "point_score": pts,
            "reasoning": j.reasoning,
        })

    # 汇总
    recall = total / len(must_have_points)
    not_full = [d["point"] for d in details if d["coverage_level"] != "full"]
    comment = (
        f"加权覆盖 {recall:.2f}（full=1/partial=0.4/none=0，共 {len(must_have_points)} 点）。"
        + (f" 未完整覆盖：{'; '.join(not_full)}" if not_full else " 全部完整覆盖。")
    )


    return to_langsmith_result({
        "key": "must_have_recall",  # LangSmith UI上的指标明
        "score": recall,   # 0-1打分
        "value": recall,   
        "comment": comment,   # 人类可读结论
        "point_details": details,   # 自定义字段作为metadata存在下来，用于后续归因分析
    })


# ===========================================================================
# E2: Faithfulness —— 忠实度 / 幻觉检测
#
# 评估目标：
#   验证 summary 中的每个事实点是否能在 gold_transcription + gold_ocr 里找到
#   支持，检测 agent 是否出现幻觉。这是学术场景最关键的指标之一。
#
# 方法：
#   两阶段 LLM judge：
#     Stage 1: 把 summary 拆成 5-15 个原子事实点（claim decomposition）
#     Stage 2: 对每个事实点，让 judge 在证据中查找支持
#
# 输入：
#   - run.outputs.summary             : agent 生成的 Markdown 总结
#   - example.outputs.gold_transcription : 参考 ASR 文本
#   - example.outputs.gold_ocr           : 参考 OCR 文本（含 LaTeX 公式）
#
# 输出：
#   - score / value : faithfulness ∈ [0, 1]，被证据支持的事实占比
#   - comment       : 人类可读结论
#   - claim_details : 每个事实点的逐条判断
# ===========================================================================


# =============== Stage 1: 事实点拆分的 schema -======================

class ClaimList(BaseModel):
    claims: list[str] = Field(
        description="从总结中拆解出的原子事实点列表。每个事实点应是一个独立、"
                    "可验证的陈述句（如：'X 方法的核心思想是 Y'、'数据集包含 N 个样本'）。"
                    "数量建议 5-15 条。仅拆解关于事实/方法/数据/结论的陈述，"
                    "忽略纯背景介绍、修辞性语句、章节标题。",
    )


# ================= Stage 2: 单个事实点判断的 schema =======================

class ClaimVerification(BaseModel):
    is_supported: bool = Field(
        description="该事实点是否能在所给证据（gold_transcription + gold_ocr）中找到支持。"
                    "支持 = 证据中存在与该事实点语义一致或明确蕴含的内容（措辞可不同）。"
                    "不支持 = 证据中完全没提及，或与证据矛盾。",
    )
    reasoning: str = Field(
        description="给出 is_supported 判断的简短理由（≤80字）。"
                    "若 True，请简要指出证据中支持该事实点的部分；"
                    "若 False，请说明该事实点是无依据的还是与证据矛盾。",
    )


# ================= Stage 1: 拆事实点 ======================

_CLAIM_DECOMPOSE_SYSTEM = (
    "你是一位学术内容分析员。"
    "你的任务是把一份会议总结拆解为若干个【原子事实点】(atomic claims)。\n"
    "原子事实点的标准：\n"
    "- 是一个独立、可被验证的陈述（如方法、数据、结论、数值、引用关系等）\n"
    "- 不要包含修辞、概述性总括或章节标题\n"
    "- 不要过度拆分（如'X方法包含A、B、C'就是一条，不必拆成三条）\n"
    "- 数量建议 5-15 条，少则太粗，多则啰嗦。"
)


def _decompose_claims(judge_llm, summary: str) -> list[str]:
    human_prompt = (
        f"## 需要拆解的会议总结\n{summary}\n\n"
        "请输出该总结中可验证的原子事实点列表。"
    )
    fallback = ClaimList(claims=[])
    result = safe_judge_with_schema(
        ClaimList, _CLAIM_DECOMPOSE_SYSTEM, human_prompt,
        fallback=fallback, llm=judge_llm,
    )
    # 去重、去空、去过短的噪声项
    seen = set()
    cleaned = []
    for c in result.claims:
        c = (c or "").strip()
        if len(c) < 8 or c in seen:
            continue
        seen.add(c)
        cleaned.append(c)
    return cleaned


# ================= Stage 2: 逐点核对 ====================

_VERIFY_SYSTEM = (
    "你是一位严格的事实核对员。"
    "你的任务是判断给定的【事实点】是否能在【证据】中找到支持。\n"
    "判断标准：\n"
    "- 支持 = 证据中存在与该事实点语义一致或明确蕴含的内容（措辞可不同）。\n"
    "- 不支持 = 证据中完全没提及该事实点，或与证据明显矛盾。\n"
    "证据可能包含口语化表达、不完整句子、LaTeX 公式片段，请综合理解。"
    "只基于所给证据判断，不要使用你自己的先验知识。"
)


def _verify_single_claim(judge_llm, claim: str, evidence: str) -> ClaimVerification:
    human_prompt = (
        f"## 事实点\n{claim}\n\n"
        f"## 证据（口述文本 + 幻灯片识别文本拼接）\n{evidence}\n\n"
        "请判断该事实点能否在证据中找到支持，并给出简短理由。"
    )
    fallback = ClaimVerification(
        is_supported=False,
        reasoning="[judge 调用失败，默认判为不支持]",
    )
    return safe_judge_with_schema(
        ClaimVerification, _VERIFY_SYSTEM, human_prompt,
        fallback=fallback, llm=judge_llm,
    )


# =============== 证据拼接 ==================

def _build_evidence(gold_transcription: str, gold_ocr: str) -> str:
    """把口述与幻灯片证据拼成一份给 judge 看的纯文本。

    分块标记让 judge 知道来源，便于做更准确的语义对齐。
    """
    parts = []
    if gold_transcription.strip():
        parts.append(f"【口述内容（ASR）】\n{gold_transcription.strip()}")
    if gold_ocr.strip():
        parts.append(f"【幻灯片文字与公式（OCR）】\n{gold_ocr.strip()}")
    return "\n\n".join(parts) if parts else "（证据为空）"


# ================ LangSmith evaluator 主函数 =======================

def faithfulness_score(run, example) -> dict[str, Any]:
    """LangSmith evaluator 接口。"""
    outputs = getattr(run, "outputs", None) or {}
    summary = outputs.get("summary", "") or ""

    gt = getattr(example, "outputs", None) or {}
    gold_transcription = gt.get("gold_transcription", "") or ""
    gold_ocr = gt.get("gold_ocr", "") or ""

    # 边界：证据缺失，无法评估
    if not gold_transcription.strip() and not gold_ocr.strip():
        return {
            "key": "faithfulness",
            "score": None,
            "comment": "该 case 缺少 gold_transcription / gold_ocr 证据，跳过 E2。",
        }

    # 边界：summary 为空
    if not summary.strip():
        return {
            "key": "faithfulness",
            "score": None,
            "comment": "summary 为空，无法做 faithfulness 评估。",
        }

    judge_llm = get_judge_llm()
    evidence = _build_evidence(gold_transcription, gold_ocr)

    # Stage 1: 拆事实点
    claims = _decompose_claims(judge_llm, summary)
    if not claims:
        return {
            "key": "faithfulness",
            "score": None,
            "comment": "无法从 summary 中拆出任何原子事实点（可能 summary 过短或 judge 调用失败）。",
        }

    # Stage 2: 逐点核对
    details: list[dict[str, Any]] = []
    supported_count = 0

    for idx, claim in enumerate(claims, start=1):
        v = _verify_single_claim(judge_llm, claim, evidence)
        if v.is_supported:
            supported_count += 1
        details.append({
            "claim_index": idx,
            "claim": claim,
            "is_supported": v.is_supported,
            "reasoning": v.reasoning,
        })

    # 汇总
    faithfulness = supported_count / len(claims)
    unsupported = [d["claim"] for d in details if not d["is_supported"]]
    comment = (
        f"支持 {supported_count}/{len(claims)} 个事实点。"
        + (f" 疑似幻觉/无依据：{len(unsupported)} 条。" if unsupported else " 全部有据。")
    )

    return to_langsmith_result({
        "key": "faithfulness",
        "score": faithfulness,
        "value": faithfulness,
        "comment": comment,
        "claim_details": details,
    })


# ===========================================================================
# E3: Summary Quality vs Ground Truth —— 总结整体语义质量
#
# 评估目标：
#   评估 agent 生成的 summary 和参考总结（ground_truth_summary）在
#   核心主旨、关键论点上的贴合度。提供一个端到端的整体质量评分。
#
# 方法：
#   LLM-as-judge with rubric。用 5 分制，每档有明确锚点描述，
#   避免 judge 自由发挥。最终归一化到 [0, 1]。
#
# 输入：
#   - run.outputs.summary                  : agent 生成的 Markdown 总结
#   - example.outputs.ground_truth_summary : 参考总结（简短，100-200 字）
#
# 输出：
#   - score / value     : 归一化分数 ∈ [0, 1]
#   - rubric_score      : 原始 1-5 分（在 metadata 里保留，便于查看）
#   - comment           : judge 给出的简短理由
# ===========================================================================

# ================ Rubric 评分 schema ================

class RubricScore(BaseModel):
    score: int = Field(
        ge=1, le=5,
        description="基于 rubric 给出 1-5 的整数分（评分从严，5 分应当罕见）。"
                    "5 = 参考总结的每个关键论点都被准确涵盖、表述精确、无遗漏无错配；"
                    "4 = 主旨与绝大多数关键论点到位，仅一处次要遗漏或轻微不精确；"
                    "3 = 抓到主旨但有明显的关键论点遗漏/不准确/重点偏移；"
                    "2 = 仅部分相关，关键论点大量缺失或被错误表达；"
                    "1 = 严重偏离主题或与参考总结几乎无关联。",
    )
    reasoning: str = Field(
        description="给出该评分的简短理由（≤120字），"
                    "指出 summary 相对参考总结的优点与不足。",
    )


_RUBRIC_SYSTEM = (
    "你是一位资深、严格的学术会议总结质量评估专家，评分倾向保守。\n"
    "你的任务：对比【agent 生成的总结】和【参考总结】，从核心主旨、关键论点覆盖、"
    "表述准确性三个维度，给出 1-5 分的整体质量评分。\n\n"
    "评分标准（从严，5 分应当罕见）：\n"
    "  5 分：参考总结里的每一个关键论点都被准确涵盖，关键概念/数值/方法表述精确，"
    "没有遗漏、没有与参考不符的表述，也没有把次要内容当主线\n"
    "  4 分：主旨与绝大多数关键论点到位，但有一处次要遗漏或轻微不精确，不影响整体理解\n"
    "  3 分：抓到主旨，但存在明显的关键论点遗漏、或有不准确/含糊的表述、或重点偏移\n"
    "  2 分：仅部分相关，关键论点大量缺失或被错误表达\n"
    "  1 分：严重偏离主题，或与参考总结几乎无关联\n\n"
    "扣分硬规则：\n"
    "- 参考总结中任何一个关键论点未被涵盖 → 最高给 3 分\n"
    "- 出现与参考总结不一致的事实/数值/方法表述 → 最高给 3 分\n"
    "- 关键概念表述含糊、需要读者脑补 → 至少扣 1 分\n\n"
    "判断原则：\n"
    "- 语义贴合即可，不要求逐字一致；公式用 LaTeX 还是自然语言表述不同，不算错\n"
    "- 不要因为 agent 总结更长或更短本身而加减分，只看关键内容是否准确到位\n"
    "- 只比较两份总结之间的贴合度，不要用你自己的先验知识做事实核查（那是其它评估器的职责）"
)


def _build_human_prompt(summary: str, gt_summary: str) -> str:
    return (
        f"## Agent 生成的总结\n{summary}\n\n"
        f"## 参考总结（Ground Truth）\n{gt_summary}\n\n"
        "请基于 rubric 给出 1-5 分的整体质量评分，并简述理由。"
    )


# =================== LangSmith evaluator 主函数 ==================

def summary_quality_score(run, example) -> dict[str, Any]:
    """LangSmith evaluator 接口。"""
    outputs = getattr(run, "outputs", None) or {}
    summary = outputs.get("summary", "") or ""

    gt = getattr(example, "outputs", None) or {}
    gt_summary = gt.get("ground_truth_summary", "") or ""

    # 边界：缺少 GT
    if not gt_summary.strip():
        return {
            "key": "summary_quality",
            "score": None,
            "comment": "该 case 缺少 ground_truth_summary 标注，跳过 E3。",
        }

    # 边界：summary 为空
    if not summary.strip():
        return {
            "key": "summary_quality",
            "score": 0.0,
            "value": 0.0,
            "rubric_score": 1,
            "comment": "summary 为空，给最低分 1/5。",
        }

    # judge 评分
    judge_llm = get_judge_llm()
    human_prompt = _build_human_prompt(summary, gt_summary)
    fallback = RubricScore(score=1, reasoning="[judge 调用失败，默认最低分]")
    result = safe_judge_with_schema(
        RubricScore, _RUBRIC_SYSTEM, human_prompt,
        fallback=fallback, llm=judge_llm,
    )

    # 1-5 → [0, 1]，线性映射：(score - 1) / 4
    normalized = (result.score - 1) / 4

    return to_langsmith_result({
        "key": "summary_quality",
        "score": normalized,
        "value": normalized,
        "rubric_score": result.score,
        "comment": f"评分 {result.score}/5。{result.reasoning}",
    })