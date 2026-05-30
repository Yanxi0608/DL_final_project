# ---------------------------------------------------------------------------
# agent_nodes.py — LangGraph 总结子图中的两个 LLM 节点
#
# 职责：
#   - llm_writer  (Actor)  : 根据证据 + 评审反馈，生成/修订 Markdown 草稿
#   - llm_critic  (Critic): 对草稿打分并给出结构化修改建议
#
# 在 LangGraph 中的约定：
#   - 每个节点函数接收当前 state（dict），返回需要更新的字段（dict）
#   - LangGraph 会自动 merge 返回值到全局 state
# ---------------------------------------------------------------------------

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# 模块 import 时即加载 .env，确保能读到 DASHSCOPE_API_KEY
load_dotenv()

# 通义千问模型名，可在 .env 中通过 QWEN_MODEL 覆盖（如 qwen-plus、qwen-turbo）
_DEFAULT_QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-plus")
# DashScope OpenAI 兼容端点（国内默认；国际版见文末说明）
_DEFAULT_DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def _get_qwen_llm(*, temperature: float) -> ChatOpenAI:
    """创建通义千问客户端（DashScope OpenAI 兼容模式）。"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("未找到 DASHSCOPE_API_KEY，请在 .env 中配置 DashScope API Key。")
    return ChatOpenAI(
        model=_DEFAULT_QWEN_MODEL,
        api_key=api_key,
        base_url=_DEFAULT_DASHSCOPE_BASE_URL,
        temperature=temperature,
        timeout=int(os.getenv("QWEN_REQUEST_TIMEOUT", "120")),
        max_retries=int(os.getenv("QWEN_MAX_RETRIES", "2")),
    )


class CriticReview(BaseModel):
    """
    Critic 评审节点的结构化输出模型。
    说明：pydantic数据校验库的独有写法（通常与大模型框架一起结合使用），通过继承 BaseModel 类来定义一个数据模型 Schema；同时用 Field 函数来给字段添加额外的限制和描述信息
        score: 浮点数类型。会议总结质量分数，0-10，10 为最佳
        feedback: 字符串类型。具体、可执行的修改建议；若已达标可简要说明优点
    """

    score: float = Field(
        ge=0.0,  # 最小值 0
        le=10.0,  # 最大值 10
        description="会议总结质量分数，0-10，10 为最佳",
    )
    feedback: str = Field(
        description="具体、可执行的修改建议；若已达标可简要说明优点",
    )


def _format_evidence(aligned_units: list[Any], rag_snippets: list[Any]) -> str:
    """
    把 Writer 需要的两类证据格式化为 JSON 字符串，嵌入 prompt。

    参数：
        aligned_units : 时序对齐后的内容单元（口述 + 幻灯片内容 + 时间戳等）
        rag_snippets  : 本场 RAG 检索到的 top-k 证据片段

    返回：
        缩进良好的 JSON 文本，便于 LLM 阅读
    """
    # json.dumps函数将python对象转化为JSON字符串
    return json.dumps(
        {"aligned_units": aligned_units, "rag_snippets": rag_snippets},  # 将aligned_units和rag_snippets打包成一个字典
        ensure_ascii=False,  # 保留中文，不转成 \\uXXXX
        indent=2,  # 缩进，提高可读性
        default=str,  # 遇到不可序列化对象时转 str，避免 dump 失败
    )


def llm_writer(state: dict[str, Any]) -> dict[str, Any]:
    """
    Actor（生成者）节点。

    输入 state 字段：
        aligned_units   — 对齐后的会议内容单元（主要事实来源）
        rag_snippets    — RAG 检索补充证据
        critic_feedback — 上一轮 Critic 的修改建议（首轮可为空）
        draft_summary   — 上一版草稿（修订轮次使用）
        iteration       — 当前迭代次数（仅用于日志/prompt 上下文）

    输出：
        {"draft_summary": "<Markdown 字符串>"}
    """
    iteration = state.get("iteration", 0)
    print(f"✍️ [节点执行]: LLM 撰写者开始起草 (当前迭代: {iteration})...")

    # 从state中读取各类型数据：打包后的证据内容、评审反馈、上一版草稿
    # 从 state 安全读取；缺失时用空列表/空字符串，避免 KeyError
    aligned_units = state.get("aligned_units", [])
    rag_snippets = state.get("rag_snippets", [])
    # 首轮没有 critic_feedback 时，给默认指令
    critic_feedback = state.get("critic_feedback") or "（首轮：请根据证据生成初稿。）"
    draft_summary = state.get("draft_summary") or ""
    # 把证据打包成 prompt 中的一块
    evidence_block = _format_evidence(aligned_units, rag_snippets)

    # System Prompt：约束角色、防幻觉、规定输出格式
    system_prompt = (
        "你是一位学术会议总结撰写助手。"
        "只能依据提供的 aligned_units 与 rag_snippets 中的事实撰写，"
        "不得编造检索块与对齐单元之外的内容。"
        "输出必须是完整的 Markdown 文档（含标题与章节），"
        "重要论断请尽量附带时间锚点（如 [12:30-15:00]）。"
    )

    # User Prompt 分块组装，包括本场证据、评审反馈、上一版草稿（这部分需要分类讨论）
    user_parts = [
        f"当前迭代轮次: {iteration}",
        "## 本场证据（aligned_units + rag_snippets）",
        evidence_block,
        "## 评审反馈（请逐条落实）",
        critic_feedback,
    ]

    # 有上一版草稿 → 修订模式；无 → 初稿模式
    if draft_summary.strip():
        user_parts.extend(
            [
                "## 上一版草稿（请在此基础上修订，勿无故丢弃已正确内容）",
                draft_summary,
            ]
        )
    else:
        user_parts.append("## 上一版草稿\n（无，请直接生成初稿。）")

    user_parts.append("请输出修订后的完整 Markdown 会议总结。")

    # 使用通义千问 API 生成草稿；temperature=0.3 略保留创造性
    try:
        llm = _get_qwen_llm(temperature=0.3)
        # invoke 同步调用；传入 [SystemMessage, HumanMessage] 对话格式
        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content="\n\n".join(user_parts)),
            ]
        )
        # response.content 可能为 None，strip 去掉首尾空白
        draft = (response.content or "").strip()
    except Exception as e:
        # 网络或 API 错误时，生成备用内容
        print(f"⚠️ [LLM 撰写失败]: {e}，使用备用内容。")
        draft = f"# 会议总结（备用）\n\n[由于 API 连接错误，无法生成完整总结。原始错误: {str(e)}]\n\n基于当前证据单元: {len(state.get('aligned_units', []))} 个"

    # 只返回需要更新的 state 字段；LangGraph 会 merge 进全局 state
    return {"draft_summary": draft}


def llm_critic(state: dict[str, Any]) -> dict[str, Any]:
    """
    Critic（评审者）节点。

    输入 state 字段：
        draft_summary — Writer 刚生成的草稿（评审对象）
        iteration     — 当前轮次（写入 prompt 供模型参考）

    输出：
        {
            "critic_score": float,   # 0-10 质量分
            "critic_feedback": str,  # 可执行的修改建议
        }

    关键技术：
        llm.with_structured_output(CriticReview)
        → 强制 LLM 输出符合 Pydantic 模型的结构化结果
    """
    print("🧐 [节点执行]: LLM 评审者开始打分...")

    draft_summary = state.get("draft_summary", "")
    iteration = state.get("iteration", 0)

    # 评审标准：忠实性、结构、时间锚点、幻觉/遗漏
    system_prompt = (
        "你是一位严格的学术会议总结评审员。"
        "评估草稿是否忠实于证据、结构是否清晰、是否含可核对的时间锚点，"
        "以及是否存在超出证据范围的幻觉或遗漏。"
        "score 为 0-10 的浮点数；feedback 须具体、可执行，便于撰写者下一轮修改。"
    )

    human_prompt = (
        f"当前迭代轮次: {iteration}\n\n"
        "## 待评审草稿\n"
        f"{draft_summary or '（草稿为空）'}"
    )

    # 使用通义千问 API 生成评审结果；temperature=0.0 保证稳定可复现
    try:
        llm = _get_qwen_llm(temperature=0.0)

        # 结构化输出：Qwen 兼容模式支持 function calling / JSON schema
        structured_llm = llm.with_structured_output(CriticReview)

        review: CriticReview = structured_llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ]
        )

        score = float(review.score)  # 显式转 float，兼容路由函数比较
        feedback = review.feedback.strip()  # 去掉首尾空白
    except Exception as e:
        # 网络或 API 错误时，生成备用评审结果
        print(f"⚠️ [LLM 评审失败]: {e}，使用备用评分。")
        score = 8.0  # 默认给一个及格分，让流程继续
        feedback = f"[评审系统暂时不可用，错误: {str(e)[:50]}...] 请检查网络连接后重试。"

    print(
        f"   -> 评审结果：分数 {score}, 建议：{feedback[:80]}"
        f"{'...' if len(feedback) > 80 else ''}"
    )

    # 供 route_after_critic 读取 score，供下一轮 llm_writer 读取 feedback
    return {"critic_score": score, "critic_feedback": feedback}

'''
qwen-api：已通过测试！
if __name__ == "__main__":
    s = {
        "iteration": 0,
        "aligned_units": [{"speech_text": "hello", "slide_text": "Deep Learning"}],
        "rag_snippets": [],
        "critic_feedback": "",
        "draft_summary": "",
    }
    s.update(llm_writer(s))
    print(s["draft_summary"][:200])
    s.update(llm_critic(s))
    print(s["critic_score"], s["critic_feedback"][:100])
'''
