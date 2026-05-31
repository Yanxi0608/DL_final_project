# ===========================================================================
# evaluator_utils.py
#
# Evaluator 公共工具：用于 E1 / E2 / E3 等基于 LLM-as-judge 的评估器。
#
# 核心导出函数：
#   judge_with_schema(schema, system_prompt, human_prompt) -> schema 实例
#     传入一个 Pydantic schema 和两段 prompt，返回结构化判断结果。
#     内部自动处理 DashScope 兼容模式的两种调用方式与降级逻辑。
#
# 设计说明：
#   - DashScope 的 OpenAI 兼容模式在 json_object 模式下要求 messages 必须
#     包含 "json" 字样，LangChain 默认路径会触发 400 错误。
#   - 本模块优先走 function_calling，失败后整轮降级到"prompt 要求 JSON +
#     手动解析"模式，对调用方完全透明。
# ===========================================================================
from __future__ import annotations

import json
import os
import re
import traceback
from typing import Type, TypeVar

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

load_dotenv()

# Judge 模型配置（独立于 agent 的 QWEN_MODEL）
_JUDGE_MODEL = os.getenv("QWEN_JUDGE_MODEL", "qwen-max")
_DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 模块级缓存：一旦 function_calling 在本次进程失败，
# 后续整批切换到 JSON 模式，避免每次重试都浪费时间
_USE_JSON_FALLBACK = False

# 用于从 LLM 自由文本中抓出 JSON 块
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


# ----------------------------------------------------------------------------
# LLM 客户端
# ----------------------------------------------------------------------------

def get_judge_llm() -> ChatOpenAI:
    """创建用于评估的 Qwen judge 客户端。temperature=0 保证可复现。"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("未找到 DASHSCOPE_API_KEY，请在 .env 中配置 DashScope API Key。")
    return ChatOpenAI(
        model=_JUDGE_MODEL,
        api_key=api_key,
        base_url=_DASHSCOPE_BASE_URL,
        temperature=0.0,
        timeout=int(os.getenv("QWEN_JUDGE_TIMEOUT", "60")),
        max_retries=int(os.getenv("QWEN_JUDGE_MAX_RETRIES", "2")),
    )


# ----------------------------------------------------------------------------
# 两种 judge 调用模式
# ----------------------------------------------------------------------------

# 类型变量：让 judge_with_schema 的返回值类型跟随传入的 schema
T = TypeVar("T", bound=BaseModel)


def _call_via_function_calling(
    llm: ChatOpenAI,
    schema: Type[T],
    system_prompt: str,
    human_prompt: str,
) -> T:
    """主路径：走 function_calling，绕开 DashScope 的 json 关键词约束。"""
    structured = llm.with_structured_output(schema, method="function_calling")
    return structured.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ])


def _call_via_json_prompt(
    llm: ChatOpenAI,
    schema: Type[T],
    system_prompt: str,
    human_prompt: str,
) -> T:
    """兜底路径：prompt 中要求 json 输出，手动解析。

    注意：instruction 中必须出现 "json" 字样以满足 DashScope 兼容模式约束。
    """
    # 动态从 schema 生成字段说明，让兜底 prompt 自动适配不同的 schema
    field_lines = []
    for name, field in schema.model_fields.items():
        desc = field.description or ""
        field_lines.append(f'  "{name}": <{desc}>')
    schema_hint = "{\n" + ",\n".join(field_lines) + "\n}"

    json_instruction = (
        "\n\n请严格只用以下 json 格式回答，不要任何其他文字（包括 markdown 代码块标记）：\n"
        f"{schema_hint}\n"
        "再次强调：直接输出 json，不要任何说明性文字。"
    )

    msg = llm.invoke([
        SystemMessage(content=system_prompt + json_instruction),
        HumanMessage(content=human_prompt),
    ])
    raw = (msg.content or "").strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    match = _JSON_BLOCK_RE.search(raw)
    if not match:
        raise ValueError(f"judge 未返回有效 JSON。原始输出: {raw[:200]}")

    data = json.loads(match.group(0))
    return schema(**data)


# ----------------------------------------------------------------------------
# 对外暴露的核心函数
# ----------------------------------------------------------------------------

def judge_with_schema(
    schema: Type[T],
    system_prompt: str,
    human_prompt: str,
    *,
    llm: ChatOpenAI | None = None,
) -> T:
    """调用 LLM judge 并返回结构化结果。

    Args:
        schema        : Pydantic 模型，定义 judge 的输出格式
        system_prompt : 系统提示词（角色、判断标准）
        human_prompt  : 用户提示词（具体内容）
        llm           : 可选，复用已有 LLM 客户端；不传则新建

    Returns:
        schema 类型的实例。调用失败时抛出最后一次异常，由调用方决定如何处理。
    """
    global _USE_JSON_FALLBACK

    judge_llm = llm or get_judge_llm()

    # 优先 function_calling；本进程一旦失败过，直接走 JSON 模式
    if not _USE_JSON_FALLBACK:
        try:
            result = _call_via_function_calling(judge_llm, schema, system_prompt, human_prompt)
            if result is not None:
                return result
            # function_calling 在模型未发起 tool call 时会"静默返回 None"（不抛异常）。
            # 这通常是个别 case 触发，不代表整体不可用，所以本次改走 JSON 兜底，
            # 但不翻 _USE_JSON_FALLBACK 全局开关（避免一个 None 把整轮都拖去 JSON 模式）。
            print("⚠️ [judge] function_calling 返回 None（模型未发起 tool call），本次改走 JSON 兜底。")
        except Exception:
            print("⚠️ [judge] function_calling 失败，本次进程切换 JSON 模式。完整错误：")
            traceback.print_exc()
            _USE_JSON_FALLBACK = True

    # 兜底
    return _call_via_json_prompt(judge_llm, schema, system_prompt, human_prompt)


def safe_judge_with_schema(
    schema: Type[T],
    system_prompt: str,
    human_prompt: str,
    *,
    fallback: T,
    llm: ChatOpenAI | None = None,
) -> T:
    """带兜底返回值的安全版本：所有异常都被吞掉，返回 fallback。

    适合在 evaluator 主流程里使用 —— 不希望某一次 judge 失败导致整轮评测崩溃。
    """
    try:
        result = judge_with_schema(schema, system_prompt, human_prompt, llm=llm)
        return result if result is not None else fallback
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        print(f"⚠️ [judge] 解析失败: {exc}")
        return fallback
    except Exception as exc:
        print(f"⚠️ [judge] 调用失败: {type(exc).__name__}: {str(exc)[:200]}")
        traceback.print_exc()
        return fallback

# ----------------------------------------------------------------------------
# LangSmith 结果格式包装
# ----------------------------------------------------------------------------

# LangSmith EvaluationResult 接受的标准字段(新版严格校验)
_LANGSMITH_ALLOWED_KEYS = {
    "key", "score", "value", "comment", "correction", "evaluator_info",
    "source_run_id", "target_run_id", "extra",
}


def to_langsmith_result(result: dict) -> dict:
    """把 evaluator 返回的字典转成 LangSmith 接受的格式。

    自定义字段(point_details / rubric_score 等)会被搬到 evaluator_info 下,
    在 LangSmith UI 上仍然可见,只是访问路径不同。

    Args:
        result : evaluator 原始返回字典,必须含 'key'

    Returns:
        合规的 EvaluationResult 字典
    """
    if not isinstance(result, dict) or "key" not in result:
        return result  # 让 LangSmith 自己抛错

    standard = {k: v for k, v in result.items() if k in _LANGSMITH_ALLOWED_KEYS}
    extra = {k: v for k, v in result.items() if k not in _LANGSMITH_ALLOWED_KEYS}

    if extra:
        # 把所有非标准字段合并到 evaluator_info(如已存在则不覆盖原有)
        existing_info = standard.get("evaluator_info") or {}
        if not isinstance(existing_info, dict):
            existing_info = {"_original": existing_info}
        existing_info.update(extra)
        standard["evaluator_info"] = existing_info

    return standard