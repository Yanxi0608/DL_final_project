# ==========================================
# 阶段一：LangSmith 评测 — 考生包装 (Agent Runner Wrapper)
# 数据集: ConferAI-Eval-Set
# inputs: {"video_filename": "xxx.mp4", "video_filepath": "D:\\...\\xxx.mp4"}
# ==========================================
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_AGENT_DIR = _PROJECT_ROOT / "agent_develop"
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

load_dotenv(_PROJECT_ROOT / ".env")

from agent_structure import build_main_graph

# 全局编译主图，供 LangSmith 批量评测复用
eval_app = build_main_graph()


def target_agent_runner(inputs: dict) -> dict:
    """
    LangSmith evaluate() 的 target 函数。

    从 inputs 读取 video_filepath，映射为 agent 的 video_path；
    运行完成后把 final_state 中 evaluator 后续会用到的字段打平返回。

    返回字段约定（evaluator 会消费这些字段）：
        - summary             : 最终 Markdown 总结（主要评估对象）
        - transcript_segments : ASR 分段结果（list of {text, t_start, t_end}）
        - ocr_blocks          : OCR 内容块（list of {text, t, type}，含公式块）
        - aligned_units       : 时序对齐后的单元（list of AlignedUnit）
        - critic_score        : Agent 内部 Critic 自评分（参考用，不能作为 evaluator）
        - critic_feedback     : Critic 自反馈
        - iteration           : 实际跑了几轮 writer-critic 循环
    """
    video_filename = inputs.get("video_filename", "unknown")
    video_filepath = inputs.get("video_path", "")

    print(f"🎬 [LangSmith 评测] 开始运行: {video_filename}")

    if not video_filepath:
        print(f"⚠️ [LangSmith 评测] 缺少 video_filepath，返回空结果: {video_filename}")
        return _empty_result()

    init = {
        "video_path": video_filepath,
        "iteration": 0,
        "max_iterations": 5,
        "score_threshold": 8.5,
    }

    try:
        final_state = eval_app.invoke(init)
    except Exception as exc:
        print(f"❌ [LangSmith 评测] 执行异常 [{video_filename}]: {exc}")
        return _empty_result()

    print(f"✅ [LangSmith 评测] 运行结束: {video_filename}")

    # 用 agent 真实的 state 字段名取值（这些字段名来自 agent_structure.py 的 ConferenceState）
    return {
        "summary": final_state.get("draft_summary", ""),
        "transcript_segments": final_state.get("transcript_segments", []),
        "ocr_blocks": final_state.get("ocr_blocks", []),
        "aligned_units": final_state.get("aligned_units", []),
        "critic_score": final_state.get("critic_score", 0.0),
        "critic_feedback": final_state.get("critic_feedback", ""),
        "iteration": final_state.get("iteration", 0),
    }


def _empty_result() -> dict:
    """统一的空结果模板，保证返回字段结构稳定。"""
    return {
        "summary": "",
        "transcript_segments": [],
        "ocr_blocks": [],
        "aligned_units": [],
        "critic_score": 0.0,
        "critic_feedback": "",
        "iteration": 0,
    }


if __name__ == "__main__":
    from langsmith import Client, evaluate

    # 1. 初始化 LangSmith 客户端，准备从云端拉取考题
    client = Client()
    dataset_name = "ConferAI-Eval-Set"

    print(f"🔍 [LangSmith] 正在连接云端，尝试获取数据集 [{dataset_name}]...")

    try:
        # 2. 获取该数据集下的所有样本（Examples）
        all_examples = list(client.list_examples(dataset_name=dataset_name))

        if not all_examples:
            print(f"❌ [LangSmith] 错误：云端数据集 '{dataset_name}' 中没有任何数据，请检查上传脚本！")
            sys.exit(1)

        # 3. 🎯 仅切片出第一个样本，实现"只跑一个案例"
        single_test_case = [all_examples[0]]

        target_filename = single_test_case[0].inputs.get("video_filename", "unknown")
        print(f"🚀 [LangSmith] 成功链接！当前开启【单案例冒烟测试】，本次仅测试第一个视频: {target_filename}")
        print("-" * 50)

        # 4. 启动单案例评测
        experiment_results = evaluate(
            target_agent_runner,
            data=single_test_case,
            evaluators=[],
            experiment_prefix="baseline_single_try_fixed",
        )

        print("-" * 50)
        print(f"🎉 [LangSmith] 单案例 [{target_filename}] 测试运行成功结束！")
        print("👉 请登录 LangSmith 网页端验证：")
        print("   1. Datasets & Testing → ConferAI-Eval-Set → 找到 'baseline_single_try_fixed' 实验")
        print("   2. 点开该次运行的 outputs，确认以下字段都有非空内容：")
        print("      - summary（Markdown 文本）")
        print("      - transcript_segments（带时间戳的语音分段列表）")
        print("      - ocr_blocks（OCR 块列表，含 type 字段）")
        print("      - aligned_units（对齐单元列表）")
        print("      - critic_score / iteration（数字）")
        print("   3. 点开 Trace 检查每个节点（preprocess/asr/ocr/align/rag_d/summarize_loop）都正常执行")

    except Exception as e:
        print(f"❌ [LangSmith] 评测启动或运行期间发生异常: {e}")
