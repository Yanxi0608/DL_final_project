# ===========================================================================
# run_full_eval.py
#
# LangSmith 完整评测入口:挂上 E1~E5 五个 evaluator,跑完 10 个 case。
#
# 前置条件:
#   1. 已重命名两个 evaluator 文件:
#      - llm_judge_evaluators.py  (含 E1/E2/E3)
#      - code_evaluators.py       (含 E4/E5)
#   2. agent_testrun.py 中的 target_agent_runner 已修复字段名
#   3. .env 配置好 LANGSMITH_API_KEY / DASHSCOPE_API_KEY 等
#   4. ConferAI-Eval-Set 数据集已上传到 LangSmith 云端
#
# 输出:
#   - 控制台进度日志
#   - LangSmith 网页端可见的 experiment(默认前缀 "baseline_full")
# ===========================================================================
from __future__ import annotations

import sys
from pathlib import Path
from dotenv import load_dotenv

# ---------- 路径配置 ----------

_PROJECT_ROOT = Path(__file__).resolve().parents[1]   # D:\DL_final_project
_AGENT_EVAL_DIR = Path(__file__).resolve().parent     # D:\DL_final_project\agent_eval
_AGENT_DEVELOP_DIR = _PROJECT_ROOT / "agent_develop"

# 让 Python 能找到 agent_develop 里的 agent_structure 等模块
for p in (str(_AGENT_EVAL_DIR), str(_AGENT_DEVELOP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

load_dotenv(_PROJECT_ROOT / ".env")


# ---------- 导入 evaluators 与 target ----------

# 5 个 evaluator
from llm_judge_evaluators import (
    must_have_points_recall,   # E1
    faithfulness_score,         # E2
    summary_quality_score,      # E3
)
from code_evaluators import (
    formula_coverage,           # E4
    asr_quality,                # E5
)

# Agent runner(target 函数,LangSmith 调用它跑出 outputs)
# 假设 agent_testrun.py 在 agent_eval 同级或 agent_develop 里;调整路径如有需要
from agent_testrun import target_agent_runner


# ---------- 主入口 ----------

def main():
    from langsmith import Client, evaluate

    client = Client()
    dataset_name = "ConferAI-Eval-Set-new"
    experiment_prefix = "baseline_clean"

    # 1. 连接数据集
    print(f"🔍 连接 LangSmith,获取数据集 [{dataset_name}]...")
    all_examples = list(client.list_examples(dataset_name=dataset_name))
    if not all_examples:
        print(f"❌ 数据集 '{dataset_name}' 为空,请检查上传脚本。")
        sys.exit(1)

    print(f"✅ 找到 {len(all_examples)} 个 case,准备开始完整评测。")
    print(f"📊 挂载 evaluator: E1 关键点召回 / E2 忠实度 / E3 整体质量 / "
          f"E4 公式覆盖率 / E5 ASR 质量")
    print(f"⏱  预计耗时: 每个 case 约 2-5 分钟(agent 跑 + E2 双阶段 LLM judge),"
          f"10 个 case 总耗时约 30-60 分钟。")
    print("-" * 60)

    # 2. 启动评测
    experiment_results = evaluate(
        target_agent_runner,
        data=all_examples,
        evaluators=[
            must_have_points_recall,   # E1
            faithfulness_score,         # E2
            summary_quality_score,      # E3
            formula_coverage,           # E4
            asr_quality,                # E5
        ],
        experiment_prefix=experiment_prefix,
        # 不并发,避免 DashScope 限流;若想加速,改为 max_concurrency=2
        max_concurrency=1,
        # 给实验加点 metadata,后期对比版本时方便辨识
        metadata={
            "stage": "baseline",
            "evaluators_count": 5,
            "agent_version": "v1.0",
        },
    )

    print("-" * 60)
    print(f"🎉 完整评测结束!")
    print(f"👉 请打开 LangSmith 网页端查看结果:")
    print(f"   Datasets & Testing → {dataset_name} → 找到 '{experiment_prefix}-*' 实验")
    print(f"   按 score 排序、查看各 case 的 trace 与 evaluator 详情")


if __name__ == "__main__":
    main()