# ---------------------------------------------------------------------------
# agent_structure.py — LangGraph 主图与子图编排
#
# 职责：
#   - 条件路由（Writer–Critic 循环终止判定）
#   - 总结子图 build_summarize_subgraph()
#   - 主图 build_main_graph() 与测试入口
# ---------------------------------------------------------------------------

from __future__ import annotations
from typing import Any, Literal, TypedDict
from langgraph.graph import END, StateGraph
from agent_nodes import llm_critic, llm_writer
from tool_nodes import node_preprocess, node_asr, node_ocr, node_align, node_rag_d_self_index
# 假设工具函数已在 tool_nodes.py 中实现；当前用 mock 以便本文件可直接运行

# 这部分也是先mock的，等到完成所有工具函数后再进一步细化
class AlignedUnit(TypedDict, total=False):
    unit_id: str
    t_start: float
    t_end: float
    speech_text: str
    slide_text: str

class ConferenceState(TypedDict, total=False):
    video_path: str
    audio_path: str
    keyframes: list[dict[str, Any]]
    transcript_segments: list[dict[str, Any]]
    ocr_blocks: list[dict[str, Any]]
    aligned_units: list[AlignedUnit]
    rag_snippets: list[dict[str, Any]]
    draft_summary: str
    critic_score: float
    critic_feedback: str
    iteration: int
    max_iterations: int
    score_threshold: float
    errors: list[str]


# ============ 4) 条件路由 ============
def route_after_critic(state: ConferenceState) -> Literal["revise", "accept"]:
    '''
    条件路由函数：输入是全局状态，输出是一个固定的字符串，用于告诉图框家下一步走向哪个节点
    '''
    # 读取核心控制变量：分数、阈值、迭代次数、最大迭代次数
    score = float(state.get("critic_score") or 0.0)
    threshold = float(state.get("score_threshold") or 8.5)
    it = int(state.get("iteration") or 0)
    max_it = int(state.get("max_iterations") or 5)

    # 如果分数达标或达到最大重试次数，则接受稿件
    if score >= threshold or it >= max_it:
        print("✅ [路由决策]: 质量达标或达到最大重试次数，接受稿件！")
        return "accept"

    print("❌ [路由决策]: 分数未达标，打回重写。")
    return "revise"

# 图节点函数：每经历一轮失败，迭代计数器+1，返回值是更新的迭代次数状态
def bump_iteration(state: ConferenceState) -> dict[str, Any]:
    return {"iteration": int(state.get("iteration") or 0) + 1}


# ============ 5) 总结子图 ============

def build_summarize_subgraph():
    '''
    将两个llm节点以及条件路由组装成一个子图，用于实现撰写-评审-打回的循环
    '''
    sg = StateGraph(ConferenceState)    # 初始化一张新的状态图

    sg.add_node("writer", llm_writer)    # 添加撰写节点
    sg.add_node("critic", llm_critic)    # 添加评审节点
    sg.add_node("incr", bump_iteration)    # 添加迭代计数器节点

    sg.set_entry_point("writer")     # 设置子图的入口为writer节点
    sg.add_edge("writer", "critic")     # 增加撰写节点-评审节点的边（撰写节点完成后，流向评审节点）
    sg.add_conditional_edges(    # 增加评审节点-条件路由节点的边（评审节点完成后，流向条件路由节点）
        "critic",
        route_after_critic,     # 调用条件路由函数（根据评审结果决定下一步走向）
        {
            "revise": "incr",     # 如果函数返回 "revise"，走向 "incr" 节点（迭代计数器节点）
            "accept": END,     # 如果函数返回 "accept"，走到 END（跳出子图）
        },
    )
    sg.add_edge("incr", "writer")     # 增加迭代计数器节点-撰写节点的边（迭代计数器节点完成后，流向撰写节点）
    return sg.compile()    # 将画好的图纸编译成一个可执行的组件


# ============ 6) 主图 ============

def build_main_graph():
    '''
    主图流程：视频预处理-语音识别-文字公式识别-时间对齐-rag检索-总结子图
    '''
    g = StateGraph(ConferenceState)

    g.add_node("preprocess", node_preprocess)
    g.add_node("asr", node_asr)
    g.add_node("ocr", node_ocr)
    g.add_node("align", node_align)
    g.add_node("rag_d", node_rag_d_self_index)

    summarize_sg = build_summarize_subgraph()
    g.add_node("summarize_loop", summarize_sg)

    g.set_entry_point("preprocess")
    g.add_edge("preprocess", "asr")
    g.add_edge("asr", "ocr")
    g.add_edge("ocr", "align")
    g.add_edge("align", "rag_d")
    g.add_edge("rag_d", "summarize_loop")
    g.add_edge("summarize_loop", END)

    return g.compile()


# 已完成测试
if __name__ == "__main__":
    

    app = build_main_graph()

    init: ConferenceState = {
        "video_path": "D:\\DL_final_project\\dataset\\videos\\case_02.mp4",
        "iteration": 0,
        "max_iterations": 5,
        "score_threshold": 8.5,
    }

    print("🚀 开始运行 Agent 工作流...\n" + "=" * 40)

    final_state = app.invoke(init)

    print("=" * 40 + "\n🎉 最终生成的总结：\n")
    print(final_state.get("draft_summary", ""))
