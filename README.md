# 学术会议总结智能体 ConferAI

# LangGraph开发 + LangSmith运维

## 一、业务背景说明

#### 1. 选题背景与核心痛点

**选题背景**：在数据科学与人工智能领域，前沿的学术会议、行业峰会和技术讲座（如 NeurIPS、ICML 或各大企业的技术发布会）是获取最新算法、基准测试和技术趋势的核心渠道。然而，这些高质量的“知识”通常以 **录屏视频、口述音频、幻灯片（PPT）画面** 的多模态形式分散存在，导致信息极难被快速检索和沉淀。

在日常的学习和科研工作中，我们发现处理这些多模态学术资源存在三个核心痛点：

- 时间成本极高： 完整消化一场 60–90 分钟的技术讲座，通常需要人工花费 2–3 小时去反复回看、截图、做笔记，效率低下。
- 单模态信息孤岛： 仅靠语音转文字（ASR）会漏掉幻灯片上的公式和核心架构图；仅靠看 PPT（OCR）又会漏掉讲者的口头补充和解释。
- 缺乏时序对齐： 市面上的普通摘要工具无法将“讲者说的话”和“当时播放的幻灯片”在时间轴上精准匹配，导致下游模型生成的总结容易张冠李戴，产生信息幻觉。

#### 2. 解决方案概览与技术架构

为了解决上述问题，本项目设计并实现了一个端到端的基于LangGraph框架的自动化多模态学术会议总结智能体（agent），通过输入.mp4学术会议视频，智能体可以自动提取语音、公式与文字并实现时序对齐，最终生成.markdown格式的会议摘要。从数据科学的视角来看，这是一个典型的非结构化数据处理与分析过程：

- 数据抽取层 (Extract)：利用 Whisper 将音频转化为带有精确时间戳的文本片段；利用 PaddleOCR 抽取幻灯片关键帧中的文字、公式和图表标题。
- 数据对齐层 (Transform)：开发时序对齐算法（Aligner），在特定的时间窗口内将口述文本与幻灯片内容融合，封装为统一的知识单元（AlignedUnit），确保图文一一对应。
- 智能生成层 (Analyze)：基于 LangGraph 构建了 Writer–Critic（协同评审）工作流。写作者 Agent 根据对齐后的数据生成摘要初稿，评审者 Agent 对照原始证据进行无幻觉打分和修正，确保最终输出的 Markdown 摘要具备高准确率与可追溯性（用户可点击时间锚点跳转原视频）。

#### 3. 项目价值与实际效果

- 量化的高效提升： 针对一场 60 分钟的技术讲座，传统人工梳理需要 2 个多小时，而agent在数分钟内即可完成全套处理，将我们从繁琐的“逐帧倍速观看”中解放出来，专注于高价值的方法论分析。
- 非结构化媒体的资产化： 系统的输出不仅仅是一篇文本，而是带时间戳、可直接检索的结构化数据库。未来可以轻松接入团队的知识库，支持“哪个讲座提到了某种优化方法”的高效检索。
- 算法质量可控： 区别于单次直接调用大模型（Demo级应用），本项目引入了 LLM-as-a-Judge 评估框架。在我们的测试集上，Writer-Critic 闭环相比单次生成，信息召回率和防幻觉率均有显著提升，验证了多智能体协同在解决复杂多模态任务中的实际价值。

---

## 二、基于LangGraph框架的agent开发部分

### 1. 感知与预处理工具

- **职责**：接收用户上传的会议录屏；分离音轨；按镜头/翻页抽取关键帧并记录时间戳。
- **产出**：带时间的音频文件、关键帧图像序列、预处理元数据（分辨率、时长等）。
- **意义**：为「听见什么」与「看见什么」提供可对齐的原始证据。

### 2. 听觉解析工具（ASR）

- **职责**：长语音转写，输出 **带起止时间的文本片段**。
- **产出**：`TranscriptSegment[]`（文本 + `start_sec` / `end_sec`）。
- **意义**：总结的语言主干来自口述内容；时间戳是后与画面、章节对齐的基础。

### 3. 视觉解析工具（OCR）

- **职责**：对关键帧进行文字与（可选）公式识别。
- **产出**：`OCRBlock[]`（文本或 LaTeX 候选 + 对应帧时间或帧 ID）。
- **意义**：补全幻灯片上的定义、公式、图表标题等口头未完全覆盖的信息。

### 4. 时序对齐与结构化聚合（Aligner）

- **职责**：将同一时间窗内的 ASR 片段与 OCR 块合并为 **内容单元（Content Unit）**。
- **产出**：`AlignedUnit[]`，每条绑定 `**time_range` + 口述文本 + 幻灯片文本 + 证据字段**。
- **意义**：作为 **RAG-D「本场自索引」** 的索引粒度：检索块天然带 **视频时间锚点**，便于报告与录像对齐。

### 5. 编排核心（LangGraph 主图）

- **职责**：用状态图串联：**预处理 → ASR → OCR → 对齐 → RAG-D → 总结子图（Writer–Critic 循环）** → 落盘报告。
- **产出**：最终 Markdown/PDF、各阶段 artifact（如 `runs/`）。
- **意义**：可观测阶段、可分支扩展（如自适应 OCR、外部论文 RAG-B），区别于单次脚本。

### 6. 总结子图：撰写智能体 + 评审智能体（核心）

- **Writer**：输入对齐单元、RAG 检索证据、上一轮 **Critic 的修改建议**；输出结构化会议总结草稿（Markdown）。
- **Critic**：对照证据约束评审草稿（如「不得引入检索块与 aligned_units 之外的事实」）；输出 **数值分数 + 分项意见 + 可执行修改建议**。
- **终止条件**：`score >= score_threshold`，或达到 `max_iterations`（避免无限循环与费用失控）。
- **意义**：体现 **生成–反思–修订** 的 Agent 行为，区别于「单次调用一个大模型」。

```python
# framework_langgraph_conference_agent.py
from __future__ import annotations
from typing import Any, Literal, TypedDict
from langgraph.graph import END, StateGraph

# ============ 1) 共享状态 (Global State) ============
# 【核心概念】：这是整个系统的“黑板”或“共享内存”，在 LangGraph 中，所有的节点（Node）都接收这个 State 作为输入，并返回一个字典，LangGraph 会自动将返回的字典更新到这个 State 中。

class AlignedUnit(TypedDict, total=False):
    """定义对齐后每个数据块的结构，方便统一管理"""
    unit_id: str
    t_start: float
    t_end: float
    speech_text: str
    slide_text: str

class ConferenceState(TypedDict, total=False):
    """全局状态定义"""
    # 1.初始输入：视频与音频路径
    video_path: str
    audio_path: str
    
    # 2. 预处理与解析产物（各模块的输出结果）
    keyframes: list[dict[str, Any]]
    transcript_segments: list[dict[str, Any]]
    ocr_blocks: list[dict[str, Any]]
    aligned_units: list[AlignedUnit]
    rag_snippets: list[dict[str, Any]]

    # 3. Actor-Critic (质询循环) 专属的状态变量
    draft_summary: str      # 当前生成的草稿
    critic_score: float     # 评审给出的分数
    critic_feedback: str    # 评审给出的修改建议
    iteration: int          # 当前已循环的次数
    max_iterations: int     # 最大允许的循环次数（防止死循环）
    score_threshold: float  # 及格线分数

    # 4. 错误记录
    errors: list[str]


# ============ 2) 工具节点占位 (Tool Nodes) ============
# 【核心概念】：节点（Node）就是一个普通的 Python 函数，它的输入是当前最新的 State，输出是一个字典，告诉 LangGraph 需要“更新”哪些状态变量。
# 【核心目标】：将原始视频转化为可以直接给LLM的结构化文本和时间戳；其中所有输入都是通过读取State来获取，输出最终都要返回dict

def node_preprocess(state: ConferenceState) -> dict[str, Any]:
    # 主要任务：将视频分离出音轨；抽取关键帧
    # 输入：视频路径；输出：音频路径、关键帧对应时间戳
    print("🎬 [节点执行]: 预处理视频...")
    return {"audio_path": "mock_audio.wav", "keyframes": [{"t": 0}]}

def node_asr(state: ConferenceState) -> dict[str, Any]:
    # 主要任务：长语音撰写，输出带起止时间的文本片段
    # 输入：音频路径；输出：transcript_segments (包含文本与时间戳的列表)
    print("🎧 [节点执行]: 语音识别...")
    return {"transcript_segments": [{"text": "hello"}]}

def node_ocr(state: ConferenceState) -> dict[str, Any]:
    print("👁️ [节点执行]: 图像文字与公式识别...")
    return {"ocr_blocks": [{"text": "Deep Learning"}]}

def node_align(state: ConferenceState) -> dict[str, Any]:
    # 主要任务：将上述抽取的元素按照时间戳打包成一个个结构化的内容单元 (AlignedUnit)
    # 输入:transcript_segments文本与时间戳列表；输出 (返回 Dict): aligned_units
    print("🔗 [节点执行]: 视听时间戳对齐...")
    return {"aligned_units": [{"speech_text": "hello", "slide_text": "Deep Learning"}]}


# ============ 3) 两个 LLM 角色：撰写 vs 评审 ============
# 【核心概念】：Actor-Critic 双模型博弈。一个负责写，一个负责挑刺。

def llm_writer(state: ConferenceState) -> dict[str, Any]:
    """Actor (生成者) 节点：负责根据证据和上一轮的反馈写文章"""
    # 输入：aligned_units (素材), critic_feedback (如果不是第一轮，则读取反馈), draft_summary (上一版的草稿)；输出：draft_summary (更新后的草稿)
    print(f"✍️ [节点执行]: LLM 撰写者开始起草 (当前迭代: {state.get('iteration', 0)})...")
    
    aligned = state.get("aligned_units", [])
    rag = state.get("rag_snippets", [])
    # 如果有评审反馈，就带上反馈；如果没有（首轮），就用默认提示词
    feedback = state.get("critic_feedback") or "（首轮：请根据证据生成初稿。）"
    prev = state.get("draft_summary") or ""
    
    # 模拟 LLM 生成过程
    draft = f"# 会议总结（迭代 {state.get('iteration', 0)}）\n根据反馈：{feedback[:10]}... 修正了内容。"
    
    # 将新生成的草稿更新到全局状态中
    return {"draft_summary": draft}


def llm_critic(state: ConferenceState) -> dict[str, Any]:
    """Critic (评审者) 节点：负责给草稿打分并给出修改建议"""
    # 输入 (读取 State): draft_summary (刚才写好的草稿), iteration (当前循环次数)；输出 (返回 Dict): critic_score (数值分数), critic_feedback (文字建议)
    print("🧐 [节点执行]: LLM 评审者开始打分...")
    draft = state.get("draft_summary", "")
    
    # 模拟 LLM 评审过程（实际开发中要求 LLM 输出 JSON 格式以便解析）
    # 这里模拟随着迭代次数增加，分数越来越高
    current_iter = state.get("iteration", 0)
    score = 6.5 + current_iter * 1.5  # 模拟提分
    
    feedback = "示例：结论段缺少时间锚点；某论断超出证据单元范围。"
    print(f"   -> 评审结果：分数 {score}, 建议：{feedback}")
    
    # 将分数和建议更新到全局状态，供下一轮 writer 使用
    return {"critic_score": score, "critic_feedback": feedback}


# ============ 4) 条件路由 (Conditional Routing) ============
# 【核心概念】：这就是状态机里决定“下一步去哪”的交通警察。

def route_after_critic(state: ConferenceState) -> Literal["revise", "accept"]:
    """
    路由函数：读取当前状态，返回下一步应该走向哪条分支。
    """
    score = float(state.get("critic_score") or 0.0)
    threshold = float(state.get("score_threshold") or 8.0)
    it = int(state.get("iteration") or 0)
    max_it = int(state.get("max_iterations") or 4)

    # 如果分数达标，或者尝试次数已经用光（防止死循环），则结束质询
    if score >= threshold or it >= max_it:
        print("✅ [路由决策]: 质量达标或达到最大重试次数，接受稿件！")
        return "accept"
    
    # 否则，打回重写
    print("❌ [路由决策]: 分数未达标，打回重写。")
    return "revise"

def bump_iteration(state: ConferenceState) -> dict[str, Any]:
    """单纯用来增加迭代计数器的节点"""
    return {"iteration": int(state.get("iteration") or 0) + 1}


# ============ 5) 总结子图 (Subgraph) ============
# 【核心概念】：把复杂逻辑封装成子模块。将“撰写-评审-打回”这个循环单独打包；这样在主图中，整个循环就只是一个叫 "summarize_loop" 的单个节点，极其优雅。

def build_summarize_subgraph():
    sg = StateGraph(ConferenceState) # 子图共享同一个 State 定义

    # 添加节点
    sg.add_node("writer", llm_writer)
    sg.add_node("critic", llm_critic)
    sg.add_node("incr", bump_iteration)

    # 设置子图的入口
    sg.set_entry_point("writer")
    
    # 画线：写完就交给评审
    sg.add_edge("writer", "critic")
    
    # 条件分支：评审完之后去哪？
    sg.add_conditional_edges(
        "critic",
        route_after_critic, # 调用路由函数
        {
            "revise": "incr", # 如果函数返回 "revise"，走向 "incr" 节点
            "accept": END     # 如果函数返回 "accept"，走到 END（跳出子图）
        },
    )
    
    # 增加完计数器后，回到 writer 继续重写，形成闭环
    sg.add_edge("incr", "writer")

    return sg.compile()


# ============ 6) 主图 (Main Graph) ============
# 【核心概念】：整个系统的宏观主干道。

def build_main_graph():
    g = StateGraph(ConferenceState)

    # 1. 注册主干节点
    g.add_node("preprocess", node_preprocess)
    g.add_node("asr", node_asr)
    g.add_node("ocr", node_ocr)
    g.add_node("align", node_align)
    g.add_node("rag_d", node_rag_d_self_index)

    # 2. 注册子图作为主图的一个特殊节点！
    summarize_sg = build_summarize_subgraph()
    g.add_node("summarize_loop", summarize_sg)

    # 3. 将单向流水线连起来
    g.set_entry_point("preprocess")
    g.add_edge("preprocess", "asr")
    g.add_edge("asr", "ocr")
    g.add_edge("ocr", "align")
    g.add_edge("align", "rag_d")
    g.add_edge("rag_d", "summarize_loop") # 数据准备完毕，进入子图开启质询循环
    
    # 子图循环结束后，直接走向整个任务的终点
    g.add_edge("summarize_loop", END)

    return g.compile()


if __name__ == "__main__":
    # 编译主图，生成可执行的 app
    app = build_main_graph()
    
    # 初始化状态
    init: ConferenceState = {
        "video_path": "lecture.mp4",
        "iteration": 0,
        "max_iterations": 4,
        "score_threshold": 8.0,
    }
    
    print("🚀 开始运行 Agent 工作流...\n" + "="*40)
    
    # invoke 会阻塞执行，直到图走到主图的 END
    final_state = app.invoke(init)
    
    print("="*40 + "\n🎉 最终生成的总结：\n")
    print(final_state.get("draft_summary", ""))
```

### 三、基于LangSmith框架的agent运维部分

**阶段一**：可观测性（运行过程白盒化）

- 目标：消除 Agent 运行时工具链与 LLM 交接的“黑盒”状态，实现端到端的耗时、Token 消耗以及中间产物的全链路追踪（Tracing）。
- 埋点追踪：重写各节点方法，给整个pipeline的节点都埋下 @traceable 装饰器
- 跑一批测试建立基线：选取三段风格不同的讲座视频，让agent跑通全流程
- 产出初始状态记录表（直接在langsmith的tracing部分看）：端到端的延迟时间统计、token成本核算、错误率error-rate

**阶段二**：评估（建立benchmark）

- 目标：构建自动化 LLM-as-a-Judge 评估管道，量化系统的总结质量，为后续的方法优化与架构重构提供数据基准。
- 准备黄金数据集/参考答案：在 LangSmith UI 中创建一个名为 `Conference-Eval-Set` 的 Dataset；手动录入 10-20 个测试切片（每个切片对应 3-5 分钟的复杂视频片段）；对于每个切片，人工手写出完美的 Markdown 摘要（Ground Truth），并列出必须包含的 3-5 个核心知识点（如特定公式、专有名词、核心结论）。
- Evaluator框架
- 产出模型能力基准报告：生成可视化指标矩阵（包含组件级误差与端到端质量指标），并通过多轮次跑批分析评估方差，量化模型在不同数据类型下的稳定性。

**Evaluator框架说明**
1. 端到端总结质量（LLM-as-Judge）：目的是评估摘要生成质量，包括关键点召回率、幻觉检测、整体语义相似度三个指标。关键点召回率将`must_have_points`里每一条作为Y/N提问judge，返回覆盖率（0-1），同时产出指标平均召回率与hard case上的召回率。幻觉检测将summary拆分成知识点，驻点判断能否在`gold_transcription + gold_ocr`里找到支持，返回被支持事实占比。整体语义相似度基于得分点给1-5分，比较summary和`ground_truth_summary`的寓意贴合度



2. agent流水线中间环节（Code Evaluator）：目的是评估agent流水线中间环节，用于失败case归因，包括公式文字识别覆盖率、ASR识别覆盖率。公式文字识别覆盖率从`gold_ocr`用正则提取所有latex公式，从`ocr_blocks` 取 `type=="formula"` 的内容，比对后返回召回率。ASR识别覆盖率把`transcript_segments` 的 `text` 字段拼起来 vs `gold_transcription`，比对后再给返回召回率。


**阶段三**：建立优化框架并验证优化成果



