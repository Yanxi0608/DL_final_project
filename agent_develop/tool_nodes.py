from __future__ import annotations
import os
import cv2
from skimage.metrics import structural_similarity as ssim
# 1. 强制关闭带有 Bug 的新版 PIR 计算引擎
os.environ["FLAGS_enable_pir_api"] = "0"
# 2. 彻底关闭导致 onednn_instruction.cc 崩溃的 oneDNN/MKLDNN 全局加速器（关键点！）
os.environ["FLAGS_use_mkldnn"] = "0"
# 3. 防止多线程库冲突引发闪退
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from pathlib import Path
from typing import Any, TypedDict

import json
import os
import subprocess
import tempfile
from pathlib import Path

# ==========共享状态定义（后续需要进一步统一数据格式）===================
class AlignedUnit(TypedDict, total=False):
    unit_id: str  
    t_start: float
    t_end: float
    speech_text: str
    slide_text: str

class ConferenceState(TypedDict, total=False):
    # 顺序处理阶段系统变量
    video_path: str
    audio_path: str
    keyframes: list[dict[str, Any]]
    transcript_segments: list[dict[str, Any]]
    ocr_blocks: list[dict[str, Any]]
    aligned_units: list[AlignedUnit]
    # 循环打分系统变量
    rag_snippets: list[dict[str, Any]]
    draft_summary: str
    critic_score: float
    critic_feedback: str
    iteration: int
    max_iterations: int
    score_threshold: float
    errors: list[str]

# ========================== wr部分 =========================



# ===========数据预处理================
# 安全地执行工具，当音视频处理卡住时，杀死进程并打印错误日志。
def _run_command(command: list[str], timeout: float = 30.0) -> str:
    try:
        # 当正常执行时，_run_command通过subprocess.run()在系统里拉起一个子进程
        completed = subprocess.run(
            command,
            capture_output=True,  # 把FFmpeg在控制台输出的所有文字拦截，存入内存
            text=True,          # 把FFmpeg吐出来的bytes流转化为python的字符串str
            encoding="utf-8",   # 明确指定编码
            timeout=timeout     # 设置超时时间（秒）
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Command failed: {' '.join(command)}\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        # 若_run_command不报错，则返回拦截到的控制台文本，作为函数返回
        return completed.stdout.strip() 
        
    except subprocess.TimeoutExpired as e:
        # 修复点：直接使用字符串，不再调用 .decode()
        stdout = e.stdout if e.stdout else ""
        stderr = e.stderr if e.stderr else ""
        raise RuntimeError(
            f"❌ 命令运行超时 ({timeout}秒)，进程已被强行终止！\n"
            f"命令: {' '.join(command)}\n"
            f"👉 此时 FFmpeg 的 stdout 输出:\n{stdout}\n"
            f"👉 此时 FFmpeg 的 stderr 输出:\n{stderr}\n"
            f"💡 提示：请观察上方输出，看 FFmpeg 究竟死锁在哪个步骤。"
        )

# 检查指定路径文件是否存在，不存在则抛出异常。
def _ensure_file(path: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    return p

# 通过ffprobe工具读取视频/音频文件总时长
def _get_media_duration(path: Path) -> float:
    output = _run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    try:
        return float(output.strip())
    except ValueError:
        return 0.0

# 根据视频总时长，等间隔生成指定数量的关键帧时间戳列表（单位：秒）
def _sample_keyframe_timestamps(duration: float, count: int = 5) -> list[float]:
    if duration <= 0:
        return [0.0]
    if count <= 1:
        return [0.0]
    return [round(min(duration, i * duration / (count - 1)), 3) for i in range(count)]



## ================核心图节点：音视频数据预处理======================

# preprocess节点的输出：
# audio_path: 转换后的音频文件路径，供ASR节点使用
# keyframes: 关键帧列表。每个元素是一个字典：
# 字典格式：{“t”:时间戳,“path”:该时间戳对应的图片物理路径path（供OCR节点使用）}

def node_preprocess(state: ConferenceState) -> dict[str, Any]:
    print("🎬 [节点执行]: 预处理视频（集成 1.ipynb 智能翻页算法）...")
    
    video_path = state.get("video_path")
    if not video_path:
        raise ValueError("state 中缺少 video_path")

    video_file = _ensure_file(video_path)
    audio_file = video_file.with_suffix(".wav")

    # ====================================================
    # Step 1: 提取音频 
    # ====================================================
    _run_command(
        [
            "ffmpeg",
            "-y", 
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_file),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-acodec",
            "pcm_s16le",
            str(audio_file),
        ]
    )

    # ====================================================
    # Step 2:智能翻页与稳定态检测算法
    # ====================================================
    # 创建临时图片缓存目录
    output_dir = video_file.parent / "keyframes_cache"
    output_dir.mkdir(exist_ok=True)

    SAMPLE_SEC = 1.0          # 每隔1秒采样一次
    SIM_THRESHOLD = 0.92      # 翻页判定阈值
    STABLE_THRESHOLD = 0.98   # 静止稳定态判定阈值

    def preprocess_image(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (640, 360))

    # 初始化视频流
    cap = cv2.VideoCapture(str(video_file))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0  # 兜底防止分母为 0
    frame_step = int(fps * SAMPLE_SEC)

    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError("无法读取视频内容，请检查视频文件是否损坏。")

    # 物理保存初始第 0 秒的第一张 PPT 
    img_name = "frame_0_0.jpg"
    img_path = output_dir / img_name
    cv2.imwrite(str(img_path), frame)

    last_saved_gray = preprocess_image(frame)
    prev_gray = last_saved_gray
    
    # 严格保持全局状态图约定的数据结构：包含 t 和 path 的字典列表
    keyframes_output = [{
        "t": 0.0, 
        "path": str(img_path)
    }]

    # 步进式跳跃遍历
    frame_idx = frame_step
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    
    saved_count = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        curr_gray = preprocess_image(frame)
        
        # 计算当前帧 vs 上一张保存的PPT
        score_vs_saved, _ = ssim(curr_gray, last_saved_gray, full=True)
        # 计算当前帧 vs 上一秒的画面
        score_vs_prev, _ = ssim(curr_gray, prev_gray, full=True)
        
        # 核心智能过滤逻辑：只有当发生翻页现象（不同于上一张）且已经稳定静止时才提取
        if score_vs_saved < SIM_THRESHOLD and score_vs_prev > STABLE_THRESHOLD:
            seconds = frame_idx / fps
            
            # 动态构造标准命名的图片路径
            img_name = f"frame_{saved_count}_{int(seconds)}.jpg"
            img_path = output_dir / img_name
            
            # 使用 OpenCV 内存写入，完美规避 FFmpeg MJPEG 编码器报错！
            cv2.imwrite(str(img_path), frame)
            
            keyframes_output.append({
                "t": round(seconds, 3), 
                "path": str(img_path)
            })
            
            last_saved_gray = curr_gray
            saved_count += 1
            print(f"   -> 📸 检测到 PPT 翻页稳定态，成功提取关键帧：{img_name} (时间戳: {int(seconds)}s)")

        # 步进至下一采样点
        prev_gray = curr_gray
        frame_idx += frame_step
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    cap.release()
    print(f"✅ 预处理流执行成功！音轨已分离，且通过 SSIM 智能过滤算法共提取了 {len(keyframes_output)} 张有效 PPT 页。")

    # 返回更新后的全局状态字段，无缝对接下游 node_ocr 节点
    return {
        "audio_path": str(audio_file),
        "keyframes": keyframes_output,
    }




def node_asr(state: ConferenceState) -> dict[str, Any]:
    print("🎧 [节点执行]: 语音识别 (智能动态 Prompt 泛化版)...")
    
    audio_file = Path(state.get("audio_path", ""))
    
    # 动态泛化：尝试从全局状态中拿此前 OCR 节点或者预处理捞到的封面文字
    # 如果有封面标题，直接动态生成提示词；没有则用通用学术大提示词兜底
    ocr_blocks = state.get("ocr_blocks", [])

    # 只用标题和普通文字块生成prompt，过滤掉LaTeX公式（避免干扰Whisper）
    cover_text = " ".join([
    b.get("text", "").strip()
    for b in ocr_blocks
    if b.get("type", "") in ("paragraph_title", "doc_title", "text", "")
    and not b.get("text", "").startswith("$$")  # 排除公式块
    ][:5])

    generic_prompt = "这是一个标准的专业学术会议、技术讲座报告，演讲中包含大量专业术语和学术论坛名词。"
    if cover_text.strip():
        dynamic_prompt = f"{generic_prompt} 本场讲座的主题围绕: {cover_text.strip()} 展开。"
    else:
        dynamic_prompt = generic_prompt

    import whisper
    model = whisper.load_model("base") 
    
    # 彻底告别硬编码，全自动泛化适配任意视频
    result = model.transcribe(str(audio_file), language="zh", initial_prompt=dynamic_prompt)
    
    segments = []
    for seg in result.get("segments", []):
        text = (seg.get("text", "") or "").strip()
        if text:
            segments.append({
                "text": text,
                "t_start": round(float(seg.get("start", 0.0)), 3),
                "t_end": round(float(seg.get("end", 0.0)), 3),
            })
    return {"transcript_segments": segments}











## ================核心图节点：试听时间戳对齐======================

# 把所有OCR识别到的文本片段用|拼接成一段长文本，作为幻灯片文本输入给LLM，在node_align中使用
def _collect_slide_text(ocr_blocks: list[dict[str, Any]]) -> str:
    text_blocks = [block.get("text", "").strip() for block in ocr_blocks if block.get("text")]
    return " | ".join(text_blocks) if text_blocks else ""


# node_align的输出与agent_structure.py中的AlignedUnit结构一致,输出示例如下：
# {
#     "aligned_units": [
#         {
#             "unit_id": "unit_1",
#             "t_start": 0.0,
#             "t_end": 5.23,
#             "speech_text": "大家好，今天我们来分享一下深度学习...",
#             "slide_text": "深度学习在NLP中的应用 | 讲师：张三 | Word2Vec 核心公式 | Loss = ..."
#         },
#         ...
#     ]
# }




def node_align(state: ConferenceState) -> dict[str, Any]:
    
    print("🔗 [节点执行]: 视听时间戳高精度智能对齐...")
    
    # 读取asr生成的segments
    transcript_segments = state.get("transcript_segments") or []
    # 读取ocr生成的ocr_blocks
    ocr_blocks = state.get("ocr_blocks") or []
    # 创建空的返回列表：aligned_units
    aligned_units: list[AlignedUnit] = []
    
    # 1. 建立时间戳到 OCR 文本列表的映射，把杂乱的块按截图时间归类
    # 例如：{ 0.0: ["标题", "简介"], 25.0: ["公式一", "参数定义"] }
    time_to_ocr: dict[float, list[str]] = {}
    for block in ocr_blocks:
        t = float(block.get("t", 0.0))
        text = block.get("text", "").strip()
        if text:
            time_to_ocr.setdefault(t, []).append(text)

    # 2. 开始逐句对齐语音
    for idx, segment in enumerate(transcript_segments):
        start = float(segment.get("t_start", 0.0))
        end = float(segment.get("t_end", start + 5.0))
        
        # 核心逻辑 A：寻找在这段语音 [start, end] 区间内截取的所有图片文字
        matched_texts = []
        for img_time, texts in time_to_ocr.items():
            # 允许 1.5 秒的边界模糊，防止 ASR 和 OCR 时间切片微弱错开
            if start <= img_time <= (end + 1.5):
                matched_texts.extend(texts)
        
        # 核心逻辑 B（关键兜底）：如果当前几秒内没有截图，说明 PPT 没翻页，去追溯“过去最近的一张 PPT”
        if not matched_texts and time_to_ocr:
            past_times = [t for t in time_to_ocr.keys() if t <= start]
            if past_times:
                closest_time = max(past_times)  # 找到离当前说话时间最近的过去截图点
                matched_texts = time_to_ocr[closest_time]
            else:
                # 如果说话太靠前连第一张图都没到，就用全场第一张图兜底
                matched_texts = time_to_ocr[min(time_to_ocr.keys())]

        # 拼接当前语音片段真正应该看到的屏幕文字
        current_slide_text = " | ".join(matched_texts) if matched_texts else "（当前画面无文字）"

        aligned_units.append({
            "unit_id": f"unit_{idx+1}",
            "t_start": round(start, 3),
            "t_end": round(end, 3),
            "speech_text": segment.get("text", "").strip(),
            "slide_text": current_slide_text,  # 此时每个单元拿到的都是只属于这个时间段内的ocr识别内容
        })

    return {"aligned_units": aligned_units}


def node_rag_d_self_index(state: ConferenceState) -> dict[str, Any]:
    """本场 RAG 占位节点（主图拓扑保留，评测阶段暂不检索）。"""
    return {"rag_snippets": []}


# ========================== OCR部分 =========================
# 说明：使用paddleocr；该深度学习模型可以同时实现文字和公式识别，
# 但是准确率不高（主要是其中的公式识别部分没法转化成tex形式）；后续打算用这个模型中的V3版本进一步提高准确率

# 使用 PP-StructureV3：版面分析 + 文本识别 + LaTeX公式识别
_struct_pipeline = None

def _get_structure_pipeline():
    global _struct_pipeline
    if _struct_pipeline is None:
        from paddleocr import PPStructureV3
        print("📖 [OCR] 初始化 PP-StructureV3 (Beamer公式识别模式)...")
        
        # 定义配置文件路径或使用默认配置映射
        # 这里我们通过传入 config 参数来指定模型名称
        _struct_pipeline = PPStructureV3(
            use_formula_recognition=True,
            use_table_recognition=False,
            use_seal_recognition=False,
            use_chart_recognition=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False
        )
    return _struct_pipeline




import cv2
import numpy as np

def node_ocr(state: ConferenceState) -> dict[str, Any]:
    print("👁️ [节点执行]: 图像文字识别 (Beamer：版面分析+LaTeX公式识别)...")
    try:
        pipeline = _get_structure_pipeline()
        blocks: list[dict[str, Any]] = []

        # 对下游LLM没有实质内容的版面类型，直接过滤
        SKIP_LABELS = {"number", "header", "footer", "header_image",
                       "footer_image", "aside_text"}

        for kf in state.get("keyframes", []):
            path = kf.get("path")
            if not path or not os.path.exists(path):
                print(f"   ⚠️ 跳过不存在的图片: {path}")
                continue
            t = float(kf.get("t", 0))

            img = cv2.imread(path)
            if img is None:
                print(f"   ❌ 无法读取图片: {path}")
                continue

            # Beamer不需要CLAHE，但视频帧分辨率可能偏低，放大有助于识别小符号
            h, w = img.shape[:2]
            if max(h, w) < 1280:
                scale = 1280 / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)),
                                 interpolation=cv2.INTER_CUBIC)

            # 调用 PP-StructureV3，传入 numpy 数组
            output = list(pipeline.predict(img))
            if not output:
                continue

            res_data = output[0].json.get("res", {})
            parsing_res_list = res_data.get("parsing_res_list", [])

            n_text, n_formula = 0, 0
            for block in parsing_res_list:
                label = block.get("block_label", "")
                if label in SKIP_LABELS:
                    continue
                content = block.get("block_content", "").strip()
                if not content:
                    continue

                # 公式块：加 $$ 包裹，LLM能直接理解LaTeX
                if label == "formula":
                    content = f"$${content}$$"
                    n_formula += 1
                else:
                    n_text += 1

                blocks.append({
                    "text": content,
                    "t": t,
                    "type": label  # 保留类型：paragraph_title / text / formula 等
                })

            print(f"   ✅ {os.path.basename(path)}: {n_text} 文本块, {n_formula} 公式块")

        print(f"✅ OCR完成，共提取 {len(blocks)} 个内容块")
        return {"ocr_blocks": blocks}

    except Exception as exc:
        print(f"⚠️ [OCR 降级]: {exc}")
        import traceback
        traceback.print_exc()
        return {"ocr_blocks": []}




## =================ocr通过测试，可分离文字和公式，太不容易了===============
if __name__ == "__main__":
    print("=" * 70)
    print("🔬 [文字与公式识别 Agent Tool 可用性测试]")
    print("=" * 70)

    # 1. 明确路径定义
    cache_dir = Path(r"D:/DL_final_project/dataset/videos/keyframes_cache")
    
    # 2. 检查目录是否存在
    if not cache_dir.exists():
        print(f"❌ 错误：目录不存在: {cache_dir}")
        exit(1)

    # 3. 获取所有 jpg 图片并排序
    jpgs = sorted(cache_dir.glob("*.jpg"))
    if not jpgs:
        print(f"❌ 错误：在 {cache_dir} 下没有找到任何 .jpg 图片")
        exit(1)

    # 4. 只取前两张进行测试
    selected_jpgs = jpgs[:2]
    print(f"📷 发现图片数量: {len(jpgs)} 张，本次测试: {len(selected_jpgs)} 张")
    for img in selected_jpgs:
        print(f"   - {img.name}")

    # 5. 构造状态 (核心：解析文件名中的时间戳)
    keyframes = []
    for img_path in selected_jpgs:
        # 文件名格式：frame_0_0.jpg, frame_1_44.jpg
        # split('_') 后为 ['frame', '0', '0.jpg']
        try:
            parts = img_path.stem.split('_')
            # 取倒数第二个部分作为时间戳 (根据你的命名规则 frame_序号_时间)
            t = float(parts[-1]) 
        except (IndexError, ValueError):
            t = 0.0
            
        keyframes.append({"t": t, "path": str(img_path)})

    mock_state: ConferenceState = {"keyframes": keyframes}

    try:
        # 调用 OCR 节点
        ocr_result = node_ocr(mock_state)
        ocr_blocks = ocr_result.get("ocr_blocks", [])

        if ocr_blocks:
            print(f"\n✅ 识别成功！共提取 {len(ocr_blocks)} 个内容块。\n")
            
            # 按时间戳分组输出
            from collections import defaultdict
            blocks_by_time = defaultdict(list)
            for block in ocr_blocks:
                blocks_by_time[block["t"]].append(block)

            for t_val, blocks in sorted(blocks_by_time.items()):
                print(f"📸 时间点: {t_val}s")
                for bidx, block in enumerate(blocks, 1):
                    # 简略预览内容
                    text_preview = block['text'][:50].replace('\n', ' ') + ('...' if len(block['text'])>50 else '')
                    print(f"   📌 块 {bidx:2d} [{block['type']}]: {text_preview}")
                print("-" * 30)
            print("✨ 结论：OCR 工具与模型加载正常。")
        else:
            print("\n⚠️ 识别完成，但未提取到有效文本。请检查是否需要调整 OCR 预处理参数。")

    except Exception as e:
        print(f"\n❌ 识别过程中发生异常: {e}")
        import traceback
        traceback.print_exc()