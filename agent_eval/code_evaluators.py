from __future__ import annotations

import re
from typing import Any
import difflib

from evaluator_utils import to_langsmith_result

# 拼音可选依赖
try:
    from pypinyin import lazy_pinyin
    _HAVE_PINYIN = True
except Exception:
    _HAVE_PINYIN = False

# ===========================================================================
# Code-based evaluator 集合（不调用 LLM，纯代码逻辑）
#
# 包含：
#   E4: OCR Formula Coverage —— OCR 公式识别覆盖率
#   E5: ASR Quality          —— ASR 质量（占位，下次实现）
#
# 设计：
#   - 共享文本归一化工具放在 "文本处理工具" 段落
#   - 每个 evaluator 独立一段，互不依赖
#   - 主入口函数遵循 LangSmith 接口约定:(run, example) -> dict
# ===========================================================================


# ===========================================================================
# 文本处理工具（E4 / E5 共享）
# ===========================================================================

# 匹配 LaTeX 公式的正则：
#   - $$...$$           : Beamer / 你的 OCR agent 输出格式
#   - $...$             : 行内公式
#   - \( ... \)         : gold_ocr 用的格式之一
#   - \[ ... \]         : gold_ocr 用的另一种格式
#   - \\( ... \\)       : 转义后的 inline 公式
#   - \\[ ... \\]       : 转义后的 display 公式
_FORMULA_PATTERNS = [
    re.compile(r"\$\$(.+?)\$\$", re.DOTALL),
    re.compile(r"(?<!\$)\$([^\$\n]+?)\$(?!\$)"),       # 单 $ 包裹,避开 $$
    re.compile(r"\\\((.+?)\\\)", re.DOTALL),
    re.compile(r"\\\[(.+?)\\\]", re.DOTALL),
]


def extract_formulas_from_text(text: str) -> list[str]:
    """从一段含 LaTeX 标记的文本中提取所有公式片段（去包裹符,保留内部内容）。"""
    if not text:
        return []
    found: list[str] = []
    for pat in _FORMULA_PATTERNS:
        for m in pat.finditer(text):
            inner = m.group(1).strip()
            if inner:
                found.append(inner)
    return found


def normalize_latex(formula: str) -> str:
    """LaTeX 公式归一化：消除常见的无意义差异,便于后续匹配。

    处理项:
        1. 统一空白(多个空格/换行 → 单个空格,再去首尾)
        2. 移除 LaTeX 修饰命令但保留语义(\\left, \\right, \\,, \\;, \\!)
        3. 统一常见同义命令(\\mathbb{R} 与 \\R, \\dot 与 \\.)
        4. 移除 \\text{...} 包裹(只保留内容)
        5. 转为小写以容忍变量大小写差异(慎用,可选)
    """
    if not formula:
        return ""

    s = formula

    # 1. 剥离常见的"间距"命令
    s = re.sub(r"\\left|\\right|\\,|\\;|\\!|\\:", "", s)

    # 2. \text{xxx} → xxx
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)

    # 3. \mathbb / \mathcal / \mathrm 等保留花括号内的内容(不影响匹配)
    s = re.sub(r"\\(mathbb|mathcal|mathrm|mathbf|mathit)\{([^{}]*)\}", r"\2", s)

    # 4. 空白归一
    s = re.sub(r"\s+", " ", s).strip()

    # 5. 大小写归一(可选,这里启用)
    s = s.lower()

    return s


def _char_set(s: str) -> set:
    """返回字符串中所有非空白字符的集合,用于 Jaccard 相似度。"""
    return set(ch for ch in s if not ch.isspace())


def _jaccard_similarity(a: str, b: str) -> float:
    """两个字符串的字符集 Jaccard 相似度 ∈ [0, 1]。"""
    sa, sb = _char_set(a), _char_set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ===========================================================================
# E4: OCR Formula Coverage —— 公式识别覆盖率
# ===========================================================================

# Jaccard 相似度阈值:超过此值视为"近似匹配"
_JACCARD_THRESHOLD = 0.75


def _match_formula(
    gold: str,
    agent_formulas_norm: list[str],
) -> tuple[str, str | None, float]:
    """对单个 gold 公式,在 agent 公式集合中查找最佳匹配。

    Returns:
        (match_level, matched_agent_formula, similarity)
          match_level ∈ {"exact", "contains", "jaccard", "miss"}
          matched_agent_formula : 命中的 agent 归一化公式;miss 时为 None
          similarity            : Jaccard 相似度(0~1)
    """
    gold_norm = normalize_latex(gold)
    if not gold_norm:
        return ("miss", None, 0.0)

    # 1. 精确匹配
    for af in agent_formulas_norm:
        if af == gold_norm:
            return ("exact", af, 1.0)

    # 2. 包含匹配(子串)
    for af in agent_formulas_norm:
        if gold_norm in af or af in gold_norm:
            return ("contains", af, _jaccard_similarity(gold_norm, af))

    # 3. Jaccard 相似度
    best_af, best_sim = None, 0.0
    for af in agent_formulas_norm:
        sim = _jaccard_similarity(gold_norm, af)
        if sim > best_sim:
            best_sim = sim
            best_af = af
    if best_sim >= _JACCARD_THRESHOLD:
        return ("jaccard", best_af, best_sim)

    return ("miss", best_af, best_sim)


def formula_coverage(run, example) -> dict[str, Any]:
    """LangSmith evaluator 接口: E4 公式识别覆盖率。

    Args:
        run.outputs.ocr_blocks         : agent 输出的 OCR 块列表
                                         每个块: {"text": str, "t": float, "type": str}
        example.outputs.gold_ocr       : 参考 OCR 文本(含 LaTeX 公式)

    Returns:
        score / value     : 覆盖率 ∈ [0, 1]
        comment           : 人类可读结论
        formula_details   : 每个 gold 公式的匹配诊断
    """
    # ---------- 1. 取数据 ----------
    outputs = getattr(run, "outputs", None) or {}
    ocr_blocks = outputs.get("ocr_blocks", []) or []

    gt = getattr(example, "outputs", None) or {}
    gold_ocr = gt.get("gold_ocr", "") or ""

    # ---------- 2. 提取 gold 公式 ----------
    gold_formulas = extract_formulas_from_text(gold_ocr)

    # 边界:GT 里没有公式 → 跳过该 case
    if not gold_formulas:
        return {
            "key": "formula_coverage",
            "score": None,
            "comment": "该 case 的 gold_ocr 中未检测到公式,跳过 E4。",
        }

    # ---------- 3. 提取 agent 公式 ----------
    # 关键修复：gold 侧会抽 gold_ocr 里所有定界公式(含大量行内公式)，
    # 所以 agent 侧也必须把正文 text 块里的行内 $...$ 一并抽出来，否则系统性判 miss。
    # 只跳过 image/figure 类块，避免把图注 OCR 垃圾(如 "$U_{2}$W")混进公式池。
    _SKIP_TYPES = {"image", "figure", "figure_title", "formula_number"}
    agent_formulas_raw: list[str] = []
    for block in ocr_blocks:
        if block.get("type") in _SKIP_TYPES:
            continue
        text = (block.get("text") or "").strip()
        if not text:
            continue
        inner = extract_formulas_from_text(text)
        if inner:
            agent_formulas_raw.extend(inner)
        elif block.get("type") == "formula":
            # formula 块即便没识别出包裹符,也整段收下(沿用原行为)
            agent_formulas_raw.append(text)

    agent_formulas_norm = [normalize_latex(f) for f in agent_formulas_raw]
    agent_formulas_norm = [f for f in agent_formulas_norm if f]  # 去空

    # 边界:agent 一个公式都没识别出来
    if not agent_formulas_norm:
        return {
            "key": "formula_coverage",
            "score": 0.0,
            "value": 0.0,
            "comment": f"agent 未识别任何公式,gold 中共 {len(gold_formulas)} 条公式全部 miss。",
            "formula_details": [
                {
                    "gold_formula": g,
                    "match_level": "miss",
                    "matched_agent_formula": None,
                    "similarity": 0.0,
                }
                for g in gold_formulas
            ],
        }

    # ---------- 4. 逐条匹配 ----------
    details: list[dict[str, Any]] = []
    level_counts = {"exact": 0, "contains": 0, "jaccard": 0, "miss": 0}

    for g in gold_formulas:
        level, matched, sim = _match_formula(g, agent_formulas_norm)
        level_counts[level] += 1
        details.append({
            "gold_formula": g,
            "match_level": level,
            "matched_agent_formula": matched,
            "similarity": round(sim, 3),
        })

    # ---------- 5. 算分 ----------
    # 覆盖率 = 任何级别命中的占比(exact + contains + jaccard)
    matched_count = level_counts["exact"] + level_counts["contains"] + level_counts["jaccard"]
    coverage = matched_count / len(gold_formulas)

    comment = (
        f"识别 {matched_count}/{len(gold_formulas)} 条公式 "
        f"(exact={level_counts['exact']}, contains={level_counts['contains']}, "
        f"jaccard={level_counts['jaccard']}, miss={level_counts['miss']})。"
        f" Agent 公式总数: {len(agent_formulas_norm)}。"
    )

    return to_langsmith_result({
        "key": "formula_coverage",
        "score": coverage,
        "value": coverage,
        "comment": comment,
        "formula_details": details,
        "agent_formula_count": len(agent_formulas_norm),
        "gold_formula_count": len(gold_formulas),
    })


# -*- coding: utf-8 -*-
# ===========================================================================
# E5: ASR Quality v2  —— 可直接替换 code_evaluators.py 中的 asr_quality 段
#
# 相比 v1 的改动（只动得分器，不动 agent）：
#   1. 关键术语优先用 dataset 里人工策展的 example.outputs.key_terms；
#      没有则回退到旧的自动抽取（4-gram + 英文 token）。
#   2. 术语匹配：精确子串 → 字符模糊 → 拼音容错（同音字不再算 miss）。
#   3. length 从"奖励项"改成"单边护栏"：只在严重漏转时扣分。
#   4. 权重 char 0.4 / term 0.6，再乘长度护栏。
#
# 依赖：pip install pypinyin（可选；缺失时自动降级为更严格的纯字符匹配）
# ===========================================================================

# ---------------- 可调参数 ----------------
_W_CHAR_SIM = 0.4          # 字符级相似度权重（v1 是 0.3）
_W_TERM_RECALL = 0.6       # 关键术语召回权重（v1 是 0.5；length 不再单独占权重）
_LENGTH_FLOOR = 0.5        # 长度护栏：agent/gold 长度比低于此值才开始扣分

_CHAR_HIT_THRESH = 0.75    # 字符模糊匹配阈值
_PY_HIT_THRESH = 0.78      # 拼音模糊匹配阈值

# 自动抽词（回退路径）参数
_MIN_EN_TOKEN_LEN = 3
_MIN_ZH_TERM_LEN = 4
_TERM_TOPK = 30

_PUNCT_RE = re.compile(r"[，。；：、“”‘’《》（）()\[\]【】！？!?,.:;\"']")
_WS_RE = re.compile(r"\s+")
_EN_RE = re.compile(r"[a-zA-Z]+")
_ZH_RE = re.compile(r"[\u4e00-\u9fff]+")


# ---------------- 文本处理 ----------------

def _concat_transcript_segments(segments: list) -> str:
    if not segments:
        return ""
    parts = []
    for seg in segments:
        if isinstance(seg, dict):
            t = (seg.get("text") or "").strip()
            if t:
                parts.append(t)
    return " ".join(parts)


def _normalize_asr_text(text: str) -> str:
    if not text:
        return ""
    s = _PUNCT_RE.sub(" ", text)
    s = _WS_RE.sub(" ", s)
    return s.strip().lower()


def _char_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


# ---------------- 自动抽词（仅当 dataset 未提供 key_terms 时回退使用）----------------

def _auto_key_terms(gold_transcription: str, gold_ocr: str) -> list[str]:
    from collections import Counter
    counter: Counter = Counter()
    counter.update(m.group(0).lower() for m in _EN_RE.finditer(gold_transcription)
                   if len(m.group(0)) >= _MIN_EN_TOKEN_LEN)
    for chunk_match in _ZH_RE.finditer(gold_transcription):
        chunk = chunk_match.group(0)
        for i in range(len(chunk) - _MIN_ZH_TERM_LEN + 1):
            counter[chunk[i:i + _MIN_ZH_TERM_LEN]] += 1
    counter.update(m.group(0).lower() for m in _EN_RE.finditer(gold_ocr)
                   if len(m.group(0)) >= _MIN_EN_TOKEN_LEN)
    stopwords = {"the", "and", "for", "are", "with", "this", "that",
                 "from", "have", "has", "was", "were", "but", "not"}
    terms = [(t, c) for t, c in counter.items() if t not in stopwords]
    terms.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in terms[:_TERM_TOPK]]


# ---------------- 模糊 / 拼音匹配 ----------------

def _pyify(s: str) -> str:
    """中文转无调拼音并拼接；英文/数字小写保留；其它丢弃。"""
    if not _HAVE_PINYIN:
        return s
    out = []
    for ch in s:
        if "\u4e00" <= ch <= "\u9fff":
            out.append(lazy_pinyin(ch)[0])
        elif ch.isalnum():
            out.append(ch.lower())
    return "".join(out)


def _best_window_ratio(needle: str, haystack: str, w_lo: int, w_hi: int) -> float:
    """在 haystack 中按若干窗口长度滑动，返回与 needle 的最高相似度。"""
    if not needle or not haystack:
        return 0.0
    best = 0.0
    for w in range(max(1, w_lo), w_hi + 1):
        limit = len(haystack) - w
        if limit < 0:
            continue
        for i in range(limit + 1):
            r = difflib.SequenceMatcher(None, needle, haystack[i:i + w]).ratio()
            if r > best:
                best = r
                if best >= 0.999:
                    return best
    return best


def _term_hit(term: str, agent_norm: str, agent_py: str) -> bool:
    """单个术语是否命中：精确子串 → 字符模糊 → 拼音容错。"""
    t = term.lower()
    if t in agent_norm:                                   # 1. 精确
        return True
    L = len(t)
    # 2. 字符模糊（窗口长度与术语相当，避免只匹配到 1 个字的假阳性）
    if _best_window_ratio(t, agent_norm, L, L + 1) >= _CHAR_HIT_THRESH:
        return True
    # 3. 拼音容错（专治同音/近音错字）
    if _HAVE_PINYIN:
        tp = _pyify(term)
        if tp:
            if tp in agent_py:
                return True
            if _best_window_ratio(tp, agent_py, max(1, len(tp) - 1), len(tp) + 1) >= _PY_HIT_THRESH:
                return True
    return False


def _term_recall(agent_norm: str, key_terms: list[str]) -> tuple[float, list[str]]:
    if not key_terms:
        return 1.0, []
    if not agent_norm:
        return 0.0, list(key_terms)
    agent_py = _pyify(agent_norm)
    missing = [t for t in key_terms if not _term_hit(t, agent_norm, agent_py)]
    found = len(key_terms) - len(missing)
    return found / len(key_terms), missing


# ---------------- 长度护栏（单边）----------------

def _length_guard(agent_text: str, gold_text: str) -> float:
    if not gold_text:
        return 1.0
    if not agent_text:
        return 0.0
    ratio = min(len(agent_text), len(gold_text)) / max(len(agent_text), len(gold_text))
    # 长度正常 → 不影响；只有严重漏转（比例低于 floor）才按比例扣
    return 1.0 if ratio >= _LENGTH_FLOOR else ratio / _LENGTH_FLOOR


# ---------------- LangSmith evaluator 主函数 ----------------

def asr_quality(run, example) -> dict[str, Any]:
    """E5 v2: ASR 质量（字符相似度 + 术语召回，长度作为单边护栏）。"""
    outputs = getattr(run, "outputs", None) or {}
    transcript_segments = outputs.get("transcript_segments", []) or []

    gt = getattr(example, "outputs", None) or {}
    gold_transcription = gt.get("gold_transcription", "") or ""
    gold_ocr = gt.get("gold_ocr", "") or ""
    curated_terms = gt.get("key_terms", None)   # 新增：dataset 人工策展词

    if not gold_transcription.strip():
        return {
            "key": "asr_quality",
            "score": None,
            "comment": "该 case 缺少 gold_transcription，跳过 E5。",
        }

    agent_text = _normalize_asr_text(_concat_transcript_segments(transcript_segments))
    gold_text = _normalize_asr_text(gold_transcription)

    if not agent_text:
        return to_langsmith_result({
            "key": "asr_quality",
            "score": 0.0,
            "value": 0.0,
            "comment": "agent 未识别出任何 ASR 内容（transcript_segments 为空）。",
            "char_similarity": 0.0,
            "term_recall": 0.0,
            "length_guard": 0.0,
        })

    # 关键术语来源：优先人工策展，否则回退自动抽取
    if curated_terms:
        key_terms = [t for t in curated_terms if (t or "").strip()]
        term_source = "curated"
    else:
        key_terms = _auto_key_terms(gold_transcription, gold_ocr)
        term_source = "auto"

    char_sim = _char_similarity(agent_text, gold_text)
    term_rec, missing = _term_recall(agent_text, key_terms)
    guard = _length_guard(agent_text, gold_text)

    base = _W_CHAR_SIM * char_sim + _W_TERM_RECALL * term_rec
    score = round(base * guard, 4)

    comment = (
        f"ASR 综合分 {score:.3f}: char_sim={char_sim:.3f}, "
        f"term_recall={term_rec:.3f} "
        f"({len(key_terms) - len(missing)}/{len(key_terms)} 术语命中, 来源={term_source}), "
        f"length_guard={guard:.3f}"
        + (f"。漏识别术语: {', '.join(missing[:8])}" if missing else "。")
    )

    return to_langsmith_result({
        "key": "asr_quality",
        "score": score,
        "value": score,
        "comment": comment,
        "char_similarity": round(char_sim, 4),
        "term_recall": round(term_rec, 4),
        "length_guard": round(guard, 4),
        "term_source": term_source,
        "key_terms_count": len(key_terms),
        "missing_terms": missing[:15],
    })



