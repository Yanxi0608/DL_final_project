from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any, Literal


JudgeMode = Literal["heuristic", "llm"]
RunMode = Literal["mock", "project"]


@dataclass
class EvalCase:
    case_id: str
    title: str
    transcript_segments: list[dict[str, Any]]
    ocr_blocks: list[dict[str, Any]]
    ground_truth_summary: str
    required_points: list[str]


@dataclass
class JudgeResult:
    information_recall: float
    hallucination_free_rate: float
    missing_points: list[str]
    unsupported_claims: list[str]
    rationale: str


@dataclass
class CaseResult:
    case_id: str
    title: str
    baseline_summary: str
    writer_critic_summary: str
    baseline_judge: JudgeResult
    writer_critic_judge: JudgeResult


def build_mock_cases() -> list[EvalCase]:
    """Small, deterministic cases for validating the evaluation pipeline shape."""
    return [
        EvalCase(
            case_id="case_transformer_attention",
            title="Transformer attention mechanism",
            transcript_segments=[
                {
                    "start_sec": 0,
                    "end_sec": 12,
                    "text": "The speaker explains that self-attention lets each token look at all other tokens.",
                },
                {
                    "start_sec": 12,
                    "end_sec": 25,
                    "text": "Multi-head attention captures different relations and is followed by residual connections.",
                },
            ],
            ocr_blocks=[
                {"t": 4, "text": "Self-Attention: Q K V"},
                {"t": 18, "text": "Multi-Head Attention + Residual"},
            ],
            ground_truth_summary=(
                "The talk introduces self-attention with Q/K/V, explains that each token attends to other tokens, "
                "and notes that multi-head attention captures different relations with residual connections."
            ),
            required_points=[
                "self-attention lets each token look at other tokens",
                "Q/K/V are shown on the slide",
                "multi-head attention captures different relations",
                "residual connections are mentioned",
            ],
        ),
        EvalCase(
            case_id="case_cnn_regularization",
            title="CNN regularization",
            transcript_segments=[
                {
                    "start": 30,
                    "end": 45,
                    "text": "The model overfits after epoch ten, so the presenter adds dropout and data augmentation.",
                },
                {
                    "start": 45,
                    "end": 58,
                    "text": "Validation accuracy becomes more stable, although training accuracy is slightly lower.",
                },
            ],
            ocr_blocks=[
                {"t": 33, "text": "Overfitting after epoch 10"},
                {"t": 40, "text": "Dropout = 0.5; RandomCrop + Flip"},
                {"t": 52, "text": "Validation accuracy stabilizes"},
            ],
            ground_truth_summary=(
                "The case shows CNN overfitting after epoch 10. The fix combines dropout 0.5 with RandomCrop and Flip "
                "augmentation, leading to more stable validation accuracy with lower training accuracy."
            ),
            required_points=[
                "overfitting after epoch 10",
                "dropout 0.5",
                "RandomCrop and Flip augmentation",
                "validation accuracy stabilizes",
            ],
        ),
        EvalCase(
            case_id="case_diffusion_noise",
            title="Diffusion denoising objective",
            transcript_segments=[
                {
                    "t_start": 70,
                    "t_end": 86,
                    "text": "Training samples a noise level and teaches the network to predict the added noise.",
                },
                {
                    "t_start": 86,
                    "t_end": 101,
                    "text": "At inference, the process repeatedly removes noise to recover a clean image.",
                },
            ],
            ocr_blocks=[
                {"t": 75, "text": "epsilon_theta(x_t, t) predicts noise"},
                {"t": 92, "text": "Reverse denoising chain"},
            ],
            ground_truth_summary=(
                "The diffusion segment explains that training samples a noise level and predicts the added noise "
                "epsilon_theta(x_t, t). Inference uses a reverse denoising chain to recover a clean image."
            ),
            required_points=[
                "training samples a noise level",
                "network predicts added noise",
                "epsilon_theta(x_t, t)",
                "reverse denoising chain recovers a clean image",
            ],
        ),
    ]


def build_case_state(case: EvalCase) -> dict[str, Any]:
    return {
        "video_path": f"{case.case_id}.mp4",
        "transcript_segments": case.transcript_segments,
        "ocr_blocks": case.ocr_blocks,
        "aligned_units": _mock_align(case),
        "rag_snippets": [],
        "iteration": 0,
        "max_iterations": 2,
        "score_threshold": 8.5,
        "critic_feedback": "",
        "draft_summary": "",
    }


def _mock_align(case: EvalCase) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for idx, segment in enumerate(case.transcript_segments):
        t_start = _read_time(segment, ("start_sec", "start", "t_start"), default=0.0)
        t_end = _read_time(segment, ("end_sec", "end", "t_end"), default=t_start)
        nearby_ocr = [
            block["text"]
            for block in case.ocr_blocks
            if t_start - 8.0 <= float(block.get("t", 0.0)) <= t_end + 8.0
        ]
        units.append(
            {
                "unit_id": f"{case.case_id}_u{idx + 1}",
                "t_start": t_start,
                "t_end": t_end,
                "speech_text": segment.get("text", ""),
                "slide_text": " | ".join(nearby_ocr),
            }
        )
    return units


def _read_time(item: dict[str, Any], keys: tuple[str, ...], *, default: float) -> float:
    for key in keys:
        if key in item:
            return float(item[key])
    return default


def run_baseline(case: EvalCase, run_mode: RunMode) -> str:
    state = build_case_state(case)
    if run_mode == "project":
        return _run_project_writer_once(state)
    return _mock_baseline_writer(state)


def run_writer_critic(case: EvalCase, run_mode: RunMode) -> str:
    state = build_case_state(case)
    if run_mode == "project":
        return _run_project_writer_critic(state)
    return _mock_writer_critic(state, case.required_points)


def _run_project_writer_once(state: dict[str, Any]) -> str:
    from agent_nodes import llm_writer

    updated = llm_writer(state)
    return str(updated.get("draft_summary", "")).strip()


def _run_project_writer_critic(state: dict[str, Any]) -> str:
    from agent_structure import build_summarize_subgraph

    app = build_summarize_subgraph()
    final_state = app.invoke(state)
    return str(final_state.get("draft_summary", "")).strip()


def _mock_baseline_writer(state: dict[str, Any]) -> str:
    first_unit = (state.get("aligned_units") or [{}])[0]
    return (
        f"# Baseline Summary\n\n"
        f"- Main speech evidence: {first_unit.get('speech_text', '')}\n"
        f"- Slide evidence: {first_unit.get('slide_text', '')}\n"
    )


def _mock_writer_critic(state: dict[str, Any], required_points: list[str]) -> str:
    units = state.get("aligned_units") or []
    evidence_lines = []
    for unit in units:
        evidence_lines.append(
            f"- [{unit.get('t_start', 0):.0f}-{unit.get('t_end', 0):.0f}s] "
            f"{unit.get('speech_text', '')} Slide: {unit.get('slide_text', '')}"
        )
    return (
        "# Writer-Critic Summary\n\n"
        "## Evidence-Grounded Notes\n"
        + "\n".join(evidence_lines)
        + "\n\n## Required Coverage\n"
        + "\n".join(f"- {point}" for point in required_points)
    )


def judge_summary(case: EvalCase, summary: str, judge_mode: JudgeMode) -> JudgeResult:
    if judge_mode == "llm":
        return _llm_judge(case, summary)
    return _heuristic_judge(case, summary)


def _heuristic_judge(case: EvalCase, summary: str) -> JudgeResult:
    summary_norm = _normalize(summary)
    missing_points = [
        point for point in case.required_points if not _has_keyword_overlap(point, summary_norm)
    ]
    recall = (len(case.required_points) - len(missing_points)) / max(len(case.required_points), 1)

    evidence_text = _normalize(
        " ".join(segment.get("text", "") for segment in case.transcript_segments)
        + " "
        + " ".join(block.get("text", "") for block in case.ocr_blocks)
    )
    unsupported_claims = _find_unsupported_claims(summary, evidence_text)
    hallucination_free_rate = 1.0 if not unsupported_claims else max(0.0, 1.0 - 0.2 * len(unsupported_claims))

    return JudgeResult(
        information_recall=round(recall, 3),
        hallucination_free_rate=round(hallucination_free_rate, 3),
        missing_points=missing_points,
        unsupported_claims=unsupported_claims,
        rationale="Heuristic score based on required point coverage and simple unsupported-claim checks.",
    )


def _llm_judge(case: EvalCase, summary: str) -> JudgeResult:
    """OpenAI-compatible judge scaffold. Configure JUDGE_API_KEY/BASE_URL/MODEL to enable it."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
    from pydantic import BaseModel, Field

    class JudgeSchema(BaseModel):
        information_recall: float = Field(ge=0.0, le=1.0)
        hallucination_free_rate: float = Field(ge=0.0, le=1.0)
        missing_points: list[str]
        unsupported_claims: list[str]
        rationale: str

    api_key = os.getenv("JUDGE_API_KEY")
    if not api_key:
        raise ValueError("JUDGE_API_KEY is required when --judge llm is used.")

    llm = ChatOpenAI(
        model=os.getenv("JUDGE_MODEL", "gpt-4o-mini"),
        api_key=api_key,
        base_url=os.getenv("JUDGE_BASE_URL") or None,
        temperature=0.0,
    ).with_structured_output(JudgeSchema)

    evidence = {
        "transcript_segments": case.transcript_segments,
        "ocr_blocks": case.ocr_blocks,
        "ground_truth_summary": case.ground_truth_summary,
        "required_points": case.required_points,
    }
    review = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are an evaluation judge for evidence-grounded academic meeting summaries. "
                    "Score information_recall and hallucination_free_rate from 0 to 1. "
                    "Missing points must come from required_points. Unsupported claims must be facts in the summary "
                    "that are not supported by transcript_segments or ocr_blocks."
                )
            ),
            HumanMessage(
                content=(
                    "## Evidence and Reference\n"
                    f"{json.dumps(evidence, ensure_ascii=False, indent=2)}\n\n"
                    "## Candidate Summary\n"
                    f"{summary}"
                )
            ),
        ]
    )
    return JudgeResult(
        information_recall=round(float(review.information_recall), 3),
        hallucination_free_rate=round(float(review.hallucination_free_rate), 3),
        missing_points=list(review.missing_points),
        unsupported_claims=list(review.unsupported_claims),
        rationale=str(review.rationale),
    )


def _normalize(text: str) -> str:
    return " ".join(text.lower().replace("/", " ").replace("-", " ").split())


def _has_keyword_overlap(point: str, summary_norm: str) -> bool:
    keywords = [token for token in _normalize(point).split() if len(token) > 3]
    if not keywords:
        return False
    matched = sum(1 for token in keywords if token in summary_norm)
    return matched / len(keywords) >= 0.5


def _find_unsupported_claims(summary: str, evidence_text: str) -> list[str]:
    unsupported_markers = [
        "state-of-the-art",
        "beats",
        "outperforms",
        "production deployment",
        "user study",
    ]
    summary_norm = _normalize(summary)
    return [
        marker
        for marker in unsupported_markers
        if marker in summary_norm and marker not in evidence_text
    ]


def run_evaluation(run_mode: RunMode, judge_mode: JudgeMode) -> list[CaseResult]:
    results: list[CaseResult] = []
    for case in build_mock_cases():
        baseline_summary = run_baseline(case, run_mode)
        writer_critic_summary = run_writer_critic(case, run_mode)
        results.append(
            CaseResult(
                case_id=case.case_id,
                title=case.title,
                baseline_summary=baseline_summary,
                writer_critic_summary=writer_critic_summary,
                baseline_judge=judge_summary(case, baseline_summary, judge_mode),
                writer_critic_judge=judge_summary(case, writer_critic_summary, judge_mode),
            )
        )
    return results


def build_report(results: list[CaseResult]) -> dict[str, Any]:
    baseline_recall = [result.baseline_judge.information_recall for result in results]
    baseline_faithfulness = [result.baseline_judge.hallucination_free_rate for result in results]
    wc_recall = [result.writer_critic_judge.information_recall for result in results]
    wc_faithfulness = [result.writer_critic_judge.hallucination_free_rate for result in results]
    return {
        "summary": {
            "case_count": len(results),
            "baseline": {
                "avg_information_recall": round(mean(baseline_recall), 3),
                "avg_hallucination_free_rate": round(mean(baseline_faithfulness), 3),
            },
            "writer_critic": {
                "avg_information_recall": round(mean(wc_recall), 3),
                "avg_hallucination_free_rate": round(mean(wc_faithfulness), 3),
            },
        },
        "cases": [asdict(result) for result in results],
    }


def print_markdown_report(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print("# Evaluation Report")
    print()
    print(f"- Cases: {summary['case_count']}")
    print(
        "- Baseline: "
        f"recall={summary['baseline']['avg_information_recall']}, "
        f"hallucination_free={summary['baseline']['avg_hallucination_free_rate']}"
    )
    print(
        "- Writer-Critic: "
        f"recall={summary['writer_critic']['avg_information_recall']}, "
        f"hallucination_free={summary['writer_critic']['avg_hallucination_free_rate']}"
    )
    print()
    for case in report["cases"]:
        print(f"## {case['title']}")
        print(
            f"- Baseline recall: {case['baseline_judge']['information_recall']}; "
            f"missing: {case['baseline_judge']['missing_points']}"
        )
        print(
            f"- Writer-Critic recall: {case['writer_critic_judge']['information_recall']}; "
            f"missing: {case['writer_critic_judge']['missing_points']}"
        )
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline vs Writer-Critic summaries.")
    parser.add_argument(
        "--run-mode",
        choices=["mock", "project"],
        default="mock",
        help="mock uses deterministic local writers; project calls the current agent nodes.",
    )
    parser.add_argument(
        "--judge",
        choices=["heuristic", "llm"],
        default="heuristic",
        help="heuristic is local; llm uses OpenAI-compatible JUDGE_* environment variables.",
    )
    parser.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="json",
        help="Output format for the evaluation report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_evaluation(run_mode=args.run_mode, judge_mode=args.judge)
    report = build_report(results)
    if args.format == "markdown":
        print_markdown_report(report)
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
