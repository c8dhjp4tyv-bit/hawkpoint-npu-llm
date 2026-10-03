#!/usr/bin/env python3
"""Evaluate model accuracy, cross-entropy loss, and perplexity on XDNA1 or API.

Provides automated evaluation of LLM generation quality, answer accuracy,
cross-entropy loss, and perplexity (PPL) using OpenAI-compatible text completion
endpoints with token logprobs.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from npu_llm.http_client import open_api_request


DEFAULT_EVAL_DATASET = [
    {
        "id": "qa-france-capital",
        "category": "factual_qa",
        "prompt": "Q: What is the capital of France?\nA: The capital of France is",
        "expected": "Paris",
        "max_tokens": 8,
    },
    {
        "id": "qa-red-planet",
        "category": "factual_qa",
        "prompt": "Q: What planet is known as the Red Planet?\nA:",
        "expected": "Mars",
        "max_tokens": 6,
    },
    {
        "id": "math-addition",
        "category": "reasoning",
        "prompt": "Calculate 15 + 27. Answer:",
        "expected": "42",
        "max_tokens": 6,
    },
    {
        "id": "logic-transitive",
        "category": "reasoning",
        "prompt": "All roses are flowers. All flowers are plants. Therefore, all roses are",
        "expected": "plants",
        "max_tokens": 6,
    },
    {
        "id": "code-add",
        "category": "coding",
        "prompt": "def add(a, b):\n    \"\"\"Return sum of a and b.\"\"\"\n   ",
        "expected": "return a + b",
        "max_tokens": 10,
    },
    {
        "id": "lang-fluency",
        "category": "completion",
        "prompt": "The quick brown fox jumps over the lazy",
        "expected": "dog",
        "max_tokens": 6,
    },
]


@dataclass
class EvalItemResult:
    """Evaluation result for a single prompt item."""

    item_id: str
    category: str
    prompt: str
    expected: str
    generated_text: str
    matched: bool
    tokens: list[str]
    token_logprobs: list[float]
    cross_entropy: float | None
    perplexity: float | None
    per_token_latency_ms: float
    decode_tokens_per_second: float
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class EvalReport:
    """Aggregated evaluation report across all evaluated prompts."""

    model: str
    total_items: int
    matched_items: int
    accuracy: float
    mean_cross_entropy: float | None
    perplexity: float | None
    mean_per_token_latency_ms: float
    mean_tokens_per_second: float
    total_prompt_tokens: int
    total_completion_tokens: int
    categories: dict[str, dict[str, float]]
    items: list[EvalItemResult]


def compute_cross_entropy_and_perplexity(
    token_logprobs: list[float],
) -> tuple[float | None, float | None]:
    """Calculate mean negative log-likelihood (cross-entropy) and perplexity."""
    valid_logprobs = [
        lp
        for lp in token_logprobs
        if lp is not None and type(lp) in (int, float) and math.isfinite(lp) and lp <= 0
    ]
    if not valid_logprobs:
        return None, None
    mean_neg_ll = -sum(valid_logprobs) / len(valid_logprobs)
    clamped_loss = min(max(mean_neg_ll, 0.0), 100.0)
    perplexity = math.exp(clamped_loss)
    return mean_neg_ll, perplexity


def evaluate_single_item(
    api_url: str,
    api_key: str,
    model: str,
    item: dict,
    timeout: float = 30.0,
) -> EvalItemResult:
    """Send completion request with logprobs and measure accuracy and loss."""
    endpoint = f"{api_url.rstrip('/')}/v1/completions"
    prompt = item["prompt"]
    expected = item["expected"]
    max_tokens = item.get("max_tokens", 16)

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "logprobs": True,
    }

    headers = {
        "Content-Type": "application/json",
    }

    req = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
    )

    start_time = time.monotonic()
    with open_api_request(req, api_key, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    duration = time.monotonic() - start_time

    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError(f"API returned no choices for item {item.get('id')}")

    choice = choices[0]
    generated_text = choice.get("text", "")
    logprobs_info = choice.get("logprobs") or {}

    tokens = logprobs_info.get("tokens", [])
    token_logprobs = logprobs_info.get("token_logprobs", [])

    cross_entropy, ppl = compute_cross_entropy_and_perplexity(token_logprobs)

    # Check if expected target answer is matched (case-insensitive substring or exact word)
    norm_expected = expected.strip().lower()
    norm_gen = generated_text.strip().lower()
    matched = norm_expected in norm_gen

    usage = data.get("usage") or {}
    num_tokens = usage.get("completion_tokens", len(tokens))
    tps = num_tokens / duration if duration > 0 else 0.0
    per_token_latency_ms = duration * 1000.0 / max(1, num_tokens)

    return EvalItemResult(
        item_id=item.get("id", "unknown"),
        category=item.get("category", "general"),
        prompt=prompt,
        expected=expected,
        generated_text=generated_text,
        matched=matched,
        tokens=tokens,
        token_logprobs=token_logprobs,
        cross_entropy=cross_entropy,
        perplexity=ppl,
        per_token_latency_ms=per_token_latency_ms,
        decode_tokens_per_second=tps,
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=num_tokens,
    )


def run_evaluation(
    api_url: str,
    api_key: str,
    model: str,
    dataset: list[dict] | None = None,
    timeout: float = 30.0,
) -> EvalReport:
    """Execute model evaluation suite across all items and aggregate metrics."""
    items_to_eval = dataset if dataset is not None else DEFAULT_EVAL_DATASET
    results: list[EvalItemResult] = []

    for item in items_to_eval:
        res = evaluate_single_item(
            api_url=api_url,
            api_key=api_key,
            model=model,
            item=item,
            timeout=timeout,
        )
        results.append(res)

    total = len(results)
    matched = sum(1 for r in results if r.matched)
    accuracy = (matched / total) if total > 0 else 0.0

    all_losses = [r.cross_entropy for r in results if r.cross_entropy is not None and math.isfinite(r.cross_entropy)]
    mean_loss = sum(all_losses) / len(all_losses) if all_losses else None
    overall_ppl = math.exp(min(max(mean_loss, 0.0), 100.0)) if mean_loss is not None else None

    mean_per_token_latency = sum(r.per_token_latency_ms for r in results) / total if total > 0 else 0.0
    mean_tps = (
        sum(r.decode_tokens_per_second for r in results) / total
        if total > 0
        else 0.0
    )

    # Breakdown by category
    categories: dict[str, dict[str, float]] = {}
    cats = sorted(set(r.category for r in results))
    for c in cats:
        cat_items = [r for r in results if r.category == c]
        cat_total = len(cat_items)
        cat_matched = sum(1 for r in cat_items if r.matched)
        cat_losses = [r.cross_entropy for r in cat_items if r.cross_entropy is not None and math.isfinite(r.cross_entropy)]
        cat_mean_loss = sum(cat_losses) / len(cat_losses) if cat_losses else None
        categories[c] = {
            "total": cat_total,
            "matched": cat_matched,
            "accuracy": cat_matched / cat_total if cat_total > 0 else 0.0,
            "cross_entropy": cat_mean_loss,
            "perplexity": math.exp(min(max(cat_mean_loss, 0.0), 100.0)) if cat_mean_loss is not None else None,
        }

    total_tokens = sum(r.completion_tokens for r in results)

    return EvalReport(
        model=model,
        total_items=total,
        matched_items=matched,
        accuracy=accuracy,
        mean_cross_entropy=mean_loss,
        perplexity=overall_ppl,
        mean_per_token_latency_ms=mean_per_token_latency,
        mean_tokens_per_second=mean_tps,
        total_prompt_tokens=sum(r.prompt_tokens for r in results),
        total_completion_tokens=total_tokens,
        categories=categories,
        items=results,
    )


def format_optional(value, spec):
    return "N/A" if value is None else format(value, spec)


def format_text_report(report: EvalReport) -> str:
    """Format evaluation results as a human-readable ASCII table."""
    lines = [
        "=" * 78,
        f"  HawkPoint NPU Model Quality & Perplexity Evaluation: {report.model}",
        "=" * 78,
        f"  Items Evaluated:   {report.total_items}",
        f"  Answer Accuracy:   {report.accuracy * 100:.1f}% ({report.matched_items}/{report.total_items})",
        f"  Cross-Entropy:     {format_optional(report.mean_cross_entropy, ".4f")} nats/token",
        f"  Perplexity (PPL):  {format_optional(report.perplexity, ".2f")}",
        f"  Mean Per-Token Latency:         {report.mean_per_token_latency_ms:.1f} ms",
        f"  Mean Throughput:   {report.mean_tokens_per_second:.1f} tok/s",
        "-" * 78,
        "  Category Breakdown:",
        f"  {'Category':<16} {'Accuracy':<12} {'Loss':<12} {'Perplexity':<12}",
        "-" * 78,
    ]
    for cat, stats in report.categories.items():
        lines.append(
            f"  {cat:<16} {stats['accuracy'] * 100:>5.1f}% ({stats['matched']}/{int(stats['total'])})   "
            f"{format_optional(stats['cross_entropy'], ">6.4f")}     {format_optional(stats['perplexity'], ">7.2f")}"
        )
    lines.extend([
        "-" * 78,
        "  Item Details:",
        f"  {'ID':<20} {'Category':<14} {'Status':<10} {'Loss':<8} {'PPL':<8} {'Output'}",
        "-" * 78,
    ])
    for it in report.items:
        status = "PASS" if it.matched else "FAIL"
        clean_out = it.generated_text.replace("\n", "\\n")[:24]
        lines.append(
            f"  {it.item_id:<20} {it.category:<14} {status:<10} {format_optional(it.cross_entropy, ">6.3f")}  {format_optional(it.perplexity, ">6.2f")}  {clean_out}"
        )
    lines.append("=" * 78)
    return "\n".join(lines)


def format_markdown_report(report: EvalReport) -> str:
    """Format evaluation results as a structured GitHub Markdown document."""
    lines = [
        f"# Model Evaluation Report: `{report.model}`",
        "",
        "## Summary Metrics",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| **Model** | `{report.model}` |",
        f"| **Total Evaluated Items** | {report.total_items} |",
        f"| **Answer Accuracy** | **{report.accuracy * 100:.1f}%** ({report.matched_items}/{report.total_items}) |",
        f"| **Mean Cross-Entropy** | {format_optional(report.mean_cross_entropy, ".4f")} nats/token |",
        f"| **Perplexity (PPL)** | **{format_optional(report.perplexity, ".2f")}** |",
        f"| **Mean Per-Token Latency** | {report.mean_per_token_latency_ms:.1f} ms |",
        f"| **Mean Decode Speed** | {report.mean_tokens_per_second:.1f} tok/s |",
        "",
        "## Category Performance",
        "",
        "| Category | Items | Matched | Accuracy | Loss | Perplexity |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for cat, stats in report.categories.items():
        lines.append(
            f"| `{cat}` | {int(stats['total'])} | {int(stats['matched'])} | "
            f"{stats['accuracy'] * 100:.1f}% | {format_optional(stats['cross_entropy'], ".4f")} | {format_optional(stats['perplexity'], ".2f")} |"
        )
    lines.extend([
        "",
        "## Item-Level Results",
        "",
        "| ID | Category | Status | Expected | Generated | Loss | PPL |",
        "| :--- | :--- | :---: | :--- | :--- | :---: | :---: |",
    ])
    for it in report.items:
        status = "**PASS**" if it.matched else "*FAIL*"
        clean_exp = it.expected.replace("\n", " ").replace("|", "\\|")
        clean_gen = it.generated_text.replace("\n", " ").replace("|", "\\|")
        lines.append(
            f"| `{it.item_id}` | `{it.category}` | {status} | `{clean_exp}` | `{clean_gen}` | "
            f"{format_optional(it.cross_entropy, ".3f")} | {format_optional(it.perplexity, ".2f")} |"
        )
    lines.append("")
    return "\n".join(lines)


class MockEvalServer(BaseHTTPRequestHandler):
    """In-process mock server providing deterministic completions for self-testing."""

    def log_message(self, format, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}

        if body.get("logprobs") is not True:
            self.send_error(400, "logprobs must be a boolean")
            return
        prompt = body.get("prompt", "")
        # Return deterministic expected answers matching prompt expectations
        if "capital of France" in prompt:
            text = " Paris, France."
            logprobs = [-0.15, -0.05, -0.01]
            tokens = [" Paris", ",", " France"]
        elif "Red Planet" in prompt:
            text = " Mars."
            logprobs = [-0.10, -0.02]
            tokens = [" Mars", "."]
        elif "15 + 27" in prompt:
            text = " 42"
            logprobs = [-0.05]
            tokens = [" 42"]
        elif "all roses are" in prompt:
            text = " plants."
            logprobs = [-0.08, -0.01]
            tokens = [" plants", "."]
        elif "def add" in prompt:
            text = "return a + b"
            logprobs = [-0.02, -0.01, -0.01, -0.01]
            tokens = ["return", " a", " +", " b"]
        else:
            text = " dog."
            logprobs = [-0.05, -0.01]
            tokens = [" dog", "."]

        response_body = {
            "id": "cmpl-mock-eval",
            "object": "text_completion",
            "created": int(time.time()),
            "model": body.get("model", "smollm2-135m-xdna1"),
            "choices": [
                {
                    "index": 0,
                    "text": text,
                    "logprobs": {
                        "tokens": tokens,
                        "token_logprobs": logprobs,
                        "top_logprobs": None,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(tokens),
                "total_tokens": len(prompt.split()) + len(tokens),
            },
        }

        encoded = json.dumps(response_body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def run_self_test() -> int:
    """Run self-test with an in-memory mock server."""
    server = HTTPServer(("127.0.0.1", 0), MockEvalServer)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        report = run_evaluation(
            api_url=f"http://127.0.0.1:{port}",
            api_key="mock-test-key",
            model="smollm2-135m-xdna1",
            timeout=5.0,
        )
        assert report.total_items == len(DEFAULT_EVAL_DATASET)
        assert report.matched_items == len(DEFAULT_EVAL_DATASET)
        assert report.accuracy == 1.0
        assert report.mean_cross_entropy < 0.2
        assert report.perplexity < 1.5

        # Format tests
        text_out = format_text_report(report)
        assert "HawkPoint NPU Model Quality" in text_out
        assert "100.0%" in text_out

        md_out = format_markdown_report(report)
        assert "# Model Evaluation Report" in md_out
        assert "**100.0%**" in md_out

        print("PASS 1 model evaluation self-test")
        return 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def main(argv: list[str] | None = None) -> int:
    """Parse CLI options, run model evaluation, and report quality metrics."""
    parser = argparse.ArgumentParser(
        description="HawkPoint NPU Model Quality, Accuracy, and Perplexity Evaluator"
    )
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8000",
        help="Base URL of HawkPoint NPU API server (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("HAWKPOINT_API_KEY"),
        help="Bearer token; reads HAWKPOINT_API_KEY when omitted",
    )
    parser.add_argument(
        "--model",
        default="smollm2-135m-xdna1",
        help="Model ID to evaluate (default: smollm2-135m-xdna1)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Path to custom evaluation dataset JSON or JSONL file",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json", "markdown"],
        default="text",
        help="Output report format (default: text)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="File path to write report output",
    )
    parser.add_argument(
        "--max-perplexity",
        type=float,
        help="Quality SLA: fail if overall perplexity exceeds this value",
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        help="Quality SLA: fail if answer accuracy is below this threshold (0.0 - 1.0)",
    )
    parser.add_argument(
        "--min-tps",
        type=float,
        help="Performance SLA: fail if mean decode speed is below this value (tok/s)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run in-process mock server self-test",
    )

    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test()

    dataset = None
    if args.dataset:
        if not args.dataset.is_file():
            parser.error(f"dataset file not found: {args.dataset}")
        content = args.dataset.read_text(encoding="utf-8")
        if args.dataset.suffix == ".jsonl":
            dataset = [json.loads(line) for line in content.splitlines() if line.strip()]
        else:
            dataset = json.loads(content)

    try:
        report = run_evaluation(
            api_url=args.api_url,
            api_key=args.api_key,
            model=args.model,
            dataset=dataset,
        )
    except (URLError, HTTPError, OSError, ValueError, RuntimeError) as exc:
        print(f"Error: failed to connect to API server at {args.api_url}: {exc}", file=sys.stderr)
        return 1

    if args.format == "json":
        output_str = json.dumps(asdict(report), indent=2, allow_nan=False)
    elif args.format == "markdown":
        output_str = format_markdown_report(report)
    else:
        output_str = format_text_report(report)

    if args.output:
        args.output.write_text(output_str, encoding="utf-8")
        print(f"Report written to {args.output}")
    else:
        print(output_str)

    # Check SLA gates
    failed_slas = []
    if args.max_perplexity is not None and report.perplexity is None:
        failed_slas.append("Perplexity is unavailable: the API returned no valid token logprobs")
    elif args.max_perplexity is not None and report.perplexity > args.max_perplexity:
        failed_slas.append(
            f"Perplexity {format_optional(report.perplexity, ".2f")} exceeded threshold {args.max_perplexity:.2f}"
        )
    if args.min_accuracy is not None and report.accuracy < args.min_accuracy:
        failed_slas.append(
            f"Accuracy {report.accuracy * 100:.1f}% below minimum {args.min_accuracy * 100:.1f}%"
        )
    if args.min_tps is not None and report.mean_tokens_per_second < args.min_tps:
        failed_slas.append(
            f"Throughput {report.mean_tokens_per_second:.1f} tok/s below minimum {args.min_tps:.1f} tok/s"
        )

    if failed_slas:
        for failure in failed_slas:
            print(f"SLA VIOLATION: {failure}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
