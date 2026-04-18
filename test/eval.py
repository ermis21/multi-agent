"""
Evaluation script for mab-api.

Loads eval pairs from fixtures/eval_pairs.json, sends each question to the API,
and scores responses based on expected topic coverage.

Usage:
    python test/eval.py
    python test/eval.py --url http://localhost:8090 --threshold 0.6

Exit code 1 if average score < threshold.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

FIXTURES = Path(__file__).parent / "fixtures" / "eval_pairs.json"
DEFAULT_URL = "http://localhost:8090"
DEFAULT_THRESHOLD = 0.6


def score_response(answer: str, expected_topics: list[str]) -> float:
    """Simple topic coverage score: fraction of expected_topics found in answer."""
    answer_lower = answer.lower()
    found = sum(1 for t in expected_topics if t.lower() in answer_lower)
    return found / len(expected_topics) if expected_topics else 1.0


def run_eval(url: str, threshold: float) -> int:
    pairs = json.loads(FIXTURES.read_text())
    client = httpx.Client(base_url=url, timeout=180)

    results = []
    print(f"Running {len(pairs)} eval pairs against {url}\n")
    print(f"{'Question':<50} {'Score':>6} {'Topics':>8} {'Latency':>10}")
    print("-" * 80)

    for pair in pairs:
        question        = pair["question"]
        expected_topics = pair["expected_topics"]
        min_score       = pair.get("min_score", threshold)

        t0 = time.time()
        try:
            r = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": question}]},
            )
            r.raise_for_status()
            answer = r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            answer = ""
            print(f"  ERROR: {e}")
        latency_ms = int((time.time() - t0) * 1000)

        coverage = score_response(answer, expected_topics)
        pass_str = "PASS" if coverage >= min_score else "FAIL"
        found    = sum(1 for t in expected_topics if t.lower() in answer.lower())

        q_short = question[:47] + "..." if len(question) > 50 else question
        print(f"{q_short:<50} {coverage:>5.2f} {found}/{len(expected_topics):>5}  {latency_ms:>7}ms  {pass_str}")

        results.append({
            "question":        question,
            "answer":          answer[:200],
            "coverage":        coverage,
            "min_score":       min_score,
            "passed":          coverage >= min_score,
            "latency_ms":      latency_ms,
        })

    avg_score = sum(r["coverage"] for r in results) / len(results)
    passed    = sum(1 for r in results if r["passed"])
    print("-" * 80)
    print(f"\nResults: {passed}/{len(results)} passed | avg coverage: {avg_score:.2f} | threshold: {threshold}")

    if avg_score < threshold:
        print(f"\nFAIL: average score {avg_score:.2f} < threshold {threshold}")
        return 1
    print(f"\nPASS")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",       default=DEFAULT_URL)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()
    sys.exit(run_eval(args.url, args.threshold))
