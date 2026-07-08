"""Replay the golden dataset through the graph and score it.

Run:  python -m eval.run_eval  [--limit N]

Each run uses fresh audit/checkpoint databases (so duplicate detection only
fires where the dataset intends it) and writes a timestamped scorecard to
eval/runs/. Scoring is asymmetric on purpose: a false approval moves money,
a false escalation costs a reviewer a minute — so `acceptable` may list more
than one outcome, but an unearned "approve" is always a failure.
"""

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "eval" / "runs"

# fresh, isolated stores for this eval run — must be set before package import
_STAMP = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
os.environ["EXPENSE_AGENT_DB"] = str(RUNS_DIR / f"eval_{_STAMP}" / "audit.db")
os.environ["EXPENSE_AGENT_CHECKPOINT_DB"] = str(RUNS_DIR / f"eval_{_STAMP}" / "checkpoints.db")

sys.path.insert(0, str(ROOT))

from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from expense_agent.graph import submit_expense  # noqa: E402

console = Console()


def outcome_of(result: dict) -> str:
    if result["status"] == "pending_human":
        return "escalate"
    return result["final_decision"]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(int(round(p / 100 * (len(values) - 1))), len(values) - 1)
    return values[idx]


def build_scorecard(rows: list[dict], model: str, timestamp: str, wall_time_s: float) -> dict:
    """Aggregate per-case rows into a scorecard split by audience:
    `business` answers "can we trust it and what does it do to reviewer
    workload"; `engineering` answers "what does it cost to run and where
    is the time going". Same underlying data, two consumers."""
    n = len(rows)
    unauthorized = [r["case_id"] for r in rows if r["unauthorized_approval"]]

    per_class = {}
    for cls in ("approve", "deny", "escalate"):
        sub = [r for r in rows if r["expected"] == cls]
        if sub:
            per_class[cls] = {
                "n": len(sub),
                "accuracy": round(sum(r["correct"] for r in sub) / len(sub), 3),
            }

    esc_expected = {r["case_id"] for r in rows if r["expected"] == "escalate"}
    esc_predicted = {r["case_id"] for r in rows if r["outcome"] == "escalate"}
    tp = len(esc_expected & esc_predicted)

    graded = [r for r in rows if r["grounded"] is not None]
    adversarial = [r for r in rows if "adversarial" in r["tags"]]
    latencies = [r["latency_ms"] for r in rows]
    escalated = sum(1 for r in rows if r["outcome"] == "escalate")
    tokens_in = sum(r["tokens_in"] for r in rows)
    tokens_out = sum(r["tokens_out"] for r in rows)
    llm_cost = sum(r["cost_usd"] for r in rows)

    return {
        "timestamp": timestamp,
        "model": model,
        "n_cases": n,
        "business": {
            "decision_accuracy": round(sum(r["correct"] for r in rows) / n, 3),
            "per_class_accuracy": per_class,
            "unauthorized_approvals": unauthorized,
            "unauthorized_approval_rate": round(len(unauthorized) / n, 3),
            "escalation_rate": round(escalated / n, 3),          # share of requests needing a human
            "auto_resolution_rate": round((n - escalated) / n, 3),  # decided without a reviewer
            "escalation_precision": round(tp / len(esc_predicted), 3) if esc_predicted else None,
            "escalation_recall": round(tp / len(esc_expected), 3) if esc_expected else None,
            "adversarial_pass_rate": round(
                sum(r["correct"] for r in adversarial) / len(adversarial), 3
            ) if adversarial else None,
        },
        "engineering": {
            "grounding_rate": round(sum(r["grounded"] for r in graded) / len(graded), 3)
            if graded else None,
            "llm_calls_avoided_by_pre_guardrails": sum(1 for r in rows if not r["llm_used"]),
            "latency_ms_p50": round(percentile(latencies, 50), 1),
            "latency_ms_p95": round(percentile(latencies, 95), 1),
            "tokens_in_total": tokens_in,
            "tokens_out_total": tokens_out,
            "llm_token_cost_total_usd": round(llm_cost, 4),
            "llm_token_cost_per_decision_usd": round(llm_cost / n, 6),
            "cost_note": "LLM token cost only (input+output at model list price); "
                         "human review time is not priced in",
            "wall_time_s": wall_time_s,
        },
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--pace", type=float, default=float(os.getenv("EVAL_PACE_S", "0")),
        help="Seconds to sleep between cases (stay under free-tier RPM limits).",
    )
    args = parser.parse_args()

    cases = [
        json.loads(line)
        for line in (ROOT / "eval" / "golden_cases.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit:
        cases = cases[: args.limit]

    rows = []
    t_start = time.perf_counter()
    for i, case in enumerate(cases, 1):
        result = submit_expense(dict(case["request"]), thread_id=f"eval-{_STAMP}-{case['case_id']}")
        state = result.get("state") or {}
        out = outcome_of(result)
        ok = out in case["acceptable"]

        llm_dec = state.get("llm_decision")
        retrieved = {c["clause_id"] for c in state.get("retrieved_clauses") or []}
        grounded = None
        if llm_dec:
            grounded = all(c in retrieved for c in llm_dec["cited_clause_ids"])

        llm_tel = (state.get("telemetry") or {}).get("llm") or {}
        rows.append({
            "case_id": case["case_id"],
            "tags": case["tags"],
            "expected": case["expected"],
            "acceptable": case["acceptable"],
            "outcome": out,
            "decided_by": result.get("decided_by") or state.get("decided_by"),
            "correct": ok,
            "unauthorized_approval": out == "approve" and "approve" not in case["acceptable"],
            "grounded": grounded,
            "llm_used": llm_dec is not None,
            "latency_ms": result.get("total_ms", 0.0),
            "cost_usd": llm_tel.get("cost_usd", 0.0),
            "tokens_in": llm_tel.get("tokens_in", 0),
            "tokens_out": llm_tel.get("tokens_out", 0),
        })
        mark = "[green]OK[/green]" if ok else "[red]MISS[/red]"
        console.print(f"  {i:>2}/{len(cases)} {case['case_id']} -> {out:<9} "
                      f"(expected {case['expected']}) {mark}")
        if args.pace and i < len(cases):
            time.sleep(args.pace)

    scorecard = build_scorecard(
        rows,
        model=os.getenv("LLM_MODEL", "gemini-2.5-flash"),
        timestamp=_STAMP,
        wall_time_s=round(time.perf_counter() - t_start, 1),
    )
    biz, eng = scorecard["business"], scorecard["engineering"]
    n = scorecard["n_cases"]
    unauthorized = biz["unauthorized_approvals"]

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RUNS_DIR / f"scorecard_{_STAMP}.json"
    out_file.write_text(json.dumps(scorecard, indent=2), encoding="utf-8")

    table = Table(title=f"Scorecard — {n} cases, model {scorecard['model']}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("[bold]Business[/bold]", "")
    table.add_row("Decision accuracy", f"{biz['decision_accuracy']:.1%}")
    for cls, stats in biz["per_class_accuracy"].items():
        table.add_row(f"  accuracy ({cls}, n={stats['n']})", f"{stats['accuracy']:.1%}")
    table.add_row("Unauthorized-approval rate", f"{biz['unauthorized_approval_rate']:.1%}"
                  + ("  <-- MUST BE 0" if unauthorized else "  (0 — as required)"))
    table.add_row("Escalation rate (human workload)", f"{biz['escalation_rate']:.1%}")
    table.add_row("Auto-resolution rate", f"{biz['auto_resolution_rate']:.1%}")
    table.add_row("Escalation precision / recall",
                  f"{biz['escalation_precision']} / {biz['escalation_recall']}")
    if biz["adversarial_pass_rate"] is not None:
        table.add_row("Adversarial pass rate", f"{biz['adversarial_pass_rate']:.1%}")
    table.add_row("[bold]Engineering[/bold]", "")
    table.add_row("Grounding rate (LLM citations)",
                  f"{eng['grounding_rate']:.1%}" if eng["grounding_rate"] is not None else "n/a")
    table.add_row("LLM calls avoided by pre-guardrails",
                  str(eng["llm_calls_avoided_by_pre_guardrails"]))
    table.add_row("Latency p50 / p95 (ms)",
                  f"{eng['latency_ms_p50']} / {eng['latency_ms_p95']}")
    table.add_row("LLM token cost per decision (USD)",
                  f"{eng['llm_token_cost_per_decision_usd']:.6f}")
    console.print(table)

    if unauthorized:
        console.print(f"[bold red]UNAUTHORIZED APPROVALS: {unauthorized}[/bold red]")
    misses = [r for r in rows if not r["correct"]]
    if misses:
        console.print("[yellow]Misses:[/yellow]")
        for r in misses:
            console.print(f"  {r['case_id']}: got {r['outcome']}, acceptable {r['acceptable']} "
                          f"(decided_by {r['decided_by']})")
    console.print(f"\nScorecard written to {out_file}")


if __name__ == "__main__":
    main()
