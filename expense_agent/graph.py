"""The LangGraph state machine.

    intake -> pre_guardrails -> retrieve -> reason -> post_guardrails -> finalize
                 |  \\                                      |
                 |   `-(hard violation)--------------> hitl (interrupt)
                 `---(deny)---------------> finalize <-----'

LangGraph is used for two properties a plain chain cannot give:
durable human-in-the-loop pauses (interrupt + checkpointer) and
replayable, explicit control flow.
"""

import sqlite3
import time
import uuid
from datetime import datetime, timezone
from typing import Optional, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import ValidationError

from . import audit, config, guardrails, llm, retrieval
from .schemas import ExpenseRequest


class AgentState(TypedDict, total=False):
    run_id: str
    request: dict                 # normalized ExpenseRequest dump
    guardrail_events: list[dict]
    retrieved_clauses: list[dict]
    llm_decision: Optional[dict]
    final_decision: Optional[str]
    decided_by: Optional[str]
    pending_reason: Optional[str]
    human: Optional[dict]
    telemetry: dict


def _node_timing(state: AgentState, node: str, t0: float) -> dict:
    tel = dict(state.get("telemetry") or {})
    nodes = dict(tel.get("node_latency_ms") or {})
    nodes[node] = round((time.perf_counter() - t0) * 1000, 1)
    tel["node_latency_ms"] = nodes
    return tel


def _request(state: AgentState) -> ExpenseRequest:
    return ExpenseRequest.model_validate(state["request"])


# --- Nodes ---

def intake(state: AgentState) -> dict:
    t0 = time.perf_counter()
    try:
        req = ExpenseRequest.model_validate(state["request"]).normalized()
        return {"request": req.model_dump(), "telemetry": _node_timing(state, "intake", t0)}
    except ValidationError as exc:
        return {
            "final_decision": "deny",
            "decided_by": "system:validation",
            "guardrail_events": [{
                "rule_id": "GR-INTAKE-SCHEMA", "stage": "pre", "action": "deny",
                "detail": f"Request failed schema validation: {exc.error_count()} error(s).",
            }],
            "telemetry": _node_timing(state, "intake", t0),
        }


def pre_guardrails(state: AgentState) -> dict:
    t0 = time.perf_counter()
    req = _request(state)
    dup = audit.duplicate_exists(req.employee_id, req.vendor, req.amount, req.request_id)
    events = guardrails.pre_check(req, duplicate_exists=dup)

    update: dict = {
        "guardrail_events": [e.model_dump() for e in events],
        "telemetry": _node_timing(state, "pre_guardrails", t0),
    }
    denies = [e for e in events if e.action == "deny"]
    escalates = [e for e in events if e.action == "escalate"]
    if denies:
        update["final_decision"] = "deny"
        update["decided_by"] = f"guardrail:{denies[0].rule_id}"
    elif escalates:
        update["final_decision"] = "escalate"
        update["decided_by"] = f"guardrail:{escalates[0].rule_id}"
        update["pending_reason"] = escalates[0].detail
    return update


def retrieve_policy(state: AgentState) -> dict:
    t0 = time.perf_counter()
    clauses = retrieval.retrieve(_request(state))
    return {
        "retrieved_clauses": clauses,
        "telemetry": _node_timing(state, "retrieve_policy", t0),
    }


def reason(state: AgentState) -> dict:
    t0 = time.perf_counter()
    decision, llm_tel = llm.decide(_request(state), state["retrieved_clauses"])
    tel = _node_timing(state, "reason", t0)
    tel["llm"] = llm_tel
    return {
        "llm_decision": decision.model_dump() if decision else None,
        "telemetry": tel,
    }


def post_guardrails(state: AgentState) -> dict:
    t0 = time.perf_counter()
    req = _request(state)
    raw = state.get("llm_decision")
    retrieved_ids = [c["clause_id"] for c in state.get("retrieved_clauses") or []]

    if raw is None:
        events = []
        final, decided_by = "escalate", "system:llm_failure"
        reason_txt = "LLM reasoning step failed schema validation twice; failing closed."
    else:
        decision = llm.LLMDecision.model_validate(raw)
        events = guardrails.post_check(req, decision, retrieved_ids)
        final, decided_by = guardrails.resolve_final(decision, events)
        reason_txt = events[0].detail if events else (
            decision.justification if final == "escalate" else None
        )

    return {
        "guardrail_events": (state.get("guardrail_events") or []) + [e.model_dump() for e in events],
        "final_decision": final,
        "decided_by": decided_by,
        "pending_reason": reason_txt if final == "escalate" else None,
        "telemetry": _node_timing(state, "post_guardrails", t0),
    }


def hitl(state: AgentState) -> dict:
    """Durable pause: the graph checkpoint persists here until a reviewer
    resumes it. The human decision is final and attributed."""
    answer = interrupt({
        "request": state["request"],
        "reason": state.get("pending_reason"),
        "llm_decision": state.get("llm_decision"),
        "guardrail_events": state.get("guardrail_events") or [],
    })
    return {
        "final_decision": answer["decision"],
        "decided_by": f"human:{answer.get('reviewer', 'unknown')}",
        "human": answer,
    }


def finalize(state: AgentState) -> dict:
    llm_tel = (state.get("telemetry") or {}).get("llm") or {}
    record = {
        "run_id": state["run_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "request": state["request"],
        "retrieved_clauses": state.get("retrieved_clauses") or [],
        "llm_decision": state.get("llm_decision"),
        "guardrail_events": state.get("guardrail_events") or [],
        "human": state.get("human"),
        "final_decision": state["final_decision"],
        "decided_by": state["decided_by"],
        "telemetry": {
            **(state.get("telemetry") or {}),
            "cost_usd": llm_tel.get("cost_usd", 0.0),
            "tokens_in": llm_tel.get("tokens_in", 0),
            "tokens_out": llm_tel.get("tokens_out", 0),
        },
    }
    audit.write_run(record)
    return {}


# --- Routing ---

def route_after_intake(state: AgentState) -> str:
    return "finalize" if state.get("final_decision") else "pre_guardrails"


def route_after_pre(state: AgentState) -> str:
    fd = state.get("final_decision")
    if fd == "deny":
        return "finalize"
    if fd == "escalate":
        return "hitl"
    return "retrieve_policy"


def route_after_post(state: AgentState) -> str:
    return "hitl" if state.get("final_decision") == "escalate" else "finalize"


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        builder = StateGraph(AgentState)
        builder.add_node("intake", intake)
        builder.add_node("pre_guardrails", pre_guardrails)
        builder.add_node("retrieve_policy", retrieve_policy)
        builder.add_node("reason", reason)
        builder.add_node("post_guardrails", post_guardrails)
        builder.add_node("hitl", hitl)
        builder.add_node("finalize", finalize)

        builder.add_edge(START, "intake")
        builder.add_conditional_edges("intake", route_after_intake,
                                      ["pre_guardrails", "finalize"])
        builder.add_conditional_edges("pre_guardrails", route_after_pre,
                                      ["retrieve_policy", "hitl", "finalize"])
        builder.add_edge("retrieve_policy", "reason")
        builder.add_edge("reason", "post_guardrails")
        builder.add_conditional_edges("post_guardrails", route_after_post,
                                      ["hitl", "finalize"])
        builder.add_edge("hitl", "finalize")
        builder.add_edge("finalize", END)

        config.CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(config.CHECKPOINT_DB, check_same_thread=False)
        _graph = builder.compile(checkpointer=SqliteSaver(conn))
    return _graph


# --- Service layer ---

def submit_expense(request: dict, thread_id: str | None = None) -> dict:
    """Run a request through the graph. Returns a completed decision, or a
    pending_human handle if the run parked at the HITL interrupt."""
    thread_id = thread_id or request.get("request_id") or uuid.uuid4().hex
    run_id = uuid.uuid4().hex
    cfg = {"configurable": {"thread_id": thread_id}}
    t0 = time.perf_counter()
    result = get_graph().invoke({"request": request, "run_id": run_id, "telemetry": {}}, cfg)
    total_ms = round((time.perf_counter() - t0) * 1000, 1)

    if "__interrupt__" in result:
        reason_txt = result.get("pending_reason") or "Escalated for manual review."
        audit.add_pending(thread_id, run_id, result["request"], reason_txt)
        # provisional audit record so pending runs are traceable too
        audit.write_run({
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "pending_human",
            "request": result["request"],
            "retrieved_clauses": result.get("retrieved_clauses") or [],
            "llm_decision": result.get("llm_decision"),
            "guardrail_events": result.get("guardrail_events") or [],
            "human": None,
            "final_decision": "escalate",
            "decided_by": result.get("decided_by"),
            "telemetry": {**(result.get("telemetry") or {}), "total_ms": total_ms},
        })
        return {"status": "pending_human", "thread_id": thread_id, "run_id": run_id,
                "reason": reason_txt, "total_ms": total_ms, "state": result}

    return {"status": "completed", "thread_id": thread_id, "run_id": run_id,
            "final_decision": result.get("final_decision"),
            "decided_by": result.get("decided_by"),
            "total_ms": total_ms, "state": result}


def resume_expense(thread_id: str, decision: str, reviewer: str) -> dict:
    """Resume a parked run with the human decision."""
    if decision not in ("approve", "deny"):
        raise ValueError("Human decision must be 'approve' or 'deny'.")
    cfg = {"configurable": {"thread_id": thread_id}}
    result = get_graph().invoke(
        Command(resume={"decision": decision, "reviewer": reviewer}), cfg
    )
    audit.resolve_pending(thread_id, decision, reviewer)
    return {"status": "completed", "thread_id": thread_id,
            "final_decision": result.get("final_decision"),
            "decided_by": result.get("decided_by"), "state": result}
