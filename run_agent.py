#!/usr/bin/env python3
from __future__ import annotations
import json
import uuid
from typing import Any, List, Literal, Annotated, Tuple, Optional
from typing_extensions import TypedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
import langgraph.errors
import argparse, sys, logging
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
import langgraph 
from utils.llm import (
    get_search_tool,
    load_llm,
    load_vectorstore,
    get_multi_queries_retrival_tool,
    get_reviews_tool,
)
from utils.prompt import (
    build_final_prompt,
    build_summary_prompt,
    build_plan_prompt,
)
try:
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from phoenix.otel import register as phoenix_register
    HAVE_PHOENIX = True
except Exception:
    HAVE_PHOENIX = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("batch-agent")

MAX_TOOL_LOOPS = 20

class PlanningState(TypedDict):
    messages: Annotated[list, add_messages]
    summary: str
    next_step: Literal["retrieve", "web_search", "final_answer", "reviews"]
    tool_loops: int

def _pretty(obj: Any) -> str:
    if isinstance(obj, BaseMessage):
        return str(obj.content)
    if isinstance(obj, list):
        return "\n\n".join(_pretty(i) for i in obj)
    return str(getattr(obj, "content", getattr(obj, "text", obj)))

def _is_tool_call(msg: BaseMessage) -> bool:
    return (
        isinstance(msg, AIMessage)
        and (
            getattr(msg, "tool_calls", None)
            or msg.additional_kwargs.get("tool_use")
        )
    )

def _extract_tool_calls(msgs: List[BaseMessage]) -> str:
    """Summarize tool invocations like the UI (for plan prompt)."""
    chunks: List[str] = []
    for m in msgs:
        for call in getattr(m, "tool_calls", []) or []:
            name = getattr(call, "name", None) or call.get("name", "")
            args = getattr(call, "args", None) or call.get("args", {})
            chunks.append(f'tool: "{name}" args: {json.dumps(args, default=str)}')
        if isinstance(getattr(m, "content", None), dict):
            d = m.content
            if "tool" in d and "args" in d:
                chunks.append(f'tool: "{d["tool"]}" args: {json.dumps(d["args"], default=str)}')
    return "\n".join(chunks)

logger.info("Loading vector stores & LLM (UI-parity)…")
vectorstore, vectorstore_reviews = load_vectorstore()
retriever_tool = get_multi_queries_retrival_tool(vectorstore)
search_tool = get_search_tool()
search_tool.name = "web_search"
reviews_tool = get_reviews_tool(vectorstore_reviews)
reviews_tool.name = "reviews"
llm = load_llm()
llm_with_tools = llm.bind_tools([retriever_tool, search_tool, reviews_tool])

if HAVE_PHOENIX:
    try:
        tp = phoenix_register(
            project_name="agent",
            endpoint="http://127.0.0.1:6006/v1/traces",
            batch=True,
            auto_instrument=False,
        )
        try:
            LangChainInstrumentor().uninstrument()
        except Exception:
            pass
        LangChainInstrumentor().instrument(tracer_provider=tp)
        logger.info("✅ Phoenix tracing enabled.")
    except Exception as e:
        logger.warning(f"Phoenix tracing disabled: {e}")
else:
    logger.info("Phoenix not available; running without tracing.")


def generate_query_or_respond(state: PlanningState):
    # init on first call
    if "tool_loops" not in state:
        state["tool_loops"] = 0

    recent = state["messages"][-2:]
    human_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    question = human_msgs[-1].content if human_msgs else "Unknown question"
    current_summary = state.get("summary", "Empty")

    summary_prompt = build_summary_prompt(current_summary, recent, question)
    new_summary_msg = llm.invoke([HumanMessage(summary_prompt)])
    new_summary = new_summary_msg.content
    state["summary"] = new_summary

    attempts = _extract_tool_calls(state["messages"])
    plan_prompt = build_plan_prompt(new_summary, question, attempts)
    plan_msg = llm_with_tools.invoke([HumanMessage(plan_prompt)])

    calls = getattr(plan_msg, "tool_calls", None) or []
    if calls:
        name = calls[0]["name"]
        decision = (
            "retrieve"   if name == "multi_query_retrieval" else
            "web_search" if name == "web_search" else
            "reviews"    if name == "reviews" else
            "final_answer"
        )
    else:
        decision = "final_answer"

    if decision in {"retrieve", "web_search", "reviews"}:
        state["tool_loops"] += 1
        if state["tool_loops"] >= MAX_TOOL_LOOPS:
            decision = "final_answer"

    return {
        "messages":  [new_summary_msg, plan_msg],
        "summary":   new_summary,
        "next_step": decision,
        "tool_loops": state["tool_loops"],
    }

def generate_answer(state: PlanningState):
    human_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    question = human_msgs[-1].content if human_msgs else "Unknown question"
    if "summary" not in state:
        state["summary"] = "Empty"
    context = state["summary"]
    prompt = build_final_prompt(question, context)
    final = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [final]}

def route_decision(state: PlanningState) -> Literal["retrieve", "web_search", "reviews", END]:
    plan = state.get("next_step", "")
    if plan == "retrieve":
        return "retrieve"
    if plan == "web_search":
        return "web_search"
    if plan == "reviews":
        return "reviews"
    return END

logger.info("Compiling LangGraph (UI-parity)…")
workflow = StateGraph(state_schema=PlanningState)
workflow.add_node("plan_or_answer", generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node("reviews", ToolNode([reviews_tool]))
workflow.add_node("web_search", ToolNode([search_tool]))
workflow.add_node("final_answer", generate_answer)
workflow.add_edge(START, "plan_or_answer")
workflow.add_conditional_edges(
    "plan_or_answer",
    route_decision,
    {
        "retrieve": "retrieve",
        "web_search": "web_search",
        "reviews": "reviews",
        END: "final_answer",
    },
)
workflow.add_edge("retrieve", "plan_or_answer")
workflow.add_edge("reviews", "plan_or_answer")
workflow.add_edge("web_search", "plan_or_answer")
workflow.add_edge("final_answer", END)
graph = workflow.compile()
logger.info("Graph compiled.")

def run_agent(prompt: str) -> str:
    """Run one prompt through the UI-parity graph and return the assistant text."""
    thread_id = uuid.uuid4().hex
    state: PlanningState = {
        "messages": [("user", prompt)],
        "summary": "Empty",
        "next_step": "final_answer",
        "tool_loops": 0,
    }
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 1000}
    final_state = graph.invoke(state, config=config)

    # last assistant message that isn't a tool call
    for msg in reversed(final_state.get("messages", [])):  # type: ignore[index]
        if isinstance(msg, AIMessage) and not _is_tool_call(msg):
            return (str(msg.content) or "").strip()
    return ""  # fallback

def process_one(i: int, prompt: str) -> Tuple[int, str]:
    logger.info(prompt)
    try:
        ans = run_agent(str(prompt))
    except langgraph.errors.GraphRecursionError as e:
        tqdm.write(f"[!] Recursion limit on row #{i+1}: {e}")
        ans = f"ERROR: GraphRecursionError: {e}"
    except Exception as e:
        tqdm.write(f"[!] Exception on row #{i+1}: {e}")
        ans = f"ERROR: {repr(e)}"
    return i, ans

def main() -> None:
    p = argparse.ArgumentParser(description="Run Agentic-RAG (UI-logic) over CSV prompts.")
    p.add_argument("--input", required=True, type=Path, help="CSV with a 'prompt' column")
    p.add_argument("--output", type=Path, required=False,
                   help="Where to save CSV with 'response' column. Omit or pass '-' to print to stdout.")
    p.add_argument("--threads", type=int, default=8, help="Number of worker threads (I/O-bound).")
    args = p.parse_args()

    df = pd.read_csv(args.input)
    if "query" not in df.columns:
        raise ValueError("Input CSV must contain a 'query' column.")

    prompts = df["query"].tolist()
    logger.info("Processing %d prompts with %d threads…", len(prompts), args.threads)

    out: List[str] = [""] * len(prompts)

    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futures = [ex.submit(process_one, i, prompt) for i, prompt in enumerate(prompts)]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"prompts (threads={args.threads})"):
            i, ans = fut.result()
            out[i] = ans

    df["response"] = out
    

    if args.output is None or str(args.output) == "-":
        logger.info("No --output provided (or '-' specified). Writing CSV to stdout.")
        df.to_csv(sys.stdout, index=False)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        logger.info("Saved results to %s", args.output)

if __name__ == "__main__":
    main()