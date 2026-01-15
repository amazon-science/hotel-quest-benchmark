from __future__ import annotations

"""Agentic RAG demo

A side‑by‑side chat UI showing the agent’s internal trace.
"""
from pymilvus import connections, utility
import json
import logging
import uuid
from pprint import PrettyPrinter
from typing import Any, List, Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from typing import Literal
from openinference.instrumentation.langchain import LangChainInstrumentor
import streamlit as st
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from utils.llm import get_search_tool, load_llm, load_vectorstore, get_multi_queries_retrival_tool, get_reviews_tool
from utils.prompt import build_final_prompt, build_summary_prompt, build_plan_prompt
import os
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from phoenix.otel import register
import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


MAX_TOOL_LOOPS = 20


@st.cache_resource(show_spinner=False)
def init_tracing():
    from phoenix.otel import register
    from openinference.instrumentation.langchain import LangChainInstrumentor

    # One-time tracer registration
    tp = register(
        project_name="app",                
        endpoint="http://127.0.0.1:6006/v1/traces",     
        batch=True,                                     
        auto_instrument=False,                       
    )

    try:
        LangChainInstrumentor().uninstrument()
    except Exception:
        pass
    LangChainInstrumentor().instrument(tracer_provider=tp)

    return tp

tracer_provider = init_tracing()
print("OpenTelemetry + LangChain instrumented for Arize Phoenix on EC2.")


logging.getLogger("streamlit.runtime.state.session_state_proxy").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)


st.set_page_config(page_title="Agentic RAG", page_icon="🎯", layout="wide")

with st.sidebar:
    st.markdown("## Session")
    if st.button("🆕 New chat", use_container_width=True):
        st.session_state.history = []
        st.session_state.thread_id = uuid.uuid4().hex
        st.rerun()


@st.cache_resource(show_spinner=False)
def _connect_milvus_once():
    """Idempotent connect to Milvus with env-driven config."""
    if connections.has_connection("default"):
        return "already-connected"

    uri  = os.getenv("MILVUS_URI")  # e.g., "http://localhost:19530" or "http://host.docker.internal:19530"
    host = os.getenv("MILVUS_HOST", "127.0.0.1")
    port = int(os.getenv("MILVUS_PORT", "19530"))
    tok  = os.getenv("MILVUS_TOKEN")  # e.g., "root:Milvus" if auth enabled

    if uri:
        connections.connect(alias="default", uri=uri, token=tok, timeout=30, secure=uri.startswith("https://"))
    else:
        connections.connect(alias="default", host=host, port=port, token=tok, timeout=30)

    _ = utility.get_server_version()
    return "connected"

@st.cache_resource(show_spinner=False)
def init_resources():
    _ = _connect_milvus_once()

    vs, vs_reviews = load_vectorstore()
    retriever_tool = get_multi_queries_retrival_tool(vs)
    search_tool = get_search_tool();   search_tool.name = "web_search"
    reviews_tool = get_reviews_tool(vs_reviews); reviews_tool.name = "reviews"

    llm = load_llm()
    llm_with_tools = llm.bind_tools([retriever_tool, search_tool, reviews_tool])
    return retriever_tool, llm_with_tools, llm, search_tool, reviews_tool


retriever_tool, llm_with_tools, llm, search_tool, reviews_tool = init_resources()

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _extract_context(msgs: List[BaseMessage]) -> str:
    """Merge any tool outputs found in *msgs* into one context string."""
    ctx_chunks: List[str] = []
    for m in msgs:
        if isinstance(m, ToolMessage):
            if isinstance(m.content, str):
                ctx_chunks.append(m.content)
            else:
                ctx_chunks.extend(getattr(d, "page_content", str(d)) for d in m.content)  # type: ignore[attr-defined]
    return "\n\n".join(ctx_chunks)


def _extract_tool_calls(msgs: list[BaseMessage]) -> str:
    import json
    call_chunks = []
    for m in msgs:
        # handle newer tool_calls
        for call in getattr(m, "tool_calls", []) or []:
            name = getattr(call, "name", None) or call.get("name", "")
            args = getattr(call, "args", None) or call.get("args", {})
            call_chunks.append(f'tool: "{name}" args: {json.dumps(args, default=str)}')
        # fallback: content as dict containing tool metadata
        if isinstance(getattr(m, "content", None), dict):
            d = m.content
            if "tool" in d and "args" in d:
                call_chunks.append(f'tool: "{d["tool"]}" args: {json.dumps(d["args"], default=str)}')
    return "\n".join(call_chunks)


def _pretty(obj: Any) -> str:
    """Human‑friendly representation used while streaming the trace."""
    if isinstance(obj, BaseMessage):
        return str(obj.content)
    if isinstance(obj, list):
        return "\n\n".join(_pretty(i) for i in obj)
    return str(getattr(obj, "content", getattr(obj, "text", obj)))


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
            "retrieve"    if name == "multi_query_retrieval" else
            "web_search"  if name == "web_search" else
            "reviews"     if name == "reviews" else
            "final_answer"
        )
    else:
        decision = "final_answer"

    # cap the number of tool loops
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


def _is_tool_call(msg: BaseMessage) -> bool:
    """True iff the message is an AI tool‑invocation stub (should be hidden)."""
    return (
        isinstance(msg, AIMessage)
        and (
            getattr(msg, "tool_calls", None)           # normal LC models
            or msg.additional_kwargs.get("tool_use")   # Anthropic style
        )
    )

def generate_answer(state: PlanningState):
    human_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    question = human_msgs[-1].content if human_msgs else "Unknown question"
    # context = _extract_context(state["messages"])
    if "summary" not in state.keys():
        state["summary"] = "Empty"
    context = state["summary"]
    prompt = build_final_prompt(question, context)
    final = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [final]}

def tools_condition(state: PlanningState) -> Literal["retrieve","web_search", "reviews",END]:
    """Return the *name* of the next node, not a dict."""
    plan = state.get("next_step", "")
    if plan == "retrieve":
        return "retrieve"
    if plan == "web_search":
        return "web_search"
    if plan == "reviews":
        return "reviews"
    return END          # go straight to the answer

class PlanningState(TypedDict):
    messages: Annotated[list, add_messages]
    summary: str
    next_step: Literal["retrieve", "web_search", "final_answer", "reviews"]
    tool_loops: int  # NEW


@st.cache_resource(show_spinner=False)
def _compile_graph():
    # workflow = StateGraph(MessagesState)
        # RIGHT—use your PlanningState
    workflow = StateGraph(state_schema=PlanningState)


    workflow.add_node("plan_or_answer", generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node("reviews", ToolNode([reviews_tool]))
    workflow.add_node("web_search", ToolNode([search_tool]))
    workflow.add_node("final_answer", generate_answer)

    workflow.add_edge(START, "plan_or_answer")
    workflow.add_conditional_edges(
        "plan_or_answer",
        tools_condition,
        {
            "retrieve":    "retrieve",
            "web_search":  "web_search",
            "reviews":  "reviews",
            END:           "final_answer"
        }
    )

    # workflow.add_edge("retrieve", "final_answer")
    workflow.add_edge("retrieve", "plan_or_answer")
    workflow.add_edge("reviews", "plan_or_answer")
    workflow.add_edge("web_search", "plan_or_answer")

    workflow.add_edge("final_answer", END)
    return workflow.compile()

graph = _compile_graph()

if "history" not in st.session_state:
    st.session_state.history: List[dict[str, str]] = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = uuid.uuid4().hex

col_chat, col_trace = st.columns([0.65, 0.35], gap="large")
pp = PrettyPrinter(indent=2, sort_dicts=True)

col_chat.markdown("## 💬 Chat")
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_prompt = col_chat.chat_input("Your message…")

col_trace.markdown("## 🔎 Internal trace")
trace_container = col_trace.container()

# ---------------------------------------------------------------------------
# Main streaming loop
# ---------------------------------------------------------------------------
def stream_and_render(prompt: str) -> None:
    st.session_state.history.append({"role": "user", "content": prompt})
    with col_chat:
        with st.chat_message("user"):
            st.markdown(prompt)

    assembled_answer = ""
    last_node = None
    answer_placeholder = col_chat.chat_message("assistant").empty()

    history_pairs = [(m["role"], m["content"]) for m in st.session_state.history]
    state: PlanningState = {
        "messages": history_pairs,
        "summary": "Empty",
        "next_step": "final_answer",
        "tool_loops": 0,
    }
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    with st.spinner("Agent is thinking…"):
        for update in graph.stream(state, config=config, stream_mode="updates"):
            for node_name, delta in update.items():
                if node_name == last_node:
                    continue
                last_node = node_name

                new_msgs = delta.get("messages", [])

                # ───────── (inside stream_and_render) ─────────
                for m in new_msgs:
                    if isinstance(m, ToolMessage):
                        continue

                    if _is_tool_call(m):
                        # keep it for the trace panel only
                        continue
                        
                    if node_name != "final_answer":
                        continue

                    chunk = _pretty(m).strip()
                    if not chunk:
                        continue

                    # show only clean assistant text
                    assembled_answer += chunk + "\n"
                    answer_placeholder.markdown(assembled_answer + "▌")




                # Render internal trace without list wrappers
                with trace_container.expander(f"🧩 {node_name}", expanded=False):
                    raw = delta.get("messages")
                    if isinstance(raw, list) and len(raw) == 1:
                        data_to_display = raw[0]
                    else:
                        data_to_display = raw if raw is not None else delta
                    try:
                        st.json(data_to_display, expanded=False)
                    except Exception:
                        serialized = (
                            json.dumps(
                                data_to_display,
                                default=str,
                                indent=2,
                                sort_keys=True,
                                ensure_ascii=False,
                            )
                            if isinstance(data_to_display, (dict, list))
                            else pp.pformat(data_to_display)
                        )
                        st.code(serialized, language="python")

    # Finalize and store assistant message
    answer_placeholder.markdown(assembled_answer.strip())
    st.session_state.history.append(
        {"role": "assistant", "content": assembled_answer.strip()}
    )

if user_prompt:
    stream_and_render(user_prompt)
