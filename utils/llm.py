from __future__ import annotations
import pathlib
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.tools import Tool
import os
from langchain.chat_models import init_chat_model
from langchain_community.tools import DuckDuckGoSearchResults
from typing import Optional, List, Dict, Tuple, Any
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool
from langchain_core.documents import Document
import json
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import connections, Collection, utility, MilvusException
from langchain_tavily import TavilySearch
from collections import defaultdict
import numpy as np
from math import exp

# ---------- Config ----------
ADDRESS    = "localhost:19530"         
COLLECTION = "reviews_server_index"



class MultiQueryArgs(BaseModel):
    queries: List[str]
    metadata: Optional[Dict[str, str]] = None
    top_k: int = Field(5, ge=1, description="docs")
    gamma: float = Field(1.0, gt=0, description="RBF width")
    agg: str = Field("sum", pattern="^(sum|mean)$")


_MODEL_NAME = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_VECTOR_DIR = "./descriptions_hotel.db"
_REVIEWS_DIR = "./hotel_reviews.db"



@st.cache_resource(show_spinner="Loading embedding model…")
def get_embeddings(model_name: str):
    emb = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cuda"},                 # or "cuda:0"
        encode_kwargs={"normalize_embeddings": True, "batch_size": 1024},
    )
    # One-time CUDA + kernel warmup so the first query is fast
    _ = emb.embed_query("warmup")
    return emb



def _connect_milvus_server(host: str = "127.0.0.1", port: int = 19530, alias: str = "default"):
    """
    Try gRPC (host/port). If that fails, try HTTP URI (http://host:port).
    Raises RuntimeError with a helpful message if both fail.
    """
    try:
        try:
            connections.disconnect(alias)
        except Exception:
            pass

        # 1) gRPC style
        connections.connect(alias=alias, host=host, port=str(port))
        # quick ping
        utility.get_server_version()
        return {"mode": "grpc", "host": host, "port": str(port)}
    except Exception as e1:
        # 2) HTTP style (Milvus 2.3/2.4 sometimes expects this)
        try:
            connections.disconnect(alias)
        except Exception:
            pass
        uri = f"http://{host}:{port}"
        connections.connect(alias=alias, uri=uri)
        utility.get_server_version()
        return {"mode": "http", "uri": uri}
    # If we got here, both failed
    raise RuntimeError(
        f"Could not connect to Milvus at {host}:{port}. "
        "Is the server running and port exposed? (docker ps / curl http://127.0.0.1:19530)"
    )

def load_vectorstore(_VECTOR_DIR=_VECTOR_DIR, _EMBED_MODEL=_EMBED_MODEL):
    embeddings = get_embeddings(_EMBED_MODEL)

    hotel_vs = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": _VECTOR_DIR},           # Milvus Lite file
        collection_name="hotel_index",
        consistency_level="Eventually",
    )

    SERVER_HOST = "127.0.0.1"
    SERVER_PORT = 19530
    conn_info = _connect_milvus_server(SERVER_HOST, SERVER_PORT, alias="default")

    if conn_info["mode"] == "grpc":
        lc_conn = {"host": SERVER_HOST, "port": str(SERVER_PORT)}
    else:
        lc_conn = {"address": conn_info["uri"]}

    col = Collection(COLLECTION)

    index_type = ""
    try:
        if col.indexes:
            params = col.indexes[0].params or {}
            index_type = (params.get("index_type") or "").upper()
    except Exception:
        pass

    if index_type == "HNSW":
        SEARCH_PARAMS = {"metric_type": "IP", "params": {"ef": 128}}
    elif index_type.startswith("IVF"):
        SEARCH_PARAMS = {"metric_type": "IP", "params": {"nprobe": 16}}
    else:
        SEARCH_PARAMS = {"metric_type": "IP"}  # flat / default

    try:
        col.load()
    except MilvusException:
        col.release()
        col.load()
    utility.wait_for_loading_complete(COLLECTION, timeout=300)

    reviews_vs = Milvus(
        embedding_function=embeddings,
        connection_args=lc_conn,  # <-- key change: aligns with how we connected
        collection_name=COLLECTION,
        search_params=SEARCH_PARAMS,
        drop_old=False,
        consistency_level="Eventually",
    )

    print("✅ Loaded collection:", COLLECTION, "| index:", index_type or "N/A")
    return hotel_vs, reviews_vs

@st.cache_resource(show_spinner="Booting local LLM …")
def load_llm(model_name: str = _MODEL_NAME):
    
    return init_chat_model(
    model_name,
    model_provider="bedrock_converse",
    region_name=os.environ["AWS_DEFAULT_REGION"],                 
    max_tokens=2048,
    temperature=0.0,
)


def get_retriever_tool(vs: FAISS) -> Tool:
    """Return a LangChain `Tool` that wraps the vector store retriever."""
    retriever = vs.as_retriever(search_kwargs={"k": 4})

    def _retrieve(query: str):
        docs: List[Document] = retriever.invoke(query)
        return {
            "results": [
                {"content": d.page_content, "metadata": d.metadata} for d in docs
            ]
        }

    return Tool(
        name="retrieve_hotels_descriptions",
        func=_retrieve,
        description="Search hotel descriptions with metadata.",
    )


def unified_rbf_ranking(
    queries: list[str],
    vectorstore,
    embeddings,
    top_k: int = 100,
    gamma: float = 1.0,
    agg: str = "sum",
    metadata_filter: dict | None = None,   # renamed
) -> list[dict]:
    """
    Extended version with optional metadata filtering.
    """

    q_embs = embeddings.embed_documents(queries)
    Q = np.vstack(q_embs).astype("float32")

    dists, idxs = vectorstore.index.search(Q, top_k)
    sims = np.exp(-gamma * dists)

    agg_scores = defaultdict(lambda: {"scores": [], "faiss_ids": []})
    for qi in range(len(queries)):
        for rank in range(top_k):
            faiss_id = int(idxs[qi, rank])
            score = float(sims[qi, rank])
            entry = agg_scores[faiss_id]
            entry["scores"].append(score)
            entry["faiss_ids"].append(faiss_id)

    results = []
    for faiss_id, info in agg_scores.items():
        doc_id = vectorstore.index_to_docstore_id[faiss_id]
        doc = vectorstore.docstore._dict[doc_id]

        # Apply metadata filter (if provided)
        if metadata_filter:
            meta = doc.metadata or {}
            if not all(
                any(str(v) == str(meta_val) for meta_val in meta.values())
                for v in metadata_filter.values()
            ):
                continue



        scores = info["scores"]
        agg_score = sum(scores) if agg == "sum" else sum(scores) / len(queries)
        results.append({
            "doc": doc,
            "score": agg_score,
            "details": {
                "per_query_scores": scores,
                "faiss_ids": info["faiss_ids"],
            }
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def dict_to_expr(meta: dict[str, str]) -> str:
    """
    Converts {"Country": "Spain", "City": "Madrid"}  →
            'Country like "%Spain%" && City like "%Madrid%"'
    """
    return " && ".join(
        f'{k} like "%{v}%"'
        for k, v in meta.items()
    )


def unified_rbf_ranking_milvus(
    queries: list[str],
    vectorstore,               # langchain_milvus.Milvus instance
    embeddings,                # any LangChain embedding function
    top_k: int = 100,
    gamma: float = 1.0,
    agg: str = "sum",          # "sum" or "mean"
    metadata_filter: dict | None = None,
):
    """
    Multi-query RBF fusion on Milvus with *server-side* metadata filtering.
    """
    # 1. Prepare the Milvus filter expression (if any)
    expr = dict_to_expr(metadata_filter) if metadata_filter else None

    # 2. Embed the queries
    Q = np.vstack(embeddings.embed_documents(queries)).astype("float32")

    # 3. Search per query and collect scores
    agg_scores = defaultdict(lambda: {"scores": [], "ids": []})

    for qi, q_vec in enumerate(Q):
        # Milvus returns (Document, distance) pairs
        docs_and_dists = vectorstore.similarity_search_with_score_by_vector(
            q_vec, k=top_k, expr=expr
        )

        for doc, dist in docs_and_dists:
            sim = exp(-gamma * dist)          # convert distance → similarity
            agg_scores[id(doc)]["scores"].append(sim)
            agg_scores[id(doc)]["ids"].append(doc)

    # 4. Aggregate
    results = []
    for _, info in agg_scores.items():
        scores = info["scores"]
        doc    = info["ids"][0]               # all entries point to the same doc
        agg_s  = sum(scores) if agg == "sum" else sum(scores) / len(queries)
        results.append(
            {
                "doc": doc,
                "score": agg_s,
                "details": {
                    "per_query_scores": scores,
                },
            }
        )

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def get_multi_queries_retrival_tool(vectorstore):
    embeddings = HuggingFaceEmbeddings(
        model_name=_EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


    def _unified_rbf_tool(
        queries: List[str],
        metadata: Optional[Dict[str, str]] = None,
        top_k: int = 100,
        gamma: float = 1.0,
        agg: str = "sum",
    ) -> Dict[str, Any]:
        ranked = unified_rbf_ranking_milvus(
            queries=queries,
            metadata_filter=metadata,
            vectorstore=vectorstore,
            embeddings=embeddings,
            top_k=top_k,
            gamma=gamma,
            agg=agg,
        )[:7]

        return {
            "results": [
                {
                    "content": hit["doc"].page_content,
                    "metadata": hit["doc"].metadata,
                    "score": hit["score"],
                }
                for hit in ranked
            ]
        }

    return StructuredTool.from_function(
        name="multi_query_retrieval",
        description=(
            "Search hotel descriptions by combining several feature keywords and optional metadata filters.\n"
            "Supply only *amenity or property* terms in `queries` — avoid city/country names or dates there.\n\n"
            "**Optional filters** (exact matches): `Rating`, `Name`, `Country`, `City`\n\n"
            "**Example Input:**\n"
            "  {\n"
            '    "queries": ["breakfast", "pool"],\n'
            '    "metadata": {\n'
            '      "Rating": "ThreeStar",\n'
            '      "Name": "Relax",\n'
            '      "Country": "Russia",\n'
            '      "City": "Vityazevo"\n'
            "    }\n"
            "  }"
        ),
        args_schema=MultiQueryArgs,
        func=_unified_rbf_tool,
    )



def duckduckgo_search(query: str) -> dict:
    """
    duckduckgo_search(query: str) -> dict

    Performs a DuckDuckGo search for the given query string and
    returns up to 5 results as a JSON‑serializable dict.
    """
    ddg = DuckDuckGoSearchResults(max_results=5)
    return ddg.invoke(query)


def Tavily_search(query: str) -> dict:
    """
    Tavily_search(query: str) -> dict

    Performs a Tavily search for the given query string and
    returns up to 5 results as a JSON‑serializable dict.
    """
    tavily_search_tool = TavilySearch(
        max_results=10,
        topic="general")
    return tavily_search_tool.invoke(query)



def get_search_tool() -> Tool:
    """Return a LangChain `Tool` that wraps the DuckDuckGo search API."""
    return Tool(
        name="web_search",
        func=Tavily_search,
        description=(
            "Perform a web search to fetch information that may not exist in the hotels descriptions or reviews like thier price or external knowledge"
            "Formulate a concise, keyword‑focused query."
            "Hint: To make effective search for hotel data use the format: [HOTEL NAME] [CITY] [KEY WORD]"
            "For example, budget friendly hotel in Eilat, you can search Hilton Eilat Price"
            "For genral knowledge searches, try to make it short but informative"
            "Input: single plain‑text search string; "
        )
        ,
        return_direct=True,
    )


def get_reviews_tool(vs) -> StructuredTool:
    """Return a LangChain StructuredTool that searches hotel REVIEWS with optional filters.
       Supports multiple hotel names and RRF fusion across per-name ranked lists.
    """

    # --------------------------- RRF helper ---------------------------
    def _rrf_fuse(rank_lists: List[List[Document]], rrf_k: int = 60) -> List[Document]:
        """
        Reciprocal Rank Fusion: score(doc) = sum_i 1 / (rrf_k + rank_i(doc))
        rank_lists: list of ranked lists (best first) of Document objects.
        Returns a single list of Documents sorted by fused score (desc), deduped.
        """
        scores: Dict[str, float] = {}
        keeper: Dict[str, Document] = {}

        def _doc_key(d: Document) -> str:
            # Prefer Milvus primary key if present; otherwise build a stable-ish key
            return (
                str(d.metadata.get("pk"))
                if "pk" in d.metadata
                else f"{d.metadata.get('Name','')}|{d.metadata.get('City','')}|{d.page_content[:80]}"
            )

        for lst in rank_lists:
            for rank, doc in enumerate(lst, start=1):
                key = _doc_key(doc)
                keeper.setdefault(key, doc)
                scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)

        fused_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        return [keeper[k] for k in fused_keys]

    # ---------------------- Milvus search helper ----------------------
    def _search_list(query: str, expr: Optional[str], k: int) -> List[Document]:
        """
        Run a single retrieval and return a ranked list of Documents (order = rank).
        Works with Milvus .similarity_search_with_score or .similarity_search fallback.
        """
        try:
            docs_scores: List[Tuple[Document, float]] = vs.similarity_search_with_score(
                query, k=k, expr=expr
            )
            ranked_docs = [d for d, _ in docs_scores]   # order already by relevance
        except AttributeError:
            docs: List[Document] = vs.similarity_search(query, k=k, expr=expr)
            ranked_docs = docs
        return ranked_docs

    # ------------------------------ Args ------------------------------
    class Args(BaseModel):
        query: str = Field(..., description="Natural language question to search within reviews.")
        # NEW: multi-name support. 'name' kept for backward compatibility.
        names: Optional[List[str]] = Field(
            None, description='List of hotel names to fuse via RRF (metadata field "Name").'
        )
        name: Optional[str] = Field(
            None, description='Single hotel name filter (metadata "Name").'
        )
        city: Optional[str] = Field(None, description='City filter (metadata "City").')
        county: Optional[str] = Field(None, description='County filter (metadata "County").')
        k: int = Field(8, ge=1, le=50, description="How many fused results to return.")
        # Optional knobs (safe defaults). Per-list depth can exceed final k to give RRF room.
        per_list_k: int = Field(
            25, ge=1, le=200,
            description="How many results to retrieve per sub-search before RRF."
        )
        rrf_k: int = Field(
            60, ge=1, le=1000,
            description="RRF constant; higher reduces the impact of rank position."
        )

    # ----------------------------- Runner -----------------------------
    def _run(query: str,
             names: Optional[List[str]] = None,
             name: Optional[str] = None,
             city: Optional[str] = None,
             county: Optional[str] = None,
             k: int = 8,
             per_list_k: int = 25,
             rrf_k: int = 60):
        # Base filters used for all sub-queries
        base_filters: Dict[str, str] = {}
        if city:
            base_filters["City"] = city
        if county:
            base_filters["County"] = county

        # Build the list of per-name ranked lists
        rank_lists: List[List[Document]] = []

        # If multiple names provided, search per name; else if single name; else single unfiltered run
        if names and len(names) > 0:
            for nm in names:
                sub_filters = dict(base_filters)
                sub_filters["Name"] = nm
                expr = dict_to_expr(sub_filters)
                rank_lists.append(_search_list(query, expr, per_list_k))
        elif name:
            sub_filters = dict(base_filters)
            sub_filters["Name"] = name
            expr = dict_to_expr(sub_filters)
            rank_lists.append(_search_list(query, expr, per_list_k))
        else:
            expr = dict_to_expr(base_filters) if base_filters else None
            rank_lists.append(_search_list(query, expr, per_list_k))

        # Fuse with RRF (handles the single-list case too)
        fused_docs = _rrf_fuse(rank_lists, rrf_k=rrf_k)[:k]

        if not fused_docs:
            return "No results."

        # Format like your previous tool (hide 'pk' if present)
        def _format_block(d: Document) -> str:
            metadata = {k: v for k, v in (d.metadata or {}).items() if k != "pk"}
            meta_str = json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True)
            return f"{d.page_content}\n{meta_str}"

        return "\n***\n".join(_format_block(d) for d in fused_docs)

    return StructuredTool(
        name="search_hotel_reviews",
        func=_run,
        args_schema=Args,
        description = (
            "Search **hotel REVIEWS** (subjective or additional details like breakfast, noise, cleanliness, staff, parking, nearby attractions). "
            "Use optional filters by Name(s), City, or County. When multiple names are supplied, results are merged. "
            "If you want to find a hotel with good breakfast, use only a very short query like 'good breakfast'. "
            "Example 1: {\n"
            "    'query': 'good breakfast',"
            "    'city': 'New York'"
            "}"
             "Example 2: {\n"
            "    'query': 'good breakfast',"
            "    'county': 'France'"
            "}"
        ),
    )





