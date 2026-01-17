GENERATE_PROMPT = (
    "You are an assistant for a hotel recommendation task.\n"
    "Your goals:\n"
    "1) Answer the user's request directly and traceably.\n"
    "2) Break the request into parts and show coverage per part.\n"
    "3) Propose multiple relevant hotel options (when possible), explain how each fits, and compare trade-offs.\n"
    "4) Validate description facts against reviews; note confirmations and contradictions.\n"
    "5) Be honest about gaps/assumptions; never invent facts.\n\n"

    "Citations:\n"
    "- Use [Hotel description: HOTEL_NAME] for description data.\n"
    "- Use [Hotel review: HOTEL_NAME] for review data.\n"
    "- Use [Web: URL] for web sources.\n\n"

    "Instructions:\n"
    "- Generate 2–4 candidate hotels IF available context supports them. Only include options that meet hard constraints; if insufficient, include near-miss options clearly labeled as 'near-miss' and explain the shortfall.\n"
    "- For EACH candidate hotel:\n"
    "   • One-sentence summary of why it fits the user.\n"
    "   • Key fit points tied to the user's requirements with citations.\n"
    "   • Review validation: confirm or contradict description claims using reviews; summarize sentiment or specific quotes (paraphrase) with citations.\n"
    "   • Trade-offs (what you give up vs. alternatives) and who this option is best for.\n"
    "- Make a final ranked recommendation with rationale, referencing the user’s priorities.\n"

    "User question:\n{question}\n\n"
    "Retrieved context (descriptions, reviews, web):\n{context}\n\n"
    "## End of the context ##"
    "Output format:\n"
    "1) Candidate options (2–4): per-option fit, review validation, trade-offs, best-for (each with citations)\n"
    "2) Final recommendation & ranking (with rationale)\n"
)


def build_final_prompt(question: str, context: str) -> str:
    """Fill the template – kept in a util so tests can cover it."""
    return GENERATE_PROMPT.format(question=question, context=context)


SUMMARY_UPDATE_TEMPLATE = (
    "User query:\n{query}\n\n"
    "Current conversation summary:\n{current_summary}\n\n"
    "New messages to incorporate:\n{messages}\n\n"
    "Update the summary so it:\n"
    "1) Integrates all relevant new facts from the latest messages.\n"
    "2) Preserves important prior context unless it is contradicted or replaced.\n"
    "3) Maintains chronological and logical consistency.\n"
    "4) Includes clear citations for every fact:\n"
    "   - [Hotel description: HOTEL_NAME] for retrieved hotel description data.\n"
    "   - [Hotel review: HOTEL_NAME] for review data.\n"
    "   - [Web: URL] for web-sourced data.\n"
    "5) Excludes any speculative or unsupported statements.\n\n"
    "Write only the updated summary — no extra commentary, no questions."
)

NOTES_UPDATE_TEMPLATE = (
    "User query:\n{query}\n\n"
    "Current notes:\n{current_summary}\n\n"
    "New messages to incorporate:\n{messages}\n\n"
    "Update the notes so they:\n"
    "1) Capture all relevant new facts, decisions, and context from the latest messages.\n"
    "2) Preserve important prior details unless contradicted or replaced.\n"
    "3) Organize content into clear bullet points under these sections:\n"
    "   - **Relevant Information**: Facts, retrieved data, and key context (with citations).\n"
    "   - **What Didn't Work**: Approaches tried that failed or produced incomplete/incorrect results.\n"
    "   - **Next Steps**: Concrete actions or investigations to continue progress.\n"
    "4) Maintain chronological and logical consistency.\n"
    "5) Include clear citations for every fact:\n"
    "   - [Hotel description: HOTEL_NAME] for retrieved hotel description data.\n"
    "   - [Hotel review: HOTEL_NAME] for review data.\n"
    "   - [Web: URL] for web-sourced data.\n"
    "6) Exclude speculation unless clearly marked as a hypothesis.\n\n"
    "Write only the updated bullet-point notes — no narrative paragraphs or unrelated commentary."
)


def build_summary_prompt(current_summary: str, recent_messages: list, query: str) -> str:
    """
    Construct a prompt to update the conversation summary.

    Args:
        current_summary: the running summary string
        recent_messages: list of the most recent message objects, each having a .content attribute
        query: the user's current query

    Returns:
        A formatted prompt string for the summarization LLM.
    """
    if not recent_messages:
        raise ValueError("recent_messages must contain at least one message")

    # Build numbered list of recent messages
    messages_text = "\n".join(f"{i+1}. {m.content}" for i, m in enumerate(recent_messages[-2:]))

    return NOTES_UPDATE_TEMPLATE.format(
        query=query,
        current_summary=current_summary,
        messages=messages_text,
    )



PLAN_PROMPT_TEMPLATE = (
    "You are an assistant for a hotel recommendation task.\n"
    "Using ONLY the following summary, the original question, and available tools, decide if you have enough information or if you need to retrieve or perform a web search.\n\n"

    "Available tools:\n"
    " 1) Vector search over hotel descriptions (for basic facts, location, amenities).\n"
    " 2) Vector search over hotel reviews (for detailed user experiences and validation).\n"
    " 3) Web search (for information missing from both internal databases).\n\n"

    "Retrieval strategy:\n"
    " * Always start with REVIEWS to create an initial set of candidate hotels.\n"
    " * If reviews do not provide enough candidates, use DESCRIPTIONS to add more candidates. For any new candidates found this way, go back and try to retrieve relevant REVIEWS for them. If reviews already provide enough candidates, use DESCRIPTIONS only to enrich details or cross-check information.\n"
    " * If the internal sources (reviews + descriptions) are not enough to fully satisfy all the reasonable aspects of the question, or if you need external knowledge, then perform a WEB SEARCH.\n\n"

    "Validation rules:\n"
    " - Always cross-check facts between descriptions and reviews.\n"
    " - When there is a conflict, prefer review evidence.\n"
    " - Explicitly note confirmations and contradictions.\n\n"

    "Ambiguity handling (hard rule):\n"
    " - Even if the query is ambiguous, you MUST infer the most reasonable interpretation and provide the best possible answer.\n"
    " - Do NOT ask follow-up clarification questions.\n"
    " - Always return an answer.\n\n"

    "CITATION POLICY (hard requirement):\n"
    " - The final answer MUST include at least one citation from the internal databases:\n"
    "   [Hotel description: HOTEL_NAME] or [Hotel review: HOTEL_NAME].\n"
    " - Do NOT produce a web-only answer. If no internal database hotel can be cited after reasonable attempts, explicitly state this limitation and suggest next retrieval steps.\n"
    " - Prefer database citations over web citations when both exist.\n\n"

    "Decision gate before answering:\n"
    "Before generating the final response, verify whether you have sufficient and relevant information "
    "to meet all user requirements (e.g., location, dates, budget, amenities, party size).\n\n"
    "– If yes: Proceed to compose a well-reasoned, evidence-based answer that includes concrete hotel "
    "recommendations and clearly explains the reasoning.\n"
    "– If not: Continue retrieval or web search iteratively until you either obtain enough information "
    "to fully satisfy *all* the user’s request perfectly."
    
    "Summary:\n{summary}\n"
    "Previous tool attempts:\n{attempts}\n\n"
    "Question:\n{question}"
)

def build_plan_prompt(summary: str, question: str, attempts: str) -> str:
    """
    Construct a prompt to decide the next step (retrieve, web_search, or final_answer).

    Args:
        summary: the updated conversation summary
        question: the user's original question

    Returns:
        A formatted prompt string for the planning LLM.
    """
    return PLAN_PROMPT_TEMPLATE.format(
        summary=summary,
        question=question,
        attempts=attempts,
        k=5
    )


def build_simple_relevance_prompt(question: str, answer: str) -> str:
    return f"""You are a relevance judge. Given a user Question and a Candidate Answer, assign:
    - 2 if the answer fully satisfies the requirements,
    - 1 if it partially does,
    - 0 if it is irrelevant.

    Output ONLY a JSON object with exactly these two fields:
    {{"score": <0|1|2>, "explanation": "<one-sentence justification>"}}

    Question:
    {question.strip()}

    Answer:
    {answer.strip()}
    """


def build_relevance_prompt(question: str, answer: str) -> str:
    return f"""You are a relevance judge. Your task is to evaluate how well the Candidate Answer responds to the User Question.
    Follow these steps:
    1. **Understand the Question**: Identify the user's intent, the type of information requested, and any explicit or implicit requirements (e.g., specific features, constraints, or context).
    2. **Understand the Answer**: Extract the key attributes, claims, or content provided in the answer.
    3. **Compare and Evaluate**: Assess how well the answer satisfies the question, considering:
    - **Exact Match**: Fully satisfies all explicit and implicit requirements.
    - **Substitute**: Does not exactly match, but could still serve the same purpose.
    - **Complement**: Related or tangentially relevant, but not addressing the question directly.
    - **Irrelevant**: Fails to address the question or has a critical mismatch.

    Assign a relevance score based on the match type:
    - **2** → Fully satisfies (Exact Match).
    - **1** → Partially satisfies (Substitute or Complement).
    - **0** → Irrelevant or incorrect.

    **Important**: Consider *every part* of the user's query. Think carefully about whether all qualifiers and requirements are addressed by the answer.

    Respond with a JSON object using the following format *only*:
    {{
    "score": <0|1|2>,
    "explanation": "<your justification>"
    }}

    Now evaluate:

    Question:
    {question.strip()}

    Answer:
    {answer.strip()}
    """


PLAN_PROMPT_TEMPLATE_v2 = (
    "You are a hotel recommendation planner.\n\n"
    "Candidates:\n{summary}\n\n"
    "QUESTION:\n{question}\n\n"
    "ATTEMPTS:\n{attempts}\n\n"

    "DECISION GUIDE:\n"
    "• ANSWER if: there are at least two distinct candidates "
    "with both review and description coverage, and together they meet all major requirements "
    "with no critical gaps.\n"
    "• SEARCH_REVIEWS if: there are no previous attempts, or there are fewer than two candidates with high score, "
    "or reviews are missing.\n"
    "• SEARCH_DESCRIPTIONS if: candidates exist but lack key details, or attempts to retrieve reviews did not return any candidates.\n"
    "• WEB_SEARCH only if: internal sources remain insufficient after {max_loops} loops, or essential external knowledge is required.\n\n"


    "RULES:\n"
    "• Always start with searching for reviews.\n"
    "• Prioritize reviews for validation and descriptions for factual detail; if conflicting, trust reviews.\n"
    "• Minimize tool calls—stop once criteria for answering are met.\n"
    "• Do not ask clarifications; infer reasonable intent.\n"
    "• Cross-check facts and call out confirmations or contradictions in the answer.\n\n"

    "OUTPUT:\n"
    "Return exactly one of: answer_now, search_reviews, search_descriptions, web_search. "
    "If not answer_now, include a short rationale."
)

def build_plan_prompt_v2(
    summary: str,
    question: str,
    attempts: str,
    min_conf: float = 0.9,
    max_loops: int = 5,
) -> str:
    """
    Build an intelligent planning prompt with candidate scores and explicit decision rules.

    Args:
        summary: Current memory summary (reranked, filtered).
        question: User's original query.
        attempts: Compact string summary of previous tool calls.
        min_conf: Confidence threshold to consider "enough evidence".
        max_loops: Max retrieval loops before switching to web search.
    """
    return PLAN_PROMPT_TEMPLATE_v2.format(
        summary=summary or "(no summary yet)",
        question=question,
        attempts=attempts or "(no previous tool attempts)",
        min_conf=min_conf,
        max_loops=max_loops,
    )
