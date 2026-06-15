from langchain_core.messages import SystemMessage


# ============================================================
# HELPER: Safe base64 guard
# ============================================================

def _safe_b64(data: dict) -> str | None:
    """Return b64 string only if non-empty, else None."""
    b64 = data.get("b64", "")
    return b64 if isinstance(b64, str) and b64.strip() else None


# ============================================================
# 1. IMAGE QUERY
# ============================================================

def image_query(query_str: str, images_data: list[dict]) -> list:
    """
    Multimodal image-based RAG query.

    Parameters
    ----------
    query_str   : The user's question or retrieval intent.
    images_data : List of dicts with keys:
                    - 'b64'       (str)  Base64-encoded JPEG image.
                    - 'source'    (str, optional) Filename / URL / label.
                    - 'caption'   (str, optional) Pre-computed caption.
                    - 'metadata'  (dict, optional) Any extra key-value pairs.

    Returns
    -------
    List of LangChain message dicts ready to pass to ChatOpenAI / ChatAnthropic.
    """

    # ------------------------------------------------------------------
    # Build image inventory header
    # ------------------------------------------------------------------
    inventory_lines = ["Visual Asset Inventory:"]
    valid_images = []

    for i, data in enumerate(images_data):
        b64 = _safe_b64(data)
        if not b64:
            continue                          # skip corrupt / empty frames

        source  = data.get("source",  f"Image {i + 1}")
        caption = data.get("caption", "No caption provided.")
        meta    = data.get("metadata", {})
        meta_str = ", ".join(f"{k}={v}" for k, v in meta.items()) if meta else "—"

        inventory_lines.append(
            f"  [{i + 1}] Source: {source} | Caption: {caption} | Metadata: {meta_str}"
        )
        valid_images.append((i + 1, b64))

    if not valid_images:
        return []

    inventory_str = "\n".join(inventory_lines)

    # ------------------------------------------------------------------
    # System prompt — strict, hallucination-resistant
    # ------------------------------------------------------------------
    system_instruction = SystemMessage(content=(
        "You are an expert visual analysis engine with strict epistemic discipline.\n\n"
        "ROLE & SCOPE\n"
        "  • Analyse only the images supplied in this message.\n"
        "  • Base every claim exclusively on visible evidence within those images.\n\n"
        "STRICT RULES — violations undermine trust and are unacceptable:\n"
        "  1. NEVER invent, assume, or infer facts not directly visible.\n"
        "  2. NEVER reference images not provided in this message.\n"
        "  3. If a detail is ambiguous or partially occluded, say so explicitly.\n"
        "  4. Distinguish clearly between 'observed' and 'possible / uncertain'.\n"
        "  5. When multiple images are present, reason across them collectively\n"
        "     before forming a conclusion — do not treat each in isolation.\n\n"
        "OUTPUT FORMAT\n"
        "  • Answer the User Query directly and concisely.\n"
        "  • Cite the image number(s) that support each claim, e.g. [Image 2].\n"
        "  • If the query cannot be answered from the provided images, state:\n"
        "    'Insufficient visual evidence to answer this query.'\n"
        "  • Do not pad the response with generic descriptions of unrelated content."
    ))

    # ------------------------------------------------------------------
    # User message — inventory + images + query
    # ------------------------------------------------------------------
    user_content: list[dict] = [
        {
            "type": "text",
            "text": (
                f"{inventory_str}\n\n"
                f"User Query:\n{query_str}\n\n"
                "Analyse the images below and answer the query according to the rules above."
            )
        }
    ]

    for idx, b64 in valid_images:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })

    return user_content, system_instruction


# ============================================================
# 2. AUDIO / TEXT QUERY
# ============================================================

def Audio_query(query_str: str, texts: list[str]) -> list:
    """
    RAG query over audio transcripts / speech-to-text chunks.

    Parameters
    ----------
    query_str : The user's question or retrieval intent.
    texts     : List of transcript snippets (strings), ordered by relevance
                or by time — either is fine; label them if possible.

    Returns
    -------
    List of LangChain message dicts.
    """

    # ------------------------------------------------------------------
    # Build transcript context
    # ------------------------------------------------------------------
    if texts:
        transcript_block = "\n".join(
            f"  [{i + 1}] {chunk.strip()}"
            for i, chunk in enumerate(texts)
            if chunk and chunk.strip()
        )
        if not transcript_block:
            transcript_block = "  [No usable transcript snippets provided.]"
    else:
        transcript_block = "  [No transcript data available.]"

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------
    system_instruction = SystemMessage(content=(
        "You are a precise audio-transcript analysis engine with strict epistemic discipline.\n\n"
        "ROLE & SCOPE\n"
        "  • You will receive numbered transcript snippets extracted from an audio source.\n"
        "  • Your sole job is to answer the User Query using those snippets.\n\n"
        "STRICT RULES:\n"
        "  1. Ground every claim in the provided transcript snippets.\n"
        "  2. NEVER fabricate quotes, facts, or speaker intent not present in the text.\n"
        "  3. If a snippet is ambiguous (mishearing, [inaudible], crosstalk), flag it.\n"
        "  4. Cite snippet numbers, e.g. [Snippet 3], for every factual claim.\n"
        "  5. If the snippets do not contain enough information to answer the query,\n"
        "     state: 'Insufficient transcript evidence to answer this query.'\n"
        "  6. Do not speculate about content that was not transcribed.\n\n"
        "OUTPUT FORMAT\n"
        "  • Answer the User Query directly.\n"
        "  • Keep the response concise; avoid summarising snippets that are irrelevant.\n"
        "  • If quoting directly, reproduce the snippet exactly as given — no paraphrasing\n"
        "    when precision matters."
    ))

    # ------------------------------------------------------------------
    # User message
    # ------------------------------------------------------------------
    user_content = (
        f"Transcript Snippets (ordered by relevance):\n"
        f"{transcript_block}\n\n"
        f"User Query:\n{query_str}\n\n"
        "Answer the query strictly from the transcript snippets above."
    )

    return  user_content, system_instruction


# ============================================================
# 3. VIDEO QUERY  (refined version of the original)
# ============================================================

def video_query(query_str: str, texts: list[str], images_data: list[dict]) -> list:
    """
    Multimodal RAG query combining video frames + transcripts.

    Parameters
    ----------
    query_str   : The user's question or retrieval intent.
    texts       : Transcript / speech-to-text snippets.
    images_data : List of dicts with keys:
                    - 'b64'       (str)  Base64-encoded JPEG frame.
                    - 'timestamp' (float, optional) Seconds from start.
                    - 'caption'   (str,  optional) Pre-computed frame caption.

    Returns
    -------
    List of LangChain message dicts.
    """

    # ------------------------------------------------------------------
    # 1. Validate & index frames
    # ------------------------------------------------------------------
    valid_frames = []
    for i, data in enumerate(images_data):
        b64 = _safe_b64(data)
        if not b64:
            continue
        valid_frames.append((i + 1, data))

    # ------------------------------------------------------------------
    # 2. Visual timeline context
    # ------------------------------------------------------------------
    visual_lines = ["Visual Evidence (ordered by relevance):"]
    for idx, data in valid_frames:
        ts      = data.get("timestamp", None)
        caption = data.get("caption",   "No caption.")
        ts_str  = f"{ts:.2f}s" if ts is not None else "unknown timestamp"
        visual_lines.append(f"  [Frame {idx}] @ {ts_str} — {caption}")

    visual_context = "\n".join(visual_lines) if len(visual_lines) > 1 \
        else "  [No valid frames provided.]"

    # ------------------------------------------------------------------
    # 3. Audio / transcript context
    # ------------------------------------------------------------------
    if texts:
        audio_context = "\n".join(
            f"  [{i + 1}] {t.strip()}"
            for i, t in enumerate(texts)
            if t and t.strip()
        ) or "  [No usable transcript snippets.]"
    else:
        audio_context = "  [No transcript data available.]"

    # ------------------------------------------------------------------
    # 4. System prompt
    # ------------------------------------------------------------------
    system_instruction = SystemMessage(content=(
        "You are a precise multimodal video analysis engine with strict epistemic discipline.\n\n"
        "INPUTS YOU WILL RECEIVE\n"
        "  A. Visual Evidence  — annotated video frames with timestamps.\n"
        "  B. Audio Context    — transcript snippets extracted from the same video.\n"
        "  C. User Query       — the specific question to answer.\n\n"
        "ANALYSIS PROTOCOL\n"
        "  1. Read the User Query first to focus your analysis.\n"
        "  2. Examine frames in the order listed; note what each frame shows.\n"
        "  3. Cross-reference frames with transcript snippets to identify events.\n"
        "  4. Map identified events to timestamps whenever possible.\n"
        "  5. Synthesise visual + audio evidence before forming your final answer.\n\n"
        "STRICT RULES — violations undermine trust and are unacceptable:\n"
        "  1. Every claim must be traceable to a specific frame and/or snippet.\n"
        "  2. NEVER invent events, timestamps, or speaker intent not in evidence.\n"
        "  3. Distinguish clearly: 'observed in Frame X' vs 'suggested by Snippet Y'.\n"
        "  4. If visual and audio evidence conflict, report both sides honestly.\n"
        "  5. If evidence is insufficient, state:\n"
        "     'Insufficient evidence to answer this query reliably.'\n"
        "  6. Do not repeat frame/snippet content verbatim unless quoting is essential.\n\n"
        "OUTPUT FORMAT\n"
        "  • Answer the User Query directly with evidence-backed statements.\n"
        "  • Cite sources inline: [Frame 2 @ 4.30s], [Snippet 1].\n"
        "  • If a timeline of events is requested, present it in chronological order."
    ))

    # ------------------------------------------------------------------
    # 5. User message payload
    # ------------------------------------------------------------------
    user_content: list[dict] = [
        {
            "type": "text",
            "text": (
                f"{visual_context}\n\n"
                f"Audio Context (transcript snippets):\n{audio_context}\n\n"
                f"User Query:\n{query_str}\n\n"
                "Analyse the frames below together with the audio context and answer the query."
            )
        }
    ]

    for idx, data in valid_frames:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{data['b64']}"}
        })

    return user_content, system_instruction


# ============================================================
# 4. DOCUMENT QUERY
# ============================================================

def document_query(query_str: str, texts: list[str]) -> list:
    """
    RAG query over retrieved document chunks (PDF, DOCX, TXT, etc.).

    Parameters
    ----------
    query_str : The user's question or retrieval intent.
    texts     : List of retrieved text chunks, each representing a passage
                from one or more source documents.
                Optionally, each chunk may begin with a source label, e.g.:
                  "[SOURCE: annual_report.pdf, p.14] Revenue grew by …"

    Returns
    -------
    List of LangChain message dicts.
    """

    # ------------------------------------------------------------------
    # Build passage block
    # ------------------------------------------------------------------
    if texts:
        passages = "\n\n".join(
            f"--- Passage {i + 1} ---\n{chunk.strip()}"
            for i, chunk in enumerate(texts)
            if chunk and chunk.strip()
        )
        if not passages:
            passages = "[No usable document passages provided.]"
    else:
        passages = "[No document passages available.]"

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------
    system_instruction = SystemMessage(content=(
        "You are a precise document analysis engine and a rigorous research assistant.\n\n"
        "ROLE & SCOPE\n"
        "  • You will receive numbered passages retrieved from one or more documents.\n"
        "  • Your sole job is to answer the User Query using those passages.\n\n"
        "STRICT RULES:\n"
        "  1. Ground every factual claim exclusively in the provided passages.\n"
        "  2. NEVER hallucinate citations, page numbers, authors, or statistics\n"
        "     that are not present in the supplied text.\n"
        "  3. If a passage is truncated, contradictory, or ambiguous, say so.\n"
        "  4. If multiple passages address the same point, synthesise them;\n"
        "     note any contradictions rather than silently resolving them.\n"
        "  5. Cite sources by passage number, e.g. [Passage 2], for every claim.\n"
        "  6. If the passages do not contain sufficient information, state:\n"
        "     'Insufficient document evidence to answer this query.'\n"
        "  7. Do not pad the answer with passage content that is irrelevant to\n"
        "     the query.\n\n"
        "OUTPUT FORMAT\n"
        "  • Answer the User Query directly and concisely.\n"
        "  • Structure the response logically (e.g. bullet points, short paragraphs)\n"
        "    when the answer has multiple parts.\n"
        "  • If the user asks for a quote, reproduce the passage text exactly.\n"
        "  • End with a brief 'Sources used: [Passage X, Passage Y]' line."
    ))

    # ------------------------------------------------------------------
    # User message
    # ------------------------------------------------------------------
    user_content = (
        f"Retrieved Document Passages:\n\n"
        f"{passages}\n\n"
        f"---\n\n"
        f"User Query:\n{query_str}\n\n"
        "Answer the query strictly from the passages above, following all rules."
    )

    return user_content, system_instruction