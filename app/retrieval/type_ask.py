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
        "You are a precise audio-transcript analysis assistant.\n\n"
        "ROLE & SCOPE\n"
        "  • You receive numbered transcript snippets extracted from an audio source.\n"
        "  • Answer the User Query using those snippets as your evidence.\n\n"
        "RULES:\n"
        "  1. Ground your answer in the provided snippets; do not invent quotes or facts.\n"
        "  2. Snippets are retrieved by relevance and may be slightly out of order or\n"
        "     incomplete — read ALL of them and synthesise before concluding.\n"
        "  3. Partial evidence is still useful: give the best answer the snippets\n"
        "     support, and note briefly if some detail is uncertain.\n"
        "  4. Only reply 'Insufficient transcript evidence to answer this query.' when\n"
        "     the snippets genuinely contain nothing relevant to the question — not\n"
        "     merely because the wording differs from the query.\n"
        "  5. Cite snippet numbers, e.g. [Snippet 3], for specific claims.\n\n"
        "OUTPUT FORMAT\n"
        "  • Answer directly and concisely.\n"
        "  • Quote exactly when precision matters."
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
        "You are a precise multimodal video analysis assistant.\n\n"
        "INPUTS YOU WILL RECEIVE\n"
        "  A. Visual Evidence  — annotated video frames with timestamps.\n"
        "  B. Audio Context    — transcript snippets from the same video.\n"
        "  C. User Query       — the question to answer.\n\n"
        "ANALYSIS PROTOCOL\n"
        "  1. Read the User Query first to focus your analysis.\n"
        "  2. Read ALL transcript snippets and examine the frames; the snippets\n"
        "     carry most of the spoken content and may be slightly out of order.\n"
        "  3. Cross-reference visual and audio evidence, then synthesise an answer.\n"
        "  4. Map events to timestamps when the evidence supports it.\n\n"
        "RULES:\n"
        "  1. Ground claims in the frames and/or snippets; do not invent events,\n"
        "     timestamps, or speaker intent.\n"
        "  2. Partial evidence is still useful — give the best supported answer and\n"
        "     note briefly what is uncertain.\n"
        "  3. Only reply 'Insufficient evidence to answer this query reliably.' when\n"
        "     neither the frames nor the snippets contain anything relevant — not\n"
        "     merely because the wording differs from the query.\n"
        "  4. Cite sources inline, e.g. [Frame 2 @ 4.30s], [Snippet 1].\n\n"
        "OUTPUT FORMAT\n"
        "  • Answer directly with evidence-backed statements.\n"
        "  • Present timelines in chronological order when asked."
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
        "You are a precise document analysis assistant and rigorous research aide.\n\n"
        "ROLE & SCOPE\n"
        "  • You receive numbered passages retrieved from one or more documents.\n"
        "  • Answer the User Query using those passages as your evidence.\n\n"
        "RULES:\n"
        "  1. Ground every factual claim in the provided passages; do not invent\n"
        "     citations, page numbers, authors, or statistics.\n"
        "  2. Passages are retrieved by relevance and may be partial or slightly out\n"
        "     of order — read ALL of them and synthesise before concluding.\n"
        "  3. Partial evidence is still useful: give the best answer the passages\n"
        "     support, noting briefly anything that is uncertain or contradictory.\n"
        "  4. Only reply 'Insufficient document evidence to answer this query.' when\n"
        "     the passages genuinely contain nothing relevant — not merely because\n"
        "     the wording differs from the query.\n"
        "  5. Cite sources by passage number, e.g. [Passage 2], for specific claims.\n\n"
        "OUTPUT FORMAT\n"
        "  • Answer directly and concisely; structure multi-part answers clearly.\n"
        "  • Reproduce passage text exactly when asked to quote.\n"
        "  • End with a brief 'Sources used: [Passage X, …]' line when you cited any."
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