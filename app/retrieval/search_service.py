
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from app.retrieval import type_ask
load_dotenv()


# A small, fast model for the cheap auxiliary calls (query rewrite). Keeping it
# separate from the main answer model lets us tune cost/latency independently.
_REWRITE_MODEL = "llama-3.1-8b-instant"
_ANSWER_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


def ask(query_str, texts=None, images_data=None, type="document"):
    if type == "image":
        user_content, system_instruction = type_ask.image_query(query_str, images_data)
    elif type == "audio":
        user_content, system_instruction = type_ask.Audio_query(query_str, texts)
    elif type == "video":
        user_content, system_instruction = type_ask.video_query(query_str, texts, images_data)
    else:
        user_content, system_instruction = type_ask.document_query(query_str, texts)

    llm = ChatGroq(model=_ANSWER_MODEL, max_tokens=1500)
    response = llm.invoke([
        system_instruction,
        HumanMessage(content=user_content),
    ])
    return response.content


def rewrite_query(query_str: str) -> str:
    """Self-RAG lite: rewrite a query for better retrieval recall.

    Only called when the first retrieval scores poorly, so the extra latency is
    paid only on hard queries. Expands abbreviations, adds likely synonyms, and
    strips conversational filler so the embedding search has more to match on.
    Falls back to the original query on any failure.
    """
    try:
        llm = ChatGroq(model=_REWRITE_MODEL, max_tokens=120, temperature=0)
        msg = llm.invoke([
            SystemMessage(content=(
                "You rewrite search queries to maximise semantic retrieval recall. "
                "Given a user question, output ONE improved search query that keeps "
                "the original intent but adds key synonyms and removes filler. "
                "Output ONLY the rewritten query, no preamble, no quotes."
            )),
            HumanMessage(content=query_str),
        ])
        rewritten = (msg.content or "").strip().strip('"')
        return rewritten or query_str
    except Exception as e:
        print(f"⚠️ Query rewrite failed, using original: {e}")
        return query_str
