
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from app.retrieval import type_ask
load_dotenv()
def ask(query_str,texts=None, images_data=None,type="document"):

    if type=="image":
        user_content, system_instruction = type_ask.image_query(query_str, images_data)
    elif type=="audio":
        user_content, system_instruction = type_ask.Audio_query(query_str, texts)
    elif type=="video":
        user_content, system_instruction = type_ask.video_query(query_str, texts, images_data)
    else:
        user_content, system_instruction = type_ask.document_query(query_str, texts)
    # -----------------------------
    # 6. LLM (Groq multimodal call)
    # -----------------------------
    llm = ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        max_tokens=1500
    )

    response = llm.invoke([
        system_instruction,
        HumanMessage(content=user_content)
    ])

    return response.content