from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

from finsightai.config import GROQ_API_KEY, LLM_MODEL
from finsightai.prompts import prompt


def get_llm():

    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is missing in .env")

    llm = ChatGroq(
        model=LLM_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.1
    )

    return llm


def create_rag_chain(llm):

    chain = prompt | llm | StrOutputParser()

    return chain


def get_answer(question, retriever, chain):

    documents = retriever.invoke(question)

    context = ""

    for doc in documents:

        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")

        if isinstance(page, int):
            page += 1

        context += f"""
Source: {source}
Page: {page}

Content:
{doc.page_content}

"""

    answer = chain.invoke({
        "context": context,
        "question": question
    })

    return answer, documents