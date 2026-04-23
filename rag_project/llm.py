import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv(override=True)

def generate_answer(query: str, context: str):
    # Make sure we use Gemini for fast text generation
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    
    prompt = f"""You are a helpful customer support assistant.
You have the following context from the knowledge base:
{context}

User Query: {query}

Determine if you can accurately answer the query based strictly on the context provided.
Follow these rules strictly:
1. If the user explicitly asks to speak to a human or says yes to a previous offer to connect to a human, you must respond EXACTLY with the word "ESCALATE".
2. If the context is missing, insufficient, or the query is outside the knowledge base, do NOT escalate immediately. Instead, respond with a polite message saying that you don't know anything about that, and ask if they would like to be connected to a human agent.
3. Otherwise, provide a helpful and professional answer based ONLY on the context.
"""
    
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            return "SYSTEM_ERROR_RATE_LIMIT"
        raise e
