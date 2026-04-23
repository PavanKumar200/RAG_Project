import os
from dotenv import load_dotenv
from graph import build_graph
from retriever import get_retriever
from llm import generate_answer
from hitl import human_intervention

load_dotenv()

def main():
    if not os.path.exists("chroma_db"):
        print("Vector database not found. Please run ingest.py first.")
        return

    retriever = get_retriever()
    graph = build_graph()

    # If the user sets the GEMINI_API_KEY environment variable
    if not os.environ.get("GOOGLE_API_KEY"):
        if os.environ.get("GEMINI_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = os.environ.get("GEMINI_API_KEY")

    print("\nCustomer Support Bot Initialized. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            query = input("\nEnter your query: ")
            if query.lower() in ['exit', 'quit']:
                break
            if not query.strip():
                continue
        except EOFError:
            break

        state = {
            "query": query,
            "retriever": retriever,
            "llm": generate_answer,
            "context": "",
            "answer": ""
        }

        print("Running graph...")
        app = graph
        result = app.invoke(state)

if __name__ == "__main__":
    main()
