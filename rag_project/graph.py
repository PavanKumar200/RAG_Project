from langgraph.graph import StateGraph, END
from typing import TypedDict, Callable, Any
from hitl import human_intervention

class State(TypedDict):
    query: str
    retriever: Any
    llm: Callable
    context: str
    answer: str

def process_node(state: State):
    retriever = state["retriever"]
    docs = retriever.invoke(state["query"])

    context = "\n".join([d.page_content for d in docs])

    answer = state["llm"](state["query"], context)

    return {"context": context, "answer": answer}

def should_escalate(state: State):
    answer = state["answer"].strip()
    if answer == "SYSTEM_ERROR_RATE_LIMIT":
        return "output"
    if answer == "ESCALATE":
        return "hitl"
    return "output"

def hitl_node(state: State):
    human_response = human_intervention(state["query"])
    return {"answer": human_response}

def output_node(state: State):
    if state["answer"].strip() == "SYSTEM_ERROR_RATE_LIMIT":
        print("\nAnswer: [System Error: Google Gemini API Rate Limit Exceeded. Please wait before asking again or upgrade your plan.]")
        return {"answer": "API Rate Limit Exceeded."}
    
    print("\nAnswer:", state["answer"])
    return {"answer": state["answer"]}

def build_graph():
    graph = StateGraph(State)

    graph.add_node("process", process_node)
    graph.add_node("hitl", hitl_node)
    graph.add_node("output", output_node)

    graph.set_entry_point("process")
    
    graph.add_conditional_edges(
        "process",
        should_escalate,
        {
            "hitl": "hitl",
            "output": "output"
        }
    )
    
    graph.add_edge("hitl", "output")
    graph.add_edge("output", END)

    return graph.compile()