
from langgraph.graph import StateGraph, END
from agents.researcher_agent import run_research_agent
from agents.answer_drafter_agent import run_drafter_agent

def input_node(state):
    print("🔁 Received input state:", state)
    return state

def research_node(state):
    query = state.get("query")
    run_research_agent(query)
    return {"query": query, "research_done": True}

def answer_node(state):
    query = state.get("query")
    run_drafter_agent(query)
    return {"query": query, "answer_done": True}

# Define state machine
def build_graph():
    graph = StateGraph()
    
    graph.add_node("Input", input_node)
    graph.add_node("Research", research_node)
    graph.add_node("Answer", answer_node)

    # Edges: Input → Research → Answer → END
    graph.set_entry_point("Input")
    graph.add_edge("Input", "Research")
    graph.add_edge("Research", "Answer")
    graph.add_edge("Answer", END)

    return graph.compile()
