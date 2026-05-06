from langgraph.graph import StateGraph, END, START # Added START
from src.state import AgentState
from src.agents import (
    research_analyst_node, 
    methodology_reviewer_node, 
    citation_analyst_node, 
    trend_intelligence_node,
    aggregator_node
)

def build_graph():
    workflow = StateGraph(AgentState)

    # 1. Add Nodes
    workflow.add_node("analyst", research_analyst_node)
    workflow.add_node("reviewer", methodology_reviewer_node)
    workflow.add_node("citator", citation_analyst_node)
    workflow.add_node("trend_intel", trend_intelligence_node)
    workflow.add_node("aggregator", aggregator_node)

    # 2. Parallel Fan-Out (The Fix)
    # Instead of set_entry_point([list]), we point START to all 4 nodes
    workflow.add_edge(START, "analyst")
    workflow.add_edge(START, "reviewer")
    workflow.add_edge(START, "citator")
    workflow.add_edge(START, "trend_intel")

    # 3. Fan-In (Wait for all to finish at the aggregator)
    workflow.add_edge("analyst", "aggregator")
    workflow.add_edge("reviewer", "aggregator")
    workflow.add_edge("citator", "aggregator")
    workflow.add_edge("trend_intel", "aggregator")

    # 4. Finish
    workflow.add_edge("aggregator", END)

    return workflow.compile()

app = build_graph()
