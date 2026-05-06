import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Initialize the High-Speed "Brain"
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.1)

def research_analyst_node(state):
    """Extracts the 'What & Why' - Objectives and Findings."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are the Lead Research Analyst. Your goal is to identify the core problem, research objectives, and primary findings of the paper. Be concise and factual."),
        ("user", "Analyze this research context: {context}")
    ])
    chain = prompt | llm
    response = chain.invoke({"context": state["paper_context"]})
    return {"research_summary": [response.content]}

def methodology_reviewer_node(state):
    """Checks the 'How' - Critiques design and logic."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Senior Methodology Reviewer. Critique the research design, sample sizes, and whether the data actually supports the conclusion. Identify any potential biases or logical gaps."),
        ("user", "Review the methodology in this context: {context}")
    ])
    chain = prompt | llm
    response = chain.invoke({"context": state["paper_context"]})
    return {"methodology_critique": [response.content]}

def citation_analyst_node(state):
    """Evaluates the 'Who & Where' - Literature and Citations."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Citation Analyst. Evaluate the paper’s place in the scientific field. Detect if key foundational works are referenced and if the sources are up-to-date."),
        ("user", "Audit the citations/references in this context: {context}")
    ])
    chain = prompt | llm
    response = chain.invoke({"context": state["paper_context"]})
    return {"citation_audit": [response.content]}

def trend_intelligence_node(state):
    """Forecasts the 'Future' - Scaling and Trends."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Trend Intelligence Agent. It is 2026. Connect this research to current industry trends and suggest how this work can be scaled, commercialized, or expanded in the future."),
        ("user", "Identify trends and scaling opportunities: {context}")
    ])
    chain = prompt | llm
    response = chain.invoke({"context": state["paper_context"]})
    return {"trend_analysis": [response.content]}

from src.prompts import AGGREGATOR_PROMPT

def aggregator_node(state):
    """Combines all agent outputs into a final structured report."""
    # Combine all previous analysis into one big string for the LLM to read
    combined_analysis = f"""
    RESEARCH SUMMARY: {state['research_summary']}
    METHODOLOGY CRITIQUE: {state['methodology_critique']}
    CITATION AUDIT: {state['citation_audit']}
    TREND ANALYSIS: {state['trend_analysis']}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", AGGREGATOR_PROMPT),
        ("user", "Synthesize this analysis into a final report: {analysis}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"analysis": combined_analysis})
    
    # We update the 'final_compiled_report' key in our State
    return {"final_compiled_report": response.content}