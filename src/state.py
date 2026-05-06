from typing import TypedDict, Annotated, List
import operator

class AgentState(TypedDict):
    # The data retrieved from Pinecone
    paper_context: List[str]
    
    # Specific outputs from your new agent structure
    # Annotated with operator.add so agents can build on each other's notes
    research_summary: Annotated[List[str], operator.add]
    methodology_critique: Annotated[List[str], operator.add]
    citation_audit: Annotated[List[str], operator.add]
    trend_analysis: Annotated[List[str], operator.add]
    
    # Final assembly
    final_compiled_report: str