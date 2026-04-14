from langchain_core.tools import tool

from agents.critic import critic_agent
from agents.planner import planner_agent
from agents.research import research_agent
from schemas import ResearchPlan, CritiqueResult


@tool
def research(request: str) -> str:
    """Execute research based on a plan or request. Returns comprehensive findings.
    Args: request: research plan or specific instructions for what to investigate"""
    try:
        result = research_agent.invoke({"messages": [("user", request)]})
        messages = result.get("messages", [])
        return messages[-1].content if messages else "No findings returned."
    except Exception as e:
        print(f"Error during research: {e}")
        return f"Error during research for: {request}"


@tool
def critique(findings: str) -> str:
    """Evaluate research findings for freshness, completeness, and structure then return structured response as a CritiqueResult object.
    Args: findings: the research findings text to evaluate
    Example: {
        "verdict": "APPROVE"
        "is_fresh": "True"
        "is_complete": "True"
        "is_well_structured": "True"
        "strengths": ["strength1", "strength2", "strength3", "strength4", "strength5", "strength6"]
        "gaps": ["gap1", "gap2", "gap3", "gap4", "gap5", "gap6"]
        "revision_requests": ["request1", "request2", "request3", "request4", "request5", "request6"]
    }
    """

    try:
        result = critic_agent.invoke({"messages": [("user", findings)]})
        print(f"Critique: {result}")
        critique_result: CritiqueResult = result["structured_response"]
        return critique_result.model_dump_json(indent=2)
    except Exception as e:
        print(f"Error during critique: {e}")
        return f"Error during critique for: {findings}"


@tool
def plan(request: str) -> str:
    """Decompose a research request into a structured ResearchPlan.
    Args: request: the user's original research request"""
    try:
        result = planner_agent.invoke({"messages": [("user", request)]})
        research_plan: ResearchPlan = result["structured_response"]
        return research_plan.model_dump_json(indent=2)
    except Exception as e:
        print(f"Error during planning: {e}")
        return f"Error during planning for: {request}"
