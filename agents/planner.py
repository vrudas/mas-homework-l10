from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_openai import ChatOpenAI

from config import settings, langfuse
from schemas import ResearchPlan
from tools import web_search, knowledge_search

planner_agent = create_agent(
    model=ChatOpenAI(
        model=settings.model_name,
        api_key=settings.api_key,
    ),
    tools=[web_search, knowledge_search],
    system_prompt=langfuse.get_prompt("planner_system_prompt", label="production").compile(),
    response_format=ResearchPlan,
    middleware=[
        ModelCallLimitMiddleware(
            run_limit=settings.max_iterations
        )
    ],
)
# result["structured_response"] → validated ResearchPlan instance
