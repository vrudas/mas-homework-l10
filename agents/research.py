from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_openai import ChatOpenAI

from config import settings, RESEARCH_SYSTEM_PROMPT
from tools import web_search, read_url, knowledge_search

research_agent = create_agent(
    model=ChatOpenAI(
        model=settings.model_name,
        api_key=settings.api_key,
    ),
    tools=[web_search, read_url, knowledge_search],
    middleware=[
        ModelCallLimitMiddleware(
            run_limit=settings.max_iterations
        )
    ],
    system_prompt=RESEARCH_SYSTEM_PROMPT,
)
