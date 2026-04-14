from datetime import date

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_openai import ChatOpenAI

import config
from config import settings
from schemas import CritiqueResult
from tools import web_search, read_url, knowledge_search

critic_agent = create_agent(
    model=ChatOpenAI(
        model=settings.model_name,
        api_key=settings.api_key,
    ),
    tools=[web_search, read_url, knowledge_search],
    system_prompt=config.CRITIC_SYSTEM_PROMPT.replace("{today}", date.today().strftime("%B %d, %Y")),
    response_format=CritiqueResult,
    middleware=[
        ModelCallLimitMiddleware(
            run_limit=settings.max_iterations
        )
    ],
)
# result["structured_response"] → validated CritiqueResult instance
