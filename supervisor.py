from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from config import settings, SUPERVISOR_SYSTEM_PROMPT
from agents.agentic_tools import plan, research, critique
from tools import save_report

supervisor_agent = create_agent(
    model=ChatOpenAI(
        model=settings.model_name,
        api_key=settings.api_key,
    ),
    tools=[plan, research, critique, save_report],
    system_prompt=SUPERVISOR_SYSTEM_PROMPT,
    middleware=[
        HumanInTheLoopMiddleware(interrupt_on={"save_report": True}),
    ],
    checkpointer=InMemorySaver(),
)
