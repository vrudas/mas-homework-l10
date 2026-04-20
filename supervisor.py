from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from agents.agentic_tools import plan, research, critique
from config import settings, SUPERVISOR_SYSTEM_PROMPT
from tools import save_report


def build_supervisor(hitl: bool = True):
    """Build a supervisor agent.

    Args:
        hitl: when True (default, production), wraps save_report in a
            HumanInTheLoopMiddleware interrupt so the user approves before
            writing to disk. Set to False in tests that need the full
            Planner→Researcher→Critic→save_report trajectory without a human
            involvement step.
    """

    middleware = (
        [HumanInTheLoopMiddleware(interrupt_on={"save_report": True})] if hitl else []
    )

    return create_agent(
        model=ChatOpenAI(
            model=settings.model_name,
            api_key=settings.api_key,
        ),
        tools=[plan, research, critique, save_report],
        system_prompt=SUPERVISOR_SYSTEM_PROMPT,
        middleware=middleware,
        checkpointer=InMemorySaver(),
    )


supervisor_agent = build_supervisor(hitl=True)
