from deepeval.models import GPTModel
from deepeval.test_case import ToolCall
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage

from config import settings


def get_eval_model() -> GPTModel:
    """Central factory for the DeepEval evaluator LLM.

    Keeps the model id in one place (config.settings.eval_model_name) so a
    single env override reconfigures every metric across every test file.
    """
    return GPTModel(
        model=settings.eval_model_name,
        api_key=settings.api_key.get_secret_value(),
    )


def extract_tool_calls_from_runnable(
        agent,  # your create_agent() result
        user_input: str,
) -> tuple[str, list[ToolCall]]:
    """
    Invokes a LangGraph/LangChain Runnable agent and extracts
    tool calls from the message trajectory.
    """
    result = agent.invoke({"messages": [HumanMessage(content=user_input)]})

    # result["messages"] contains the full conversation trajectory
    messages = result["messages"]

    tool_calls: list[ToolCall] = []
    final_output = ""

    for msg in messages:
        # AIMessage with tool_calls = agent decided to call a tool
        if isinstance(msg, AIMessage):
            for tc in msg.tool_calls:  # list of dicts: {name, args, id}
                tool_calls.append(
                    ToolCall(
                        name=tc["name"],
                        input_parameters=tc["args"],
                    )
                )
            # Last AIMessage without tool_calls = final response
            if not msg.tool_calls and msg.content:
                final_output = (
                    msg.content
                    if isinstance(msg.content, str)
                    else str(msg.content)
                )

        # ToolMessage = tool execution result → attach output to last ToolCall
        elif isinstance(msg, ToolMessage) and tool_calls:
            tool_calls[-1] = ToolCall(
                name=tool_calls[-1].name,
                input_parameters=tool_calls[-1].input_parameters,
                output=msg.content,
            )

    # Handle structured output (ResearchPlan) — serialize it
    if not final_output and hasattr(result.get("messages", [])[-1], "content"):
        last = result["messages"][-1]
        if hasattr(last, "content"):
            final_output = str(last.content)

    return final_output, tool_calls
