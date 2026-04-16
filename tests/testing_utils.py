from deepeval.test_case import ToolCall
from langchain_classic.agents import AgentExecutor
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage


def extract_tool_calls_from_agent(
        agent_executor: AgentExecutor,
        user_input: str,
) -> tuple[str, list[ToolCall]]:
    """
    Runs the agent and extracts actual output + tool calls
    in deepeval's ToolCall format.
    """

    result = agent_executor.invoke({"input": user_input})
    final_output = result["output"]

    tool_calls: list[ToolCall] = []

    for action, tool_output in result.get("intermediate_steps", []):
        tool_calls.append(
            ToolCall(
                name=action.tool,
                input_parameters=action.tool_input
                if isinstance(action.tool_input, dict)
                else {"input": action.tool_input},
                output=str(tool_output),
            )
        )

    return final_output, tool_calls


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
