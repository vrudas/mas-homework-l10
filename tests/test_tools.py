from deepeval.evaluate import assert_test
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

from agents.planner import planner_agent
from agents.research import research_agent
from supervisor import build_supervisor
from tests.testing_utils import extract_tool_calls_from_runnable, get_eval_model

llm_model = get_eval_model()


def test_planner_tools():
    tool_name_correctness_metric = ToolCorrectnessMetric(
        threshold=0.5,
        model=llm_model
    )

    user_input = "Langchain vs LangGraph vs Langfuse"

    actual_output, actual_tool_calls = extract_tool_calls_from_runnable(
        planner_agent, user_input
    )

    test_case = LLMTestCase(
        input=user_input,
        actual_output=actual_output,
        tools_called=actual_tool_calls,
        expected_tools=[
            ToolCall(name="web_search", input_parameters={"query": user_input}),
            ToolCall(name="knowledge_search", input_parameters={"query": user_input}),
        ],
    )

    assert_test(test_case, [tool_name_correctness_metric])


def test_researcher_tools():
    tool_name_correctness_metric = ToolCorrectnessMetric(
        threshold=0.3,
        model=llm_model
    )

    user_input = """
    {
        "goal":"Explain the role of Retrieval-Augmented Generation (RAG) in AI systems: what it does, why it is used, and the main benefits/limitations.",
        "search_queries":[
            "retrieval-augmented generation role definition benefits limitations hallucination reduction",
            "RAG how it works retrieve documents then generate response use cases",
            "retrieval-augmented generation in LLMs external knowledge base updated information"
        ],
        "sources_to_check":["knowledge_base","web_search"],
        "output_format":"Short explanatory answer with 1) a plain-English definition of RAG’s role, 2) key reasons it is used, 3) a brief list of benefits and limitations, and 4) one concrete example use case."
    }
    """

    actual_output, actual_tool_calls = extract_tool_calls_from_runnable(
        research_agent, user_input
    )

    test_case = LLMTestCase(
        input=user_input,
        actual_output=actual_output,
        tools_called=actual_tool_calls,
        expected_tools=[
            ToolCall(name="knowledge_search", input_parameters={
                "query": "RAG how it works retrieve documents then generate response use cases"}),
            ToolCall(name="web_search", input_parameters={
                "query": "retrieval-augmented generation role definition benefits limitations hallucination reduction"}),
            ToolCall(name="read_url", input_parameters={
                "url": "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"}),
        ],
    )

    assert_test(test_case, [tool_name_correctness_metric])


def test_supervisor_save():
    tool_name_correctness_metric = ToolCorrectnessMetric(
        threshold=0.5,
        model=llm_model
    )

    user_input = "How to build RAG?"
    supervisor_agent = build_supervisor(hitl=False)

    actual_output, actual_tool_calls = extract_tool_calls_from_runnable(
        supervisor_agent.with_config(
            config={"configurable": {"thread_id": "thread-1"}}
        ), user_input
    )

    test_case = LLMTestCase(
        input=user_input,
        actual_output=actual_output,
        tools_called=actual_tool_calls,
        expected_tools=[
            ToolCall(name="plan", input_parameters={"request": user_input}),
            ToolCall(name="research"),
            ToolCall(name="critique"),
            ToolCall(name="save_report"),
        ],
    )

    assert_test(test_case, [tool_name_correctness_metric])
