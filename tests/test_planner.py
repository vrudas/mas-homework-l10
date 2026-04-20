from deepeval.evaluate import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

from agents.planner import planner_agent
from tests.testing_utils import get_eval_model

plan_quality = GEval(
    name="Plan Quality",
    evaluation_steps=[
        "Check that the plan contains specific search queries (not vague)",
        "Check that sources_to_check includes relevant sources for the topic",
        "Check that the output_format matches what the user asked for",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=get_eval_model(),
    threshold=0.6,
)

plan_quality_edge = GEval(
    name="Plan Quality (Edge)",
    evaluation_steps=[
        "Check that the plan contains specific search queries (not vague)",
        "Check that sources_to_check includes relevant sources for the topic",
        "Check that the output_format matches what the user asked for",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=get_eval_model(),
    threshold=0.6,
)


def _run_planner(user_input: str) -> str:
    return planner_agent.invoke({
        "messages": [{"role": "user", "content": user_input}]
    })["messages"][-1].content


def test_plan_quality():
    user_input = "Compare naive RAG vs sentence-window retrieval"
    test_case = LLMTestCase(input=user_input, actual_output=_run_planner(user_input))
    assert_test(test_case, [plan_quality])


def test_plan_quality_edge_case():
    user_input = "Compare LangChain, LangGraph, and Langfuse in one paragraph each covering purpose, core abstraction, and when to use."
    test_case = LLMTestCase(input=user_input, actual_output=_run_planner(user_input))
    assert_test(test_case, [plan_quality_edge])
