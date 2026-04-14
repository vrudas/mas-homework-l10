from deepeval.evaluate import assert_test
from deepeval.metrics import GEval
from deepeval.models import GPTModel
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

from agents.planner import planner_agent
from config import settings

plan_quality = GEval(
    name="Plan Quality",
    evaluation_steps=[
        "Check that the plan contains specific search queries (not vague)",
        "Check that sources_to_check includes relevant sources for the topic",
        "Check that the output_format matches what the user asked for",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=GPTModel(
        model="gpt-5.4-mini",
        api_key=settings.api_key.get_secret_value(),
    ),
    threshold=0.7,
)

def test_plan_quality():
    test_case_input = "Compare naive RAG vs sentence-window retrieval"

    agent_output = planner_agent.invoke({
        "messages": [
            {"role": "user", "content": test_case_input}
        ]
    })["messages"][-1].content

    expected_output = """
    {
        "goal":"Compare naive RAG vs sentence-window retrieval",
        "search_queries":[
            "comparative analysis of naive RAG and sentence-window retrieval",
            "sentence-window retrieval effectiveness over naive RAG",
            "performance metrics of naive RAG vs sentence-window retrieval",
            "naive retrieval RAG applications and limitations",
            "sentence-window retrieval benefits in RAG systems",
            "case studies illustrating naive RAG vs sentence-window retrieval"
        ],
        "sources_to_check":["knowledge_base","web_search"],
        "output_format":"comparison table detailing features, advantages, and performance metrics of naive RAG and sentence-window retrieval"
    }"""

    test_case = LLMTestCase(
        input=test_case_input,
        actual_output=agent_output,
        expected_output=expected_output
    )

    assert_test(test_case, [plan_quality])
