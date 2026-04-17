from deepeval.evaluate import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

from agents.critic import critic_agent
from tests.testing_utils import get_eval_model

critique_quality = GEval(
    name="Critique Quality",
    evaluation_steps=[
        "Check that the critique identifies specific issues, not vague complaints",
        "Check that revision_requests are actionable (researcher can act on them)",
        "If verdict is APPROVE, gaps list should be empty or contain only minor items",
        "If verdict is REVISE, there must be at least one revision_request",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=get_eval_model(),
    threshold=0.6,
)


def test_critique_quality():
    test_case_input = """
    1) Plain-English definition: Retrieval-Augmented Generation (RAG) is a way to make an AI model answer using both its built-in training and extra, up-to-date information pulled from external documents or databases before it responds.

    2) Why it is used: It helps the model access domain-specific or current information that may not be in its training data, and it can reduce the need to retrain the model every time new information appears.

    3) Benefits and limitations:
    - Benefits:
      - Improves factuality by grounding answers in retrieved sources
      - Can reduce hallucinations
      - Supports updated or private knowledge bases
      - Can provide source traceability/transparency
    - Limitations:
      - If retrieval finds the wrong documents, the answer can still be wrong
      - The model may misread or misuse retrieved text out of context
      - It depends on the quality and freshness of the external knowledge base
      - It adds system complexity and retrieval overhead

    4) Example use case: A customer-support chatbot for a company can use RAG to search the company’s internal help docs and policy pages before generating an answer to a user’s question.
    """

    agent_output = critic_agent.invoke({
        "messages": [
            {"role": "user", "content": test_case_input}
        ]
    })["messages"][-1].content

    expected_output = """
    {
        "verdict":"APPROVE",
        "is_fresh":true,
        "is_complete":true,
        "is_well_structured":true,
        "strengths":[
            "Covers the requested definition, purpose, benefits, limitations, and example use case.",
            "Uses plain-English language appropriate for a non-technical audience.",
            "Includes both upside and downside points, giving a balanced explanation of RAG.",
            "The example is concrete and directly matches the customer-support chatbot scenario requested."
        ],
        "gaps":[],
        "revision_requests":[]
    }"""

    test_case = LLMTestCase(
        input=test_case_input,
        actual_output=agent_output,
        expected_output=expected_output
    )

    assert_test(test_case, [critique_quality])


def test_critique_revise():
    test_case_input = """
    RAG is retrieval augmented generation. It uses retrieval and generation.
    It has some benefits.
    """

    agent_output = critic_agent.invoke({
        "messages": [
            {"role": "user", "content": test_case_input}
        ]
    })["messages"][-1].content

    expected_output = """
    {
        "verdict":"REFISE",
        "is_fresh":false,
        "is_complete":false,
        "is_well_structured":false,
        "strengths":[],
        "gaps":[
            "The definition is too vague and doesn't explain what RAG actually does or how it works.",
            "The purpose of RAG is not explained at all.",
            "The benefits are mentioned but not specified, and limitations are not mentioned at all.",
            "The response is very brief and lacks the depth and structure expected for this topic."
        ],
        "revision_requests":[]
    }"""

    test_case = LLMTestCase(
        input=test_case_input,
        actual_output=agent_output,
        expected_output=expected_output
    )

    assert_test(test_case, [critique_quality])
