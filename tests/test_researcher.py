from deepeval.evaluate import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

from agents.research import research_agent
from retriever import get_retriever
from tests.testing_utils import get_eval_model

retriever = get_retriever()

groundedness = GEval(
    name="Groundedness",
    evaluation_steps=[
        "Extract every factual claim from 'actual output'",
        "For each claim, check if it can be directly supported by 'retrieval context'",
        "Claims not present in retrieval context count as ungrounded, even if true",
        "Score = number of grounded claims / total claims",
    ],
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    model=get_eval_model(),
    threshold=0.7,
)

groundedness_edge = GEval(
    name="Groundedness (Edge)",
    evaluation_steps=[
        "Extract every factual claim from 'actual output'",
        "For each claim, check if it can be directly supported by 'retrieval context'",
        "Claims not present in retrieval context count as ungrounded, even if true",
        "Score = number of grounded claims / total claims",
    ],
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    model=get_eval_model(),
    threshold=0.1,
)


def _run_research(plan_json: str, retrieval_query: str) -> tuple[str, list[str]]:
    output = research_agent.invoke({
        "messages": [{"role": "user", "content": plan_json}]
    })["messages"][-1].content

    docs = retriever.invoke(retrieval_query)
    retrieval_context = [d.page_content for d in docs]

    return output, retrieval_context


def test_groundedness():
    test_case_input = """
    {
        "goal":"Explain the role of Retrieval-Augmented Generation (RAG) in AI systems: what it does, why it is used, and the main benefits/limitations.",
        "search_queries":[
            "retrieval-augmented generation role definition benefits limitations hallucination reduction",
            "RAG how it works retrieve documents then generate response use cases",
            "retrieval-augmented generation in LLMs external knowledge base updated information"
        ],
        "sources_to_check":["knowledge_base","web_search"],
        "output_format":"Short explanatory answer with 1) a plain-English definition of RAG's role, 2) key reasons it is used, 3) a brief list of benefits and limitations, and 4) one concrete example use case."
    }
    """
    output, ctx = _run_research(
        test_case_input,
        "RAG how it works retrieve documents then generate response use cases",
    )

    test_case = LLMTestCase(input=test_case_input, actual_output=output, retrieval_context=ctx)

    assert_test(test_case, [groundedness])


def test_groundedness_edge_case():
    plan_json = """
            {
                "goal":"Describe the role of a cross-encoder reranker in a RAG pipeline.",
                "search_queries":[
                    "cross-encoder reranker RAG pipeline",
                    "bi-encoder vs cross-encoder retrieval"
                ],
                "sources_to_check":["knowledge_base","web_search"],
                "output_format":"One paragraph explanation."
            }
            """

    retrieval_query = "cross-encoder reranker RAG pipeline"

    output, ctx = _run_research(plan_json, retrieval_query)
    test_case = LLMTestCase(input=plan_json, actual_output=output, retrieval_context=ctx)
    assert_test(test_case, [groundedness_edge])
