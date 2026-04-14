from deepeval.evaluate import assert_test
from deepeval.metrics import GEval
from deepeval.models import GPTModel
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

from agents.research import research_agent
from config import settings

from retriever import get_retriever

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
    model=GPTModel(
        model="gpt-5.4-mini",
        api_key=settings.api_key.get_secret_value(),
    ),
    threshold=0.7,
)


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
        "output_format":"Short explanatory answer with 1) a plain-English definition of RAG’s role, 2) key reasons it is used, 3) a brief list of benefits and limitations, and 4) one concrete example use case."
    }
   """

    agent_output = research_agent.invoke({
        "messages": [
            {"role": "user", "content": test_case_input}
        ]
    })["messages"][-1].content

    retreived_documents = retriever.invoke("RAG how it works retrieve documents then generate response use cases")
    retrieval_context = [document.page_content for document in retreived_documents]

    test_case = LLMTestCase(
        input=test_case_input,
        actual_output=agent_output,
        retrieval_context=retrieval_context
    )

    assert_test(test_case, [groundedness])
