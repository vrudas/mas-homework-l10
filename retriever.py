"""
Hybrid retrieval module.

Combines semantic search (vector DB) + BM25 (lexical) + cross-encoder reranking.
"""
import pickle

from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from config import settings
from utils import index_exists

embedding_model_name = settings.embedding_model
embedding_model = OpenAIEmbeddings(
    api_key=settings.api_key,
    model=embedding_model_name
)


def get_retriever() -> ContextualCompressionRetriever | None:
    # 1. Load vector store from disk (config.index_dir)
    # 2. Create semantic retriever from vector store
    # 3. Load chunks and create BM25 retriever
    # 4. Combine into ensemble retriever (semantic + BM25)
    # 5. Add cross-encoder reranker on top
    # 6. Return the final retriever

    if not index_exists(settings.index_dir):
        print("⁉️ Index not found, please run 'ingest.py' first")
        return None

    vectorstore = FAISS.load_local(
        folder_path=settings.index_dir,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,
    )

    semantic_retreiver = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.retrieval_top_k}  # Adjust k as needed
    )

    bm25_retriever = create_bm25_retriever()

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retreiver],
        weights=[0.4, 0.6],  # Adjust weights as needed
    )

    print("⏳ Loading reranker model (first time may take a minute)...")
    reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")

    compressor = CrossEncoderReranker(
        model=reranker_model,
        top_n=settings.retrieval_top_k  # keep only top k most relevant, adjust k as needed
    )

    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever,
    )


def create_bm25_retriever() -> BM25Retriever:
    documents_for_bm25 = load_documents_for_bm25_retriever()
    bm25_retriever = BM25Retriever.from_documents(documents=documents_for_bm25)
    bm25_retriever.k = settings.retrieval_top_k  # Adjust k as needed
    return bm25_retriever


def load_documents_for_bm25_retriever() -> list[Document]:
    print("⏳ Loading documents for BM25 retriever from index.pkl...")

    with open(f"./{settings.index_dir}/index.pkl", "rb") as index:
        pkl_data = pickle.load(index)

    docstore = pkl_data[0]  # InMemoryDocstore
    index_map = pkl_data[1]  # {faiss_idx: doc_id}

    documents = [docstore.search(doc_id) for doc_id in index_map.values()]

    print(f"✅ Loaded {len(documents)} documents from index.pkl")

    return documents
