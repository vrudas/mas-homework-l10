from pathlib import Path

from langfuse import Langfuse
from pydantic import SecretStr
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    api_key: SecretStr
    model_name: str = "gpt-4o-mini"
    eval_model_name: str = "gpt-5.4-mini"

    langfuse_secret_key: SecretStr
    langfuse_public_key: SecretStr
    langfuse_base_url: SecretStr

    skip_details: bool = True

    # Web search
    max_search_results: int = 5
    max_url_content_length: int = 5000

    # RAG
    embedding_model: str = "text-embedding-3-small"
    data_dir: str = "data"
    index_dir: str = "index"
    chunk_size: int = 500
    chunk_overlap: int = 100
    retrieval_top_k: int = 10
    rerank_top_n: int = 3

    # Agent
    output_dir: str = "output"
    max_iterations: int = 5

    model_config = {"env_file": Path(__file__).parent / ".env"}


settings = Settings()

langfuse = Langfuse(
    public_key=settings.langfuse_public_key.get_secret_value(),
    secret_key=settings.langfuse_secret_key.get_secret_value(),
    # base_url="https://cloud.langfuse.com", # 🇪🇺 EU region
    base_url="https://us.cloud.langfuse.com",  # 🇺🇸 US region
)
