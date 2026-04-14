import os
import pickle

import faiss

from config import settings

_index_directory = settings.index_dir
_index_file_name_prefix = "index"


def debug_pickle():
    with open(f"./{_index_directory}/{_index_file_name_prefix}.pkl", "rb") as f:
        data = pickle.load(f)

    print(type(data))
    print(data)


def debug_faiss():
    index = faiss.read_index(f"./{_index_directory}/{_index_file_name_prefix}.faiss")
    print(f"Total vectors: {index.ntotal}")
    print(f"Dimensions: {index.d}")


def index_exists(path: str) -> bool:
    return (
            os.path.exists(os.path.join(path, "index.faiss")) and
            os.path.exists(os.path.join(path, "index.pkl"))
    )
