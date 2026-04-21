import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Optional

class Embedder:
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_file: str = "data/embedded_data.pkl"
    ):
        self.model = SentenceTransformer(model_name)
        self.cache_file = cache_file
    
    def embed_documents(self, documents: List[dict]) -> np.ndarray:
        texts = [doc["text"] for doc in documents]
        print(f"Embedding {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=64)
        
        with open(self.cache_file, "wb") as f:
            pickle.dump({"documents": documents, "embeddings": embeddings}, f)
        
        print(f"Cached to {self.cache_file}")
        return embeddings
    
    def load_cached(self) -> Tuple[Optional[List[dict]], Optional[np.ndarray]]:
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                data = pickle.load(f)
            print(f"Loaded {len(data['documents'])} chunks from cache")
            return data["documents"], data["embeddings"]
        return None, None
    
    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode([query])[0]