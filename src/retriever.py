import numpy as np
from typing import List, Tuple

class Retriever:
    
    def __init__(self, documents: List[dict], embeddings: np.ndarray, k: int = 5, threshold: float = 0.3):
        self.documents = documents
        self.embeddings = embeddings
        self.k = k
        self.threshold = threshold
        # pre-normalize all embeddings once, so retrieval is just a dot product
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.normalized_embeddings = self.embeddings / (norms + 1e-8)
    
    def retrieve(self, query_embedding: np.ndarray) -> List[Tuple[str, str, float]]:
        # normalize query
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        
        # cosine similarity = dot product of normalized vectors
        similarities = np.dot(self.normalized_embeddings, query_norm)
        
        top_indices = np.argsort(similarities)[::-1][:self.k]
        
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score < self.threshold:
                break
            results.append((
                self.documents[idx]["doc_name"],
                self.documents[idx]["text"],
                score
            ))
        
        return results