# to check working of pdf_loader.py (to form chunks from pdf)

# from src.pdf_loader import PDFLoader
# loader = PDFLoader()
# docs = loader.load_all_pdfs()

# print(f"Total chunks: {len(docs)}")
# print(f"PDFs loaded: {len(set(d['doc_name'] for d in docs))}")
# print(f"\nSample chunk:")
# print(docs[0])


# # to check working of embedder.py
# import time
# from src.pdf_loader import PDFLoader
# from src.embedder import Embedder

# loader = PDFLoader()
# docs = loader.load_all_pdfs()

# embedder = Embedder()

# # First run: embed and cache
# start = time.perf_counter()
# embeddings = embedder.embed_documents(docs)
# cold_time = time.perf_counter() - start
# print(f"\nCold embedding time: {cold_time:.2f}s")
# print(f"Embeddings shape: {embeddings.shape}")

# # Second run: load from cache
# start = time.perf_counter()
# cached_docs, cached_embeddings = embedder.load_cached()
# warm_time = time.perf_counter() - start
# print(f"Warm load time: {warm_time:.2f}s")

# # Test query embedding
# start = time.perf_counter()
# q_emb = embedder.embed_query("How do I apply for a passport?")
# query_time = time.perf_counter() - start
# print(f"Query embed time: {query_time*1000:.1f}ms")
# print(f"Query embedding shape: {q_emb.shape}")

# # to test retrieve.py
# import time
# from src.pdf_loader import PDFLoader
# from src.embedder import Embedder
# from src.retriever import Retriever

# loader = PDFLoader()
# docs = loader.load_all_pdfs()

# embedder = Embedder()
# cached_docs, cached_embeddings = embedder.load_cached()

# retriever = Retriever(cached_docs, cached_embeddings)

# # test queries
# queries = [
#     "How do I apply for a passport?",
#     "What documents are needed for voter ID registration?",
#     "How do I file an RTI application?",
#     "What is the income tax return deadline?",
#     "Who is eligible for Ayushman Bharat?"
# ]

# for query in queries:
#     start = time.perf_counter()
#     q_emb = embedder.embed_query(query)
#     results = retriever.retrieve(q_emb)
#     elapsed = (time.perf_counter() - start) * 1000
    
#     print(f"\nQuery: {query}")
#     print(f"Time: {elapsed:.1f}ms | Results: {len(results)}")
#     for doc_name, text, score in results:
#         print(f"  [{score:.3f}] {doc_name} — {text[:80]}...")



import time
from dotenv import load_dotenv
from src.pdf_loader import PDFLoader
from src.embedder import Embedder
from src.retriever import Retriever
from src.llm_client import LLMClient

load_dotenv()

# load pipeline
docs, embeddings = Embedder().load_cached()
retriever = Retriever(docs, embeddings)
embedder = Embedder()
llm = LLMClient()

# test
queries = [
    ("How do I apply for a fresh passport?", "English"),
    ("voter ID के लिए कौन से documents चाहिए?", "Hindi"),
]

for query, language in queries:
    print(f"\n{'='*60}")
    print(f"Query ({language}): {query}")
    
    start = time.perf_counter()
    q_emb = embedder.embed_query(query)
    results = retriever.retrieve(q_emb)
    answer = llm.generate_answer(query, results, language)
    elapsed = time.perf_counter() - start
    
    print(f"Time: {elapsed:.2f}s")
    print(f"Answer:\n{answer}")