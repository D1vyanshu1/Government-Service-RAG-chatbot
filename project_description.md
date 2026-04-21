# Project: Indian Government-Services RAG Chatbot

## Purpose

A production-grade Retrieval-Augmented Generation (RAG) chatbot that answers questions
about Indian government services (Passport, Voter ID, RTI, Income Tax, Driving Licence,
Ayushman Bharat) by retrieving from official government PDFs and generating cited answers
via a free LLM API.

Built as a portfolio project for ML/Data Engineering internship applications .
The goal is NOT to train a model — it is to demonstrate production systems thinking:
data pipelines, caching, retrieval, LLM integration, deployment, and error handling.

---

## Repository Structure

```
gov-services-rag/
├── project_description.md               ← you are here (IDE context file)
├── README.md                ← public-facing project README
├── requirements.txt
├── .env                     ← never commit this
├── .env.example             ← commit this (template, no real keys)
├── .gitignore
│
├── app.py                   ← Streamlit entrypoint
│
├── src/
│   ├── __init__.py
│   ├── pdf_loader.py        ← PDF parsing + chunking
│   ├── embedder.py          ← sentence-transformers wrapper + pickle cache
│   ├── retriever.py         ← cosine similarity search
│   └── llm_client.py        ← Gemini API wrapper (generation + citation)
│
├── data/
│   ├── pdfs/                ← raw government PDFs (10–12 files, ~40–60 pages)
│   │   ├── passport_fresh.pdf
│   │   ├── passport_renewal.pdf
│   │   ├── voter_id_form6.pdf
│   │   ├── voter_id_faq.pdf
│   │   ├── rti_act_2005.pdf
│   │   ├── income_tax_itr1.pdf
│   │   ├── income_tax_itr2.pdf
│   │   ├── driving_licence_guide.pdf
│   │   ├── vehicle_registration_rc.pdf
│   │   └── ayushman_bharat_eligibility.pdf
│   └── embedded_data.pkl    ← cached embeddings (auto-generated, gitignored)
│
└── tests/
    ├── test_retriever.py    ← at least 3 test cases
    └── test_pdf_loader.py   ← chunk size, empty PDF handling
```

---

## Architecture

```
User Query (English or Hindi)
        │
        ▼
[ Embedder ]
  sentence-transformers/all-MiniLM-L6-v2
  384-dimensional dense vector
        │
        ▼
[ Retriever ]
  Cosine similarity over cached embeddings (numpy dot product)
  Returns top-5 (doc_name, text_chunk, similarity_score)
        │
        ▼
[ LLM Client ]
  Gemini 1.5 Flash (free tier)
  System prompt enforces: cite sources, answer only from docs,
  respond in user's chosen language
        │
        ▼
[ Streamlit UI ]
  Chat interface + source expander + language toggle
```

**Key design decision:** Embeddings are computed once at startup and cached to
`data/embedded_data.pkl`. Every query does only: one embed call + one numpy dot product +
one LLM API call. No vector DB needed at this scale (~1000 chunks from 10 PDFs).

When to upgrade to a vector DB: If corpus grows beyond ~50,000 chunks, replace the
numpy retriever with `chromadb` or `faiss-cpu` for approximate nearest neighbor search.

---

## Module Contracts

### `src/pdf_loader.py` — `PDFLoader`

**Responsibility:** Read all PDFs from `data/pdfs/`, extract text page by page,
split into overlapping chunks.

**Key method:**
```python
def load_all_pdfs(self) -> List[dict]:
    # Returns: [{"doc_name": str, "page": int, "text": str}, ...]
```

**Chunking strategy:** Split on sentence boundaries (`.`), accumulate until
chunk reaches `chunk_size` (default 512 chars), then start a new chunk.
Never split mid-sentence. Skip chunks shorter than 50 chars (they're noise).

**Error handling:** Wrap each PDF in try/except. Log failures, continue loading others.
A broken PDF must not crash the whole pipeline.

---

### `src/embedder.py` — `Embedder`

**Responsibility:** Embed text using sentence-transformers. Cache to pickle so
startup on subsequent runs is instant.

**Key methods:**
```python
def embed_documents(self, documents: List[dict]) -> np.ndarray  # shape: (N, 384)
def load_cached(self) -> Tuple[List[dict], np.ndarray]          # returns (None, None) if no cache
def embed_query(self, query: str) -> np.ndarray                 # shape: (384,)
```

**Cache behavior:** On first run, embed all chunks and save to `data/embedded_data.pkl`.
On subsequent runs, load from pickle — no re-embedding. If PDFs change, delete
the pickle to force re-indexing.

**Model:** `sentence-transformers/all-MiniLM-L6-v2` — 22MB, runs on CPU in <1s per query.

---

### `src/retriever.py` — `Retriever`

**Responsibility:** Given a query embedding, return the top-k most similar document chunks.

**Key method:**
```python
def retrieve(self, query_embedding: np.ndarray) -> List[Tuple[str, str, float]]:
    # Returns: [(doc_name, text_chunk, similarity_score), ...]
    # Sorted by similarity descending
```

**Math:** Cosine similarity = dot(embeddings, query) / (norm(embeddings) * norm(query)).
Uses numpy broadcasting — O(N) time, runs in <5ms for N=1000.

**Fallback behavior:** If the highest similarity score is below `threshold` (default 0.3),
return an empty list. The LLM client should detect this and say "I don't have
reliable information on this."

---

### `src/llm_client.py` — `LLMClient`

**Responsibility:** Given a user query + retrieved context, call Gemini API and
return a cited answer.

**Key method:**
```python
def generate_answer(self, query: str, retrieved_docs: List[Tuple], language: str) -> str:
```

**Prompt engineering rules:**
- System prompt instructs: answer ONLY from provided documents, cite the source PDF name,
  respond in the specified language, if information not found say so explicitly.
- Never hallucinate steps not present in the retrieved chunks.
- Format citations inline: "According to `passport_fresh.pdf` (page 3), ..."

**API:** Google Gemini 1.5 Flash. Free tier: 15 requests/minute, no daily limit.
Key loaded from environment variable `GOOGLE_API_KEY`.

**Error handling:** Wrap API call in try/except. On failure, return a user-friendly
error message, do not crash the app.

---

### `app.py` — Streamlit App

**Responsibility:** UI layer. Wires together all src modules.

**Features:**
- `st.chat_input` for query entry
- `st.chat_message` for conversation history (stored in `st.session_state.messages`)
- Sidebar: language toggle (English / Hindi), "About" section
- Source citations in `st.expander("📚 Sources")` below each answer
- `st.spinner` while retrieving + generating
- `@st.cache_resource` on pipeline loading — loads once, never reloads on rerun

**Startup flow:**
1. Load cached embeddings (or embed on first run)
2. Initialize retriever and LLM client
3. Render chat UI

---

## Environment Variables

```bash
# .env (never commit)
GOOGLE_API_KEY=your_key_here

# Optional fallback LLM
GROQ_API_KEY=your_key_here
```

Get Gemini key free at: https://ai.google.dev/
Get Groq key free at: https://console.groq.com/

---

## Tech Stack

| Component | Library | Why |
|-----------|---------|-----|
| PDF parsing | `pypdf` | Lightweight, handles most govt PDFs |
| Embedding | `sentence-transformers` | Free, CPU-friendly, 384-dim vectors |
| Retrieval | `numpy` | Cosine similarity is O(N), fast enough at this scale |
| Caching | `pickle` | Zero dependencies, instant load |
| LLM | `google-generativeai` | Free tier, strong multilingual, good instruction following |
| UI | `streamlit` | Fast to build, easy to deploy |
| Deployment | HuggingFace Spaces | Free, always-on, shareable link |

---

## Data Sources (PDFs to Download)

| Service | URL | Files to Download |
|---------|-----|-------------------|
| Passport | https://www.passportindia.gov.in | Instruction booklets: fresh + renewal |
| Voter ID | https://www.eci.gov.in | Form 6, Form 7, Form 8 instructions |
| RTI | https://rti.gov.in | RTI Act 2005 PDF |
| Income Tax | https://www.incometax.gov.in | ITR-1 and ITR-2 instruction booklets |
| Driving Licence | https://parivahan.gov.in | Sarathi guide + RC guide |
| Ayushman Bharat | https://nha.gov.in | PM-JAY eligibility and enrollment guide |

**Target:** 10–12 PDFs, 40–60 pages total. This gives ~800–1200 chunks after parsing.

---

## Performance Targets

| Metric | Target |
|--------|--------|
| Startup time (cold, first run) | < 60s (embedding ~800 chunks) |
| Startup time (warm, cached) | < 3s |
| Query latency (embed + retrieve) | < 200ms |
| LLM response latency | < 3s (Gemini free tier) |
| Total end-to-end | < 4s |

Measure these with `time.perf_counter()` and log them. Know your numbers — interviewers ask.

---

## Testing

### `tests/test_retriever.py`
- `test_returns_k_results` — retriever always returns exactly k results
- `test_scores_sorted_descending` — similarity scores are in descending order
- `test_relevant_doc_retrieved` — "passport renewal" query returns a passport PDF

### `tests/test_pdf_loader.py`
- `test_chunks_not_empty` — no chunk is shorter than 50 chars
- `test_all_pdfs_loaded` — number of unique doc_names == number of PDFs in folder
- `test_broken_pdf_does_not_crash` — passing a corrupt file doesn't raise an exception

Run with: `pytest tests/ -v`

---

## Deployment Checklist

```
[ ] All PDFs in data/pdfs/ (at least 8)
[ ] requirements.txt tested fresh (pip install -r requirements.txt in clean venv)
[ ] .env.example committed (no real keys)
[ ] .gitignore includes: .env, data/embedded_data.pkl, __pycache__, .venv
[ ] app.py runs locally, 10 test queries answered correctly
[ ] pytest tests/ passes
[ ] GitHub repo is public with descriptive commit messages
[ ] HuggingFace Space created (SDK: Streamlit)
[ ] GOOGLE_API_KEY added as HF Space secret (Settings → Repository secrets)
[ ] Live link tested from a different browser (not localhost)
[ ] README.md has: live demo link, architecture, tech stack, setup instructions
```

---

## Known Limitations (be honest in interviews)

1. **Manual PDF updates** — If the government updates a document, someone must re-download
   and re-index manually. Fix: scheduled job (GitHub Actions cron) to re-download and
   rebuild the pickle.

2. **No conversation memory** — Each query is independent. The system doesn't know
   "what did I ask before?" Fix: pass last N messages as additional context to the LLM.

3. **Pickle is not a vector DB** — At 10,000+ chunks, cosine similarity over the whole
   corpus becomes slow. Fix: migrate to `chromadb` or `faiss-cpu` with ANN indexing.

4. **English PDF + Hindi query mismatch** — Embedding model aligns English and Hindi
   semantically, but retrieval quality drops for Hindi queries against English PDFs.
   Fix: translate query to English before embedding, then generate answer in Hindi.

5. **No rate limiting** — Free Gemini tier allows 15 req/min. Under heavy use, this breaks.
   Fix: add a token bucket rate limiter around `llm_client.generate_answer`.

---

## Interview Talking Points

**Q: Walk me through your data pipeline.**
A: PDFs → page extraction (pypdf) → sentence-boundary chunking (512 chars) →
   dense embedding (all-MiniLM-L6-v2, 384-dim) → cached to pickle. At query time:
   embed query → cosine similarity over all cached vectors (numpy, O(N)) → top-5 →
   Gemini prompt with context → cited answer. One-time indexing cost, fast at query time.

**Q: Why not fine-tune a model?**
A: No labeled (question, answer) pairs exist for this domain. RAG is the right tool:
   it's auditable (cites sources), handles document updates without retraining, and
   runs on free-tier compute. Fine-tuning would cost GPU hours and still hallucinate.

**Q: How would you scale to 100,000 documents?**
A: Replace pickle + numpy with Faiss (approximate nearest neighbor, sublinear retrieval).
   Move to async LLM calls. Add Redis caching for repeated queries. Batch-index new
   documents nightly via a scheduled job.

**Q: What would you add next?**
A: Multi-turn conversation memory (pass chat history as context), user feedback loop
   (thumbs up/down to measure retrieval quality), and auto-refresh of PDFs via cron.

---

## Day-by-Day Build Plan

| Day | Task | Done? |
|-----|------|-------|
| 1 | Repo setup, download PDFs, get API key | [ ] |
| 2 | Build pdf_loader, embedder, retriever, llm_client | [ ] |
| 3 | Build Streamlit app, test locally | [ ] |
| 4 | Error handling, edge cases, write tests, measure latency | [ ] |
| 5 | Deploy to HuggingFace Spaces, clean GitHub | [ ] |
| 6 | Write README, record demo video | [ ] |
| 7 | Interview prep: answer all questions above out loud | [ ] |