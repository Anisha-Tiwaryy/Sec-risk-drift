# SEC Risk Signal Drift Analyzer

A semantic drift detection system for Fortune 500 10-K risk factor disclosures. Encodes annual filing paragraphs with Sentence Transformers, builds a FAISS vector index, and uses cosine similarity to flag risk signals that appeared, disappeared, or persisted year-over-year.

Built with Python, Sentence Transformers, FAISS, Gemini API, SEC EDGAR API, and Streamlit.

Live demo: https://sec-risk-drift-gysyvkrtskzd82ohvhf82q.streamlit.app

---

## What it does

Most NLP projects on financial documents are Q&A chatbots. This is different — it treats risk disclosure language as a signal stream and detects semantic change over time. A paragraph that was prominent in a company's 2022 10-K but absent in 2023 is flagged as a removed signal. A paragraph in 2023 with no close match in 2022 is flagged as a new signal. The system quantifies how much a company's risk language shifted between two filing years and surfaces the specific paragraphs driving that shift.

Practical output: "Company X quietly introduced cybersecurity incident language in FY23 while dropping all COVID-related supply chain disclosures."

---

## Technical approach

```
10-K Filing (SEC EDGAR API)
        |
        v
Extract Item 1A - Risk Factors section
        |
        v
Split into paragraphs (minimum 20 words)
        |
        v
Encode with Sentence Transformers (all-MiniLM-L6-v2, 384-dim, L2-normalized)
        |
        v
Build FAISS IndexFlatL2 per filing year
        |
        v
Compute cosine similarity for each paragraph across years
        |
        v
Classify: NEW (sim < 0.72) / PERSISTED (sim >= 0.72) / REMOVED
        |
        v
Auto-classify risk category (Cybersecurity, Regulatory, Financial, AI/Technology, etc.)
        |
        v
Gemini API summary of newly emerged risk themes
        |
        v
Streamlit dashboard + evaluation log export
```

---

## Dashboard views

| View | Description |
|------|-------------|
| Overview | System description, KPIs, full pipeline walkthrough |
| Drift Analysis | Signal flow chart, category breakdown, Gemini summary, drift gauge |
| New Risk Signals | Paragraph-level new signals with novelty scores, keyword search, CSV export |
| Removed Signals | Removed risk disclosures with category filter and CSV export |
| Similarity Heatmap | FAISS paragraph similarity matrix with max-sim distribution |
| Cross-Company Benchmark | Drift score comparison across 10 Fortune 500 companies |
| Temporal Trend | Multi-year drift trend with macro-event annotations |
| Evaluation Log | Structured run metadata with JSON export |

---

## Project structure

```
sec-risk-drift/
├── sec_analyzer.py       # Core engine: EDGAR API, embeddings, FAISS, drift logic, Gemini
├── app.py                # Streamlit dashboard
├── requirements.txt
├── README.md
├── TESTING.md            # 18 documented test cases
└── .github/
    └── workflows/
        └── ci.yml        # GitHub Actions CI
```

---

## Setup

```bash
git clone https://github.com/Anisha-Tiwaryy/sec-risk-drift.git
cd sec-risk-drift

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Optional - enables Gemini AI summaries
export GEMINI_API_KEY="your-key-here"

streamlit run app.py
```

The app runs in demo mode by default using realistic mock 10-K data - no API key or internet connection required. Toggle "Use SEC EDGAR API" in the sidebar to fetch live Fortune 500 filings.

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| Embedding model | all-MiniLM-L6-v2 | 384-dim Sentence Transformer |
| Similarity threshold | 0.72 | Below this, a paragraph is classified as new or removed |
| Min paragraph length | 20 words | Filters noise paragraphs |
| FAISS index type | IndexFlatL2 | Exact nearest-neighbor search |

---

## Data source

Live mode pulls from the SEC EDGAR REST API - the same public data source used by financial analysts and institutional investors. No authentication required.

Demo mode uses programmatically generated mock risk factor text designed to reflect known macro events: COVID-19 supply chain disclosures (2020-2021), AI regulation and cybersecurity escalation (2023), and geopolitical export control language (2023).

---

## Tech stack

| Component | Technology |
|-----------|------------|
| Embedding model | sentence-transformers (all-MiniLM-L6-v2) |
| Vector index | faiss-cpu (IndexFlatL2) |
| Filing ingestion | SEC EDGAR REST API |
| AI summaries | Google Gemini 1.5 Flash |
| Dashboard | Streamlit |
| Visualization | matplotlib, seaborn |
| Language | Python 3.10+ |

---

Anisha Tiwary - B.Tech Electronics and Computer Science, KIIT University 2026
github.com/Anisha-Tiwaryy | linkedin.com/in/anisha-tiwaryyy
