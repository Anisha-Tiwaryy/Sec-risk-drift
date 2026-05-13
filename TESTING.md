# TESTING.md — SEC Risk Signal Drift Analyzer

Documented test cases covering core functionality, edge cases, and evaluation scenarios.

---

## Test Environment

```bash
pip install -r requirements.txt
python sec_analyzer.py          # standalone test
streamlit run app.py            # dashboard test
```

---

## Test Cases

### TC-01: Mock Data Generation

**What it tests:** `generate_mock_filings()` returns valid text for both years.  
**Steps:** `python -c "from sec_analyzer import generate_mock_filings; d = generate_mock_filings('AAPL'); print(len(d['2022']), len(d['2023']))"`  
**Expected:** Two non-empty strings, each > 1000 characters.  
**Status:** PASS ✅

---

### TC-02: Paragraph Splitting — Minimum Word Filter

**What it tests:** Paragraphs below 20 words are filtered out.  
**Steps:** Pass a text containing short lines (< 20 words) through `DriftAnalyzer()._split_paragraphs()`.  
**Expected:** Short lines excluded; result contains only paragraphs with ≥ 20 words.  
**Status:** PASS ✅

---

### TC-03: Sentence Transformer Encoding Shape

**What it tests:** `_encode()` returns correct shape and L2-normalized vectors.  
**Steps:** Encode 5 sample paragraphs; check `vecs.shape == (5, 384)` and norms ≈ 1.0.  
**Expected:** Shape `(5, 384)`, all norms in range `[0.99, 1.01]`.  
**Status:** PASS ✅

---

### TC-04: FAISS Index Build and Search

**What it tests:** `_build_faiss_index()` builds index; `_search()` returns valid indices.  
**Steps:** Build index from 10 random normalized vectors; search with a query vector; check returned index is in `[0, 9]`.  
**Expected:** Top-1 index is a valid integer; distance ≥ 0.  
**Status:** PASS ✅

---

### TC-05: Drift Classification — Identical Texts

**What it tests:** When old and new filings are identical, drift score ≈ 0.  
**Steps:** `analyze_drift(text, text, "2022", "2023")`.  
**Expected:** `drift_score < 0.05`, `new_signals` list nearly empty.  
**Status:** PASS ✅

---

### TC-06: Drift Classification — Completely Different Texts

**What it tests:** When texts are entirely different, drift score ≈ 1.0.  
**Steps:** `analyze_drift(text_a, text_b, "2022", "2023")` where texts share no topic.  
**Expected:** `drift_score > 0.80`, all paragraphs classified as NEW.  
**Status:** PASS ✅

---

### TC-07: Risk Category Classification

**What it tests:** `classify_risk_category()` assigns correct category for known keyword matches.  
**Steps:**  
  - Para containing "ransomware" → expect `"Cybersecurity"`  
  - Para containing "GDPR compliance" → expect `"Regulatory"`  
  - Para containing "interest rate" → expect `"Financial"`  
  - Para containing "generative artificial intelligence" → expect `"AI / Technology"`  
**Expected:** All four match expected categories.  
**Status:** PASS ✅

---

### TC-08: Evaluation Log Structure

**What it tests:** `evaluation_log` in result contains all required fields.  
**Required fields:** `run_timestamp`, `model`, `similarity_threshold`, `year_old`, `year_new`,
  `paragraphs_old`, `paragraphs_new`, `new_signals`, `removed_signals`, `persisted_signals`,
  `drift_score`, `category_breakdown_new`, `category_breakdown_removed`.  
**Expected:** All fields present with correct types.  
**Status:** PASS ✅

---

### TC-09: Gemini API Fallback

**What it tests:** `summarize_with_gemini()` gracefully handles missing API key.  
**Steps:** Call with `GEMINI_API_KEY = ""`.  
**Expected:** Returns a string containing "not configured" — no exception raised.  
**Status:** PASS ✅

---

### TC-10: EDGAR CIK Resolution — Invalid Ticker

**What it tests:** `get_company_cik()` raises `ValueError` for unknown tickers.  
**Steps:** `get_company_cik("XXXXINVALID")`.  
**Expected:** `ValueError: Ticker 'XXXXINVALID' not found in SEC EDGAR registry.`  
**Status:** PASS ✅

---

### TC-11: EDGAR Caching

**What it tests:** Second call to `get_company_cik()` with same ticker uses cache (no network call).  
**Steps:** Call twice; verify `sec_cache/` contains a `.json` file after first call; second call is instant.  
**Expected:** Cache file exists; second call returns same result without hitting EDGAR API.  
**Status:** PASS ✅

---

### TC-12: Similarity Matrix Dimensions

**What it tests:** `similarity_matrix` in result is correctly shaped (n_new × n_old, capped at 50).  
**Expected:** `len(similarity_matrix) <= 50` and `len(similarity_matrix[0]) <= 50`.  
**Status:** PASS ✅

---

### TC-13: Multi-Year Trend — Enough Years

**What it tests:** `compute_temporal_trend()` returns N-1 data points for N years of mock data.  
**Steps:** Mock data has 4 years (2020–2023); expect 3 trend entries.  
**Expected:** `len(trend) == 3`.  
**Status:** PASS ✅

---

### TC-14: CSV Export Content

**What it tests:** New signals CSV contains all expected columns.  
**Expected columns:** `Risk Category`, `Novelty Score`, `Cosine Sim`, `Paragraph`, `Provenance`.  
**Steps:** Trigger download button logic; parse CSV and check headers.  
**Status:** PASS ✅

---

### TC-15: Streamlit Session State Caching

**What it tests:** Re-running the same ticker/year combination does not re-trigger analysis.  
**Expected:** `STATE_KEY` found in `st.session_state` on second load; `run_analysis()` not called again.  
**Status:** PASS ✅

---

### TC-16: Edge Case — Very Short Filing Text

**What it tests:** Filing text with < 3 paragraphs after filtering doesn't crash the pipeline.  
**Steps:** Pass a 50-word string as filing text.  
**Expected:** Result returned with empty or near-empty signal lists; no exception.  
**Status:** PASS ✅

---

### TC-17: Novelty Score Range

**What it tests:** All novelty scores in `new_signals` are in `[0, 1]`.  
**Expected:** `all(0 <= s['novelty_score'] <= 1 for s in result['new_signals'])`.  
**Status:** PASS ✅

---

### TC-18: Numpy Fallback (no faiss-cpu)

**What it tests:** When FAISS is not installed, `_search()` falls back to numpy cosine similarity.  
**Expected:** Analysis completes without error; results are comparable to FAISS version.  
**Status:** PASS ✅

---

## Running All Tests

```python
# Quick smoke test — paste into Python REPL
from sec_analyzer import (
    generate_mock_filings, DriftAnalyzer, classify_risk_category,
    summarize_with_gemini, compute_temporal_trend
)

texts   = generate_mock_filings("AAPL")
analyzer = DriftAnalyzer()
result  = analyzer.analyze_drift(texts["2022"], texts["2023"], "2022", "2023")

assert 0 <= result["drift_score"] <= 1, "TC-01: drift_score out of range"
assert len(result["evaluation_log"]) >= 12, "TC-08: eval log missing fields"
assert all(0 <= s["novelty_score"] <= 1 for s in result["new_signals"]), "TC-17: novelty out of range"
assert classify_risk_category("ransomware data breach hacking") == "Cybersecurity", "TC-07a"
assert classify_risk_category("GDPR compliance regulation fine") == "Regulatory", "TC-07b"
print("All smoke tests passed.")
```
