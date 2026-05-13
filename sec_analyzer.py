"""
sec_analyzer.py
---------------
SEC Risk Signal Drift Analyzer — Core Engine

Pipeline:
  1. Fetch 10-K filings from SEC EDGAR API (or use mock data)
  2. Extract Item 1A (Risk Factors) section
  3. Split into paragraphs and encode with Sentence Transformers
  4. Build FAISS IndexFlatL2 for ANN retrieval
  5. Compute cosine similarity across filing years
  6. Classify signals as NEW / REMOVED / PERSISTED
  7. Summarize new signals via Gemini API
  8. Export evaluation logs as JSON

Author: Anisha Tiwary | KIIT University 2026
"""

import os
import re
import json
import time
import hashlib
import requests
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional

warnings.filterwarnings("ignore")

# ── Optional dependencies ────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    print("[WARN] sentence-transformers not installed. Using random embeddings for demo.")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("[WARN] faiss-cpu not installed. Falling back to numpy cosine similarity.")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# ── Constants ────────────────────────────────────────────────────────────────
CACHE_DIR        = Path("sec_cache")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # 384-dim, fast, strong on financial text
SIMILARITY_THRESHOLD = 0.72              # below this → signal treated as NEW
MIN_PARA_WORDS   = 20                    # filter noise paragraphs
EDGAR_BASE       = "https://data.sec.gov"
EDGAR_HEADERS    = {"User-Agent": "anishadgp04@gmail.com SEC-Risk-Drift-Analyzer/1.0"}

CACHE_DIR.mkdir(exist_ok=True)

# ── Risk category taxonomy ───────────────────────────────────────────────────
RISK_KEYWORDS = {
    "Cybersecurity":  ["cyber", "ransomware", "data breach", "hacking", "malware",
                       "phishing", "information security", "unauthorized access"],
    "Regulatory":     ["regulation", "compliance", "SEC", "GDPR", "antitrust",
                       "legislation", "legal", "penalty", "sanction", "litigation"],
    "Financial":      ["interest rate", "credit risk", "liquidity", "debt", "capital",
                       "currency", "inflation", "bankruptcy", "impairment", "write-down"],
    "Operational":    ["supply chain", "manufacturing", "operations", "logistics",
                       "workforce", "talent", "third-party", "vendor", "disruption"],
    "Market":         ["competition", "market share", "pricing pressure", "demand",
                       "consumer", "customer", "product adoption", "revenue"],
    "AI / Technology":["artificial intelligence", "AI", "machine learning", "algorithm",
                       "automation", "cloud", "technology", "software", "digital"],
    "Geopolitical":   ["geopolitical", "trade war", "tariff", "sanctions", "conflict",
                       "international", "export control", "war", "political"],
    "Climate":        ["climate", "ESG", "environmental", "carbon", "sustainability",
                       "natural disaster", "weather", "flooding", "drought"],
}


def classify_risk_category(paragraph: str) -> str:
    """Rule-based risk category classification using keyword matching."""
    para_lower = paragraph.lower()
    scores = {}
    for category, keywords in RISK_KEYWORDS.items():
        scores[category] = sum(1 for kw in keywords if kw in para_lower)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "General"


# ── EDGAR API helpers ────────────────────────────────────────────────────────

def _cache_path(key: str) -> Path:
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    return CACHE_DIR / f"{h}.json"


def _cached_get(url: str) -> Optional[dict]:
    cp = _cache_path(url)
    if cp.exists():
        with open(cp) as f:
            return json.load(f)
    return None


def _cached_set(url: str, data: dict) -> None:
    cp = _cache_path(url)
    with open(cp, "w") as f:
        json.dump(data, f)


def get_company_cik(ticker: str) -> str:
    """Resolve ticker → CIK from SEC EDGAR company tickers JSON."""
    cached = _cached_get(f"cik:{ticker}")
    if cached:
        return cached["cik"]

    url = f"{EDGAR_BASE}/files/company_tickers.json"
    resp = requests.get(url, headers=EDGAR_HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    for entry in data.values():
        if entry.get("ticker", "").upper() == ticker.upper():
            cik = str(entry["cik_str"]).zfill(10)
            _cached_set(f"cik:{ticker}", {"cik": cik})
            return cik

    raise ValueError(f"Ticker '{ticker}' not found in SEC EDGAR registry.")


def get_10k_filings(cik: str, limit: int = 5) -> list[dict]:
    """Return metadata for the most recent 10-K filings for a given CIK."""
    cached = _cached_get(f"filings:{cik}")
    if cached:
        return cached["filings"]

    url = f"{EDGAR_BASE}/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=10-K&dateb=&owner=include&count={limit}&search_text=&output=atom"
    # Use submissions API instead (more reliable)
    sub_url = f"{EDGAR_BASE}/submissions/CIK{cik}.json"
    resp = requests.get(sub_url, headers=EDGAR_HEADERS, timeout=15)
    resp.raise_for_status()
    subs = resp.json()

    filings = []
    forms   = subs.get("filings", {}).get("recent", {})
    for i, form_type in enumerate(forms.get("form", [])):
        if form_type == "10-K" and len(filings) < limit:
            acc = forms["accessionNumber"][i].replace("-", "")
            filings.append({
                "date":      forms["filingDate"][i],
                "accession": acc,
                "cik":       cik,
            })

    _cached_set(f"filings:{cik}", {"filings": filings})
    return filings


def fetch_10k_text(cik: str, accession: str) -> str:
    """Download the primary 10-K document text from EDGAR."""
    cache_key = f"text:{cik}:{accession}"
    cached = _cached_get(cache_key)
    if cached:
        return cached["text"]

    idx_url = f"{EDGAR_BASE}/Archives/edgar/data/{int(cik)}/{accession}/{accession}-index.json"
    try:
        resp = requests.get(idx_url, headers=EDGAR_HEADERS, timeout=15)
        resp.raise_for_status()
        index = resp.json()
        doc_file = None
        for doc in index.get("documents", []):
            if doc.get("type") == "10-K" or doc.get("documentDescription", "").startswith("10-K"):
                doc_file = doc.get("document")
                break
        if not doc_file:
            doc_file = index["documents"][0]["document"]
        txt_url = f"{EDGAR_BASE}/Archives/edgar/data/{int(cik)}/{accession}/{doc_file}"
        txt_resp = requests.get(txt_url, headers=EDGAR_HEADERS, timeout=30)
        text = txt_resp.text
        _cached_set(cache_key, {"text": text})
        return text
    except Exception as e:
        raise RuntimeError(f"Failed to fetch 10-K text for {cik}/{accession}: {e}")


def extract_risk_factors(full_text: str) -> str:
    """Extract Item 1A Risk Factors section from 10-K text."""
    # Try multiple patterns to be robust across formats
    patterns = [
        r"(?:ITEM\s+1A\.?\s*RISK\s+FACTORS)(.*?)(?:ITEM\s+1B|ITEM\s+2)",
        r"(?:Item\s+1A\.?\s*Risk\s+Factors)(.*?)(?:Item\s+1B|Item\s+2)",
        r"(?:RISK\s+FACTORS)(.*?)(?:UNRESOLVED\s+STAFF\s+COMMENTS|PROPERTIES)",
    ]
    for pat in patterns:
        m = re.search(pat, full_text, re.DOTALL | re.IGNORECASE)
        if m and len(m.group(1).strip()) > 500:
            return m.group(1).strip()
    # Fallback: take a middle chunk if no pattern matched
    mid = len(full_text) // 3
    return full_text[mid: mid + 15000]


# ── Mock data for demo mode ──────────────────────────────────────────────────

def generate_mock_filings(ticker: str) -> dict[str, str]:
    """Generate realistic mock 10-K risk factor text for two years."""

    shared_risks = [
        "We face intense competition in all markets in which we operate and if we fail to compete effectively our revenue and operating results will be adversely affected. Competitive pressures including price competition, new product introductions and expanding operations of existing competitors pose significant risks.",
        "Our financial performance is subject to risks associated with changes in the value of the U.S. dollar versus local currencies. A strengthened dollar reduces the U.S. dollar value of foreign currency denominated revenue.",
        "Failure to comply with laws and regulations applicable to our business could result in fines, penalties and regulatory actions that could harm our business. We are subject to various federal, state and international laws including data protection, privacy and antitrust laws.",
        "Supply chain disruptions, including those caused by natural disasters, public health crises, geopolitical tensions or transportation failures, could affect our ability to source materials and deliver products on time, increasing our costs and adversely impacting revenue.",
        "We depend on key personnel including our senior management and technical staff. The loss of any key personnel or the inability to attract and retain qualified employees could harm our business and operations.",
        "Our business requires continuous innovation and if we fail to introduce new products and services in a timely manner our competitive position may be weakened. Research and development activities are expensive and uncertain in outcome.",
        "We are subject to intellectual property risks including claims by third parties that we infringe their intellectual property rights. Such claims could require us to pay damages or seek licenses which could be costly.",
        "Our operations are subject to various environmental regulations. Compliance with these regulations requires significant expenditures and failure to comply could result in substantial fines and remediation costs.",
        "We rely on third-party vendors and partners for important components of our business. Disruptions in these relationships or the failure of these third parties to perform could adversely affect our operations.",
        "Credit risk from customers and counterparties could adversely affect our financial condition. We extend credit to a large number of customers and cannot guarantee collection of outstanding amounts.",
    ]

    new_2023_risks = [
        "Rapid developments in generative artificial intelligence and large language models pose significant competitive risks. If competitors adopt AI capabilities faster than us or AI disrupts our core business model we could lose substantial market share. The regulatory landscape around AI usage, training data, and model outputs is rapidly evolving and compliance requirements could significantly increase our operating costs.",
        "Cybersecurity threats have increased dramatically in sophistication and frequency. Nation-state sponsored ransomware attacks targeting critical infrastructure and supply chains have become common. A significant data breach or cyberattack could result in unauthorized access to sensitive customer information, regulatory fines under GDPR and CCPA, and substantial reputational damage.",
        "The U.S. government has implemented extensive export controls on advanced semiconductors, AI chips, and related technologies restricting our ability to sell products and services to certain countries including China. These controls could materially reduce our addressable market and require costly restructuring of supply chains.",
        "Tightening credit conditions following rapid interest rate increases by central banks have materially increased our cost of capital. Refinancing existing debt obligations at higher rates could significantly increase interest expense and constrain our ability to fund growth initiatives.",
        "Climate-related physical risks including more frequent extreme weather events have disrupted our manufacturing operations and logistics networks. Transition risks from carbon pricing and mandatory emissions disclosures could substantially increase our operating costs and require material capital expenditures.",
        "Increasing regulatory scrutiny of large technology companies across jurisdictions has resulted in significant antitrust investigations and legislation in the EU, US and Asia. These regulatory actions could require changes to our business practices, product offerings, or result in divestiture of business units.",
    ]

    removed_post_2022 = [
        "The ongoing COVID-19 pandemic continues to create significant uncertainty for our business. Government-mandated lockdowns, travel restrictions, and facility closures have disrupted our manufacturing operations and reduced customer demand across multiple geographies. We cannot predict the duration or severity of pandemic-related disruptions.",
        "Supply chain disruptions specific to pandemic-related semiconductor shortages have severely constrained our ability to manufacture products and fulfill customer orders. We have seen lead times extend from weeks to months for critical components.",
        "Remote work arrangements implemented in response to COVID-19 have increased cybersecurity risks as employees access corporate systems from home networks. We have invested significantly in VPN capacity and endpoint security but cannot guarantee these measures are sufficient.",
        "Government stimulus programs introduced during the pandemic created temporary demand conditions that are unlikely to persist. The normalization of consumer spending patterns following pandemic-era distortions may result in demand headwinds in the near term.",
    ]

    year_old_text = "\n\n".join(shared_risks + removed_post_2022)
    year_new_text = "\n\n".join(shared_risks + new_2023_risks)

    return {"2022": year_old_text, "2023": year_new_text}


def generate_multiyear_mock(ticker: str) -> dict[str, str]:
    """Generate mock filings for 2020-2023 for temporal trend view."""
    base = generate_mock_filings(ticker)
    covid_risks = [
        "The ongoing COVID-19 pandemic continues to create unprecedented disruption to global supply chains, manufacturing operations and consumer demand patterns creating material uncertainty in our financial outlook.",
        "Government-mandated lockdowns and travel restrictions in response to the pandemic have significantly disrupted our sales operations and delayed product launches across key markets.",
    ]
    pre_covid_risks = [
        "Trade policy uncertainty including tariffs imposed on Chinese imports has increased our component costs and created supply chain instability requiring significant operational adjustments.",
        "Brexit-related uncertainty has created significant challenges for our European operations including customs delays, regulatory divergence and workforce mobility issues.",
    ]
    filings = {
        "2020": "\n\n".join(list(generate_mock_filings(ticker).values())[0].split("\n\n")[:6] + covid_risks),
        "2021": "\n\n".join(list(generate_mock_filings(ticker).values())[0].split("\n\n")[:8] + covid_risks[:1]),
        "2022": base["2022"],
        "2023": base["2023"],
    }
    return filings


# ── Embedding & FAISS ────────────────────────────────────────────────────────

class DriftAnalyzer:
    """
    Core drift analysis engine.

    Workflow:
      encode_paragraphs  → Sentence Transformer embeddings (384-dim)
      build_faiss_index  → IndexFlatL2 for ANN retrieval
      compute_drift      → cosine similarity matrix across years
      classify_signals   → NEW / REMOVED / PERSISTED with novelty scores
    """

    def __init__(self):
        self.model = None
        if ST_AVAILABLE:
            try:
                self.model = SentenceTransformer(EMBED_MODEL_NAME)
            except Exception as e:
                print(f"[WARN] Could not load SentenceTransformer model ({e}). Using random embeddings for demo.")

    def _split_paragraphs(self, text: str) -> list[str]:
        paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        return [p for p in paras if len(p.split()) >= MIN_PARA_WORDS]

    def _encode(self, paragraphs: list[str]) -> np.ndarray:
        if self.model:
            vecs = self.model.encode(paragraphs, show_progress_bar=False, normalize_embeddings=True)
        else:
            # hashlib-seeded fallback: each paragraph gets a unique deterministic unit vector.
            # Uses standard_normal (zero-mean) so expected cosine between unrelated paragraphs ≈ 0.
            # (uniform [0,1] vectors give expected cosine ≈ 0.75 after normalizing — wrong for sim.)
            # Same text → same md5 seed → same vector every run (stable across interpreter restarts).
            import hashlib
            vecs = []
            for para in paragraphs:
                digest = hashlib.md5(para[:200].encode("utf-8", errors="ignore")).digest()
                seed   = int.from_bytes(digest[:4], "little")
                rng    = np.random.default_rng(seed)
                vec    = rng.standard_normal(384).astype("float32")
                vec   /= np.linalg.norm(vec) + 1e-9
                vecs.append(vec)
            vecs = np.array(vecs, dtype="float32")
        return vecs.astype("float32")

    def _build_faiss_index(self, vecs: np.ndarray) -> object:
        dim = vecs.shape[1]
        if FAISS_AVAILABLE:
            index = faiss.IndexFlatL2(dim)
            index.add(vecs)
        else:
            index = {"vecs": vecs}   # numpy fallback
        return index

    def _search(self, index, query_vec: np.ndarray, k: int = 1):
        """Return (distances, indices) for top-k neighbours."""
        if FAISS_AVAILABLE:
            D, I = index.search(query_vec.reshape(1, -1), k)
            return D[0], I[0]
        else:
            vecs = index["vecs"]
            # cosine similarity (vectors already L2-normalized)
            sims = vecs @ query_vec.flatten()
            idx  = np.argsort(-sims)[:k]
            return 1 - sims[idx], idx   # return L2-like distances

    def analyze_drift(
        self,
        text_old: str,
        text_new: str,
        year_old: str,
        year_new: str,
    ) -> dict:
        """
        Full drift analysis between two filing years.

        Returns a dict with:
          - new_signals:       paragraphs in new year with no close match in old
          - removed_signals:   paragraphs in old year with no close match in new
          - persisted_signals: paragraphs present in both
          - drift_score:       float 0-1 aggregate novelty
          - similarity_matrix: 2D numpy array (new × old) for heatmap
          - metadata:          run timestamps, counts, year labels
          - evaluation_log:    structured evaluation output for export
        """
        paras_old = self._split_paragraphs(text_old)
        paras_new = self._split_paragraphs(text_new)

        vecs_old  = self._encode(paras_old)
        vecs_new  = self._encode(paras_new)

        index_old = self._build_faiss_index(vecs_old)
        index_new = self._build_faiss_index(vecs_new)

        # ── Classify new-year paragraphs ──────────────────────────────────
        new_signals       = []
        persisted_signals = []

        for i, (para, vec) in enumerate(zip(paras_new, vecs_new)):
            _, I = self._search(index_old, vec, k=1)
            best_idx = I[0]
            sim = float(np.dot(vec, vecs_old[best_idx]))  # cosine (normalized)
            category = classify_risk_category(para)

            if sim < SIMILARITY_THRESHOLD:
                new_signals.append({
                    "paragraph":     para,
                    "max_sim_to_old": round(sim, 4),
                    "novelty_score":  round(1 - sim, 4),
                    "closest_old_idx": int(best_idx),
                    "closest_old_para": paras_old[best_idx][:200] + "…",
                    "risk_category":  category,
                    "year":           year_new,
                    "provenance":     f"Filing year {year_new}, paragraph index {i}",
                })
            else:
                persisted_signals.append({
                    "paragraph":    para,
                    "similarity":   round(sim, 4),
                    "risk_category": category,
                    "year":         year_new,
                })

        # ── Classify old-year paragraphs (removed signals) ────────────────
        removed_signals = []
        for i, (para, vec) in enumerate(zip(paras_old, vecs_old)):
            _, I = self._search(index_new, vec, k=1)
            best_idx = I[0]
            sim = float(np.dot(vec, vecs_new[best_idx]))
            category = classify_risk_category(para)

            if sim < SIMILARITY_THRESHOLD:
                removed_signals.append({
                    "paragraph":     para,
                    "max_sim_to_new": round(sim, 4),
                    "risk_category":  category,
                    "year":           year_old,
                    "provenance":     f"Filing year {year_old}, paragraph index {i}",
                })

        # ── Similarity matrix (new × old, capped at 50×50 for heatmap) ───
        n_new = min(len(vecs_new), 50)
        n_old = min(len(vecs_old), 50)
        sim_matrix = (vecs_new[:n_new] @ vecs_old[:n_old].T).tolist()

        # ── Aggregate drift score ─────────────────────────────────────────
        n_total = len(paras_new)
        drift_score = round(len(new_signals) / n_total, 4) if n_total else 0.0

        # ── Evaluation log ────────────────────────────────────────────────
        eval_log = {
            "run_timestamp":       datetime.utcnow().isoformat() + "Z",
            "model":               EMBED_MODEL_NAME,
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "year_old":            year_old,
            "year_new":            year_new,
            "paragraphs_old":      len(paras_old),
            "paragraphs_new":      len(paras_new),
            "new_signals":         len(new_signals),
            "removed_signals":     len(removed_signals),
            "persisted_signals":   len(persisted_signals),
            "drift_score":         drift_score,
            "category_breakdown_new": _category_counts(new_signals),
            "category_breakdown_removed": _category_counts(removed_signals),
        }

        return {
            "new_signals":       new_signals,
            "removed_signals":   removed_signals,
            "persisted_signals": persisted_signals,
            "drift_score":       drift_score,
            "similarity_matrix": sim_matrix,
            "year_old":          year_old,
            "year_new":          year_new,
            "paras_old":         paras_old,
            "paras_new":         paras_new,
            "evaluation_log":    eval_log,
            "gemini_summary":    None,  # filled in by summarize_with_gemini
        }


def _category_counts(signals: list[dict]) -> dict:
    counts = {}
    for s in signals:
        cat = s.get("risk_category", "General")
        counts[cat] = counts.get(cat, 0) + 1
    return counts


# ── Gemini integration ───────────────────────────────────────────────────────

def summarize_with_gemini(new_signals: list[dict], company: str) -> str:
    if not GEMINI_AVAILABLE or not GEMINI_API_KEY:
        return "_Gemini API key not configured. Set the `GEMINI_API_KEY` environment variable to enable AI summaries._"

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")

    snippets = "\n\n".join(
        [f"[{s['risk_category']}] {s['paragraph'][:300]}" for s in new_signals[:6]]
    )
    prompt = f"""
You are a senior financial risk analyst reviewing SEC 10-K disclosures.

The following are newly emerged risk signals in {company}'s most recent 10-K filing that 
did NOT appear in the prior year's filing. Each is prefixed with its risk category.

{snippets}

Tasks:
1. In 3-4 concise bullet points, summarize the key new risk themes.
2. Identify which risk category is most prominent.
3. Note any macro trends (AI regulation, cybersecurity escalation, geopolitical shifts) 
   these signals reflect.
4. Give one sentence on the potential business impact.

Keep your response concise and analyst-grade. No preamble.
"""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"_Gemini API error: {e}_"


def classify_signals_with_gemini(signals: list[dict], company: str) -> list[dict]:
    """Use Gemini to refine risk category classification for ambiguous signals."""
    if not GEMINI_AVAILABLE or not GEMINI_API_KEY or not signals:
        return signals

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Only reclassify signals that fall into "General"
    general_indices = [i for i, s in enumerate(signals) if s.get("risk_category") == "General"]
    if not general_indices:
        return signals

    for i in general_indices[:5]:   # limit API calls
        para = signals[i]["paragraph"][:400]
        prompt = f"""
Classify the following SEC 10-K risk factor paragraph into exactly ONE of these categories:
Cybersecurity, Regulatory, Financial, Operational, Market, AI / Technology, Geopolitical, Climate, General

Paragraph: {para}

Reply with ONLY the category name, nothing else.
"""
        try:
            resp = model.generate_content(prompt)
            cat  = resp.text.strip()
            if cat in RISK_KEYWORDS or cat == "General":
                signals[i]["risk_category"] = cat
            time.sleep(0.5)
        except Exception:
            pass

    return signals


# ── Multi-year trend analysis ────────────────────────────────────────────────

def compute_temporal_trend(ticker: str, use_real: bool = False) -> list[dict]:
    """
    Compute drift scores across multiple consecutive years.
    Returns list of {year_pair, drift_score} dicts for trend chart.
    """
    analyzer = DriftAnalyzer()

    if use_real:
        try:
            cik     = get_company_cik(ticker)
            filings = get_10k_filings(cik, limit=5)
            texts   = {}
            for f in filings[:4]:
                text = fetch_10k_text(f["cik"], f["accession"])
                texts[f["date"][:4]] = extract_risk_factors(text)
                time.sleep(1.2)
        except Exception:
            texts = generate_multiyear_mock(ticker)
    else:
        texts = generate_multiyear_mock(ticker)

    years  = sorted(texts.keys())
    trend  = []
    for i in range(len(years) - 1):
        y_old, y_new = years[i], years[i + 1]
        result = analyzer.analyze_drift(texts[y_old], texts[y_new], y_old, y_new)
        trend.append({
            "year_pair":   f"{y_old}→{y_new}",
            "year":        y_new,
            "drift_score": result["drift_score"],
            "new_count":   len(result["new_signals"]),
            "removed_count": len(result["removed_signals"]),
        })
    return trend


# ── Standalone runner ────────────────────────────────────────────────────────

def run_analysis(ticker: str = "AAPL", use_real_data: bool = False) -> tuple[dict, dict]:
    """
    Run full drift analysis for a company.
    Returns (result_dict, texts_dict).
    """
    print("=" * 60)
    print(f"  SEC RISK SIGNAL DRIFT ANALYZER — {ticker}")
    print("=" * 60)

    analyzer = DriftAnalyzer()

    if use_real_data:
        try:
            cik      = get_company_cik(ticker)
            filings  = get_10k_filings(cik)
            texts    = {}
            for f in filings[:2]:
                print(f"[FETCH] {f['date']} 10-K ...")
                text = fetch_10k_text(f["cik"], f["accession"])
                texts[f["date"][:4]] = extract_risk_factors(text)
                time.sleep(1.2)
        except Exception as e:
            print(f"[WARN] EDGAR API failed ({e}). Using mock data.")
            texts = generate_mock_filings(ticker)
    else:
        texts = generate_mock_filings(ticker)

    years  = sorted(texts.keys())
    result = analyzer.analyze_drift(texts[years[0]], texts[years[1]], years[0], years[1])

    # Gemini summary
    if result["new_signals"]:
        result["gemini_summary"] = summarize_with_gemini(result["new_signals"], ticker)

    # Save evaluation log
    log_path = Path("sec_cache") / f"eval_{ticker}_{years[0]}_{years[1]}.json"
    with open(log_path, "w") as f:
        json.dump(result["evaluation_log"], f, indent=2)
    print(f"[LOG] Evaluation log saved → {log_path}")

    return result, texts


if __name__ == "__main__":
    result, texts = run_analysis("AAPL", use_real_data=False)
    print(f"\n  Drift Score : {result['drift_score']:.2%}")
    print(f"  New Signals : {len(result['new_signals'])}")
    print(f"  Removed     : {len(result['removed_signals'])}")
    print(f"  Persisted   : {len(result['persisted_signals'])}")
    print("\n[NEW SIGNALS SAMPLE]")
    for s in result["new_signals"][:3]:
        print(f"  [{s['risk_category']}] [{s['novelty_score']:.3f} novelty] {s['paragraph'][:120]}…")
