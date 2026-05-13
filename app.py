"""
app.py
------
SEC Risk Signal Drift Analyzer — Streamlit Dashboard

Views:
  1. 🏠 Overview          — System description, KPIs, pipeline diagram
  2. 📊 Drift Analysis    — Signal flow, category breakdown, Gemini summary
  3. 🆕 New Risk Signals  — Para-level new signals, novelty scores, search, CSV export
  4. ❌ Removed Signals   — Para-level removed signals, CSV export
  5. 🔥 Similarity Heatmap— FAISS-powered paragraph similarity matrix
  6. 🏢 Cross-Company     — Drift score benchmarking across companies
  7. 📈 Temporal Trend    — Multi-year drift trend with macro-event annotations
  8. 📋 Evaluation Log    — Structured output logs, JSON export

Author: Anisha Tiwary | KIIT University 2026
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import streamlit as st

warnings.filterwarnings("ignore")

from sec_analyzer import (
    run_analysis,
    compute_temporal_trend,
    DriftAnalyzer,
    generate_mock_filings,
    summarize_with_gemini,
    SIMILARITY_THRESHOLD,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SEC Risk Signal Drift Analyzer",
    page_icon="📑",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .new-signal  {
        background: #d4edda; border-left: 4px solid #28a745;
        padding: 0.65rem 0.9rem; margin: 0.35rem 0;
        border-radius: 6px; font-size: 13px; line-height: 1.5;
    }
    .gone-signal {
        background: #f8d7da; border-left: 4px solid #dc3545;
        padding: 0.65rem 0.9rem; margin: 0.35rem 0;
        border-radius: 6px; font-size: 13px; line-height: 1.5;
    }
    .persist-sig {
        background: #e9ecef; border-left: 4px solid #6c757d;
        padding: 0.65rem 0.9rem; margin: 0.35rem 0;
        border-radius: 6px; font-size: 13px; line-height: 1.5;
    }
    .header-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
        color: white; padding: 1.2rem 1.5rem; border-radius: 12px;
        margin-bottom: 1rem;
    }
    .kpi-card {
        background: #f8f9fa; border: 1px solid #dee2e6;
        border-radius: 10px; padding: 1rem; text-align: center;
    }
    .kpi-value { font-size: 2rem; font-weight: 700; color: #0f3460; }
    .kpi-label { font-size: 0.8rem; color: #6c757d; margin-top: 4px; }
    .cat-badge {
        display: inline-block; padding: 2px 8px; border-radius: 12px;
        font-size: 11px; font-weight: 600; margin-right: 4px;
        background: #e2e3e5; color: #333;
    }
    .search-highlight { background: #fff3cd; border-radius: 3px; padding: 1px 3px; }
</style>
""", unsafe_allow_html=True)

# ── Category color map ────────────────────────────────────────────────────────
CAT_COLORS = {
    "Cybersecurity":   "#e74c3c",
    "Regulatory":      "#8e44ad",
    "Financial":       "#2980b9",
    "Operational":     "#27ae60",
    "Market":          "#f39c12",
    "AI / Technology": "#16a085",
    "Geopolitical":    "#d35400",
    "Climate":         "#1abc9c",
    "General":         "#95a5a6",
}

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/document-scanner.png", width=60)
    st.title("SEC Risk Drift Analyzer")
    st.markdown("---")

    view = st.radio("Navigate", [
        "🏠 Overview",
        "📊 Drift Analysis",
        "🆕 New Risk Signals",
        "❌ Removed Signals",
        "🔥 Similarity Heatmap",
        "🏢 Cross-Company Benchmark",
        "📈 Temporal Trend",
        "📋 Evaluation Log",
    ])

    st.markdown("---")
    st.subheader("⚙️ Configuration")
    ticker    = st.text_input("Company Ticker", value="AAPL").upper().strip()
    year_from = st.selectbox("From Year", ["2022", "2021", "2020"], index=0)
    year_to   = st.selectbox("To Year",   ["2023", "2022", "2021"], index=0)
    use_real  = st.checkbox("Use SEC EDGAR API (live data)", value=False)
    st.caption("⚠️ Uncheck for demo mode with realistic mock data. Live EDGAR requires internet.")

    run_btn = st.button("▶ Run Analysis", type="primary", use_container_width=True)
    st.markdown("---")
    st.markdown("""
**Stack**  
`Sentence Transformers` · `FAISS` · `Gemini API` · `SEC EDGAR API` · `Streamlit`

**Author**  
Anisha Tiwary  
KIIT University 2026
""")

# ── Session state cache ───────────────────────────────────────────────────────
STATE_KEY = f"result_{ticker}_{year_from}_{year_to}_{use_real}"

if run_btn or STATE_KEY not in st.session_state:
    with st.spinner("Running drift analysis... encoding paragraphs with Sentence Transformers..."):
        try:
            result, texts = run_analysis(ticker, use_real_data=use_real)
            st.session_state[STATE_KEY] = (result, texts)
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.stop()

result, texts = st.session_state[STATE_KEY]

new_sigs       = result["new_signals"]
removed_sigs   = result["removed_signals"]
persisted_sigs = result["persisted_signals"]
drift_score    = result["drift_score"]
sim_matrix     = np.array(result["similarity_matrix"])
eval_log       = result["evaluation_log"]
year_old       = result["year_old"]
year_new       = result["year_new"]

# ─────────────────────────────────────────────────────────────────────────────
# VIEW 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if view == "🏠 Overview":
    st.markdown("""
<div class="header-box">
  <h2 style="margin:0; font-size:1.5rem;">📑 SEC Risk Signal Drift Analyzer</h2>
  <p style="margin:0.3rem 0 0 0; opacity:0.85; font-size:14px;">
    Detecting year-over-year semantic drift in Fortune 500 10-K risk disclosures 
    using Sentence Transformers, FAISS vector retrieval, and Gemini AI.
  </p>
</div>
""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label, color in [
        (c1, f"{drift_score:.1%}",      "Drift Score",         "#dc3545"),
        (c2, len(new_sigs),             "New Signals",         "#28a745"),
        (c3, len(removed_sigs),         "Removed Signals",     "#fd7e14"),
        (c4, len(persisted_sigs),       "Persisted Signals",   "#6c757d"),
    ]:
        col.markdown(f"""
<div class="kpi-card">
  <div class="kpi-value" style="color:{color};">{val}</div>
  <div class="kpi-label">{label}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("")
    st.markdown(f"**Analysis:** `{ticker}` · `{year_old}` → `{year_new}` · Threshold: `{SIMILARITY_THRESHOLD}`")

    st.markdown("---")
    st.subheader("🔬 Technical Pipeline")
    pipeline_steps = [
        ("1", "SEC EDGAR API", "Automated 10-K filing ingestion by ticker → CIK resolution → Item 1A extraction"),
        ("2", "Paragraph Splitting", "Segment risk factor text into meaningful paragraph units (min 20 words)"),
        ("3", "Sentence Transformers", "Encode paragraphs with all-MiniLM-L6-v2 (384-dim normalized embeddings)"),
        ("4", "FAISS Index", "Build IndexFlatL2 for fast approximate nearest-neighbor retrieval"),
        ("5", "Cosine Similarity", "Compute similarity for each paragraph across filing years"),
        ("6", "Drift Classification", f"Flag NEW (sim < {SIMILARITY_THRESHOLD}) · REMOVED · PERSISTED signals"),
        ("7", "Risk Categorization", "Auto-classify signals: Cybersecurity, Regulatory, Financial, AI/Technology…"),
        ("8", "Gemini Summary", "AI-powered synthesis of newly emerged risk themes"),
        ("9", "Evaluation Logs", "Export structured JSON logs for reproducibility and audit trails"),
    ]
    for num, title, desc in pipeline_steps:
        st.markdown(f"**`{num}.`** **{title}** — {desc}")

    st.markdown("---")
    st.subheader("📂 Project Structure")
    st.code("""
sec-risk-drift/
├── sec_analyzer.py      # Core engine: EDGAR fetch, FAISS, drift analysis, Gemini
├── app.py               # Streamlit dashboard (8 views)
├── requirements.txt     # Python dependencies
├── README.md            # Full documentation
├── TESTING.md           # Test cases and evaluation scenarios
├── .github/
│   └── workflows/
│       └── ci.yml       # GitHub Actions CI
└── sec_cache/           # Auto-created: EDGAR response cache + eval logs
""", language="")


# ─────────────────────────────────────────────────────────────────────────────
# VIEW 2 — DRIFT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
elif view == "📊 Drift Analysis":
    st.header(f"📊 Drift Analysis — {ticker} ({year_old} → {year_new})")

    # Signal flow bar chart
    st.subheader("Signal Flow Overview")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    categories = ["New Signals", "Removed Signals", "Persisted Signals"]
    values     = [len(new_sigs), len(removed_sigs), len(persisted_sigs)]
    colors_bar = ["#28a745",     "#dc3545",          "#6c757d"]
    bars = ax.barh(categories, values, color=colors_bar, height=0.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                str(v), va="center", fontsize=11, fontweight="bold")
    ax.set_xlabel("Number of Paragraphs")
    ax.set_title(f"{ticker}: Risk Signal Classification ({year_old} → {year_new})")
    ax.set_xlim(0, max(values) * 1.2 + 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Risk category breakdown (new signals)
    if new_sigs:
        st.subheader("Risk Category Breakdown (New Signals)")
        cat_counts = {}
        for s in new_sigs:
            cat = s.get("risk_category", "General")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        fig2, ax2 = plt.subplots(figsize=(7, 3.5))
        sorted_cats = sorted(cat_counts.items(), key=lambda x: -x[1])
        cats   = [c[0] for c in sorted_cats]
        counts = [c[1] for c in sorted_cats]
        bar_colors = [CAT_COLORS.get(c, "#95a5a6") for c in cats]
        ax2.bar(cats, counts, color=bar_colors, edgecolor="white", linewidth=1.5)
        ax2.set_ylabel("Count")
        ax2.set_title("New Risk Signal Categories")
        ax2.tick_params(axis="x", rotation=30)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # Gemini AI summary
    st.subheader("🤖 Gemini AI Risk Summary")
    gemini_summary = result.get("gemini_summary")
    if not gemini_summary:
        gemini_summary = summarize_with_gemini(new_sigs, ticker)
        result["gemini_summary"] = gemini_summary
        st.session_state[STATE_KEY] = (result, texts)
    st.info(gemini_summary)

    # Drift score gauge
    st.subheader("Aggregate Drift Score")
    col1, col2 = st.columns([1, 2])
    with col1:
        level = "🔴 High" if drift_score > 0.4 else ("🟡 Moderate" if drift_score > 0.2 else "🟢 Low")
        st.metric("Drift Score", f"{drift_score:.1%}", help="Fraction of new-year paragraphs with no close match in prior year")
        st.markdown(f"**Risk Level:** {level}")
    with col2:
        fig3, ax3 = plt.subplots(figsize=(5, 1.5))
        ax3.barh([""], [drift_score], color="#dc3545" if drift_score > 0.4 else ("#f39c12" if drift_score > 0.2 else "#27ae60"))
        ax3.barh([""], [1 - drift_score], left=[drift_score], color="#e9ecef")
        ax3.set_xlim(0, 1)
        ax3.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax3.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)
        ax3.spines["left"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# VIEW 3 — NEW RISK SIGNALS
# ─────────────────────────────────────────────────────────────────────────────
elif view == "🆕 New Risk Signals":
    st.header(f"🆕 Newly Emerged Risk Signals — {year_new}")
    st.caption(f"Paragraphs in {year_new} filing with cosine similarity < {SIMILARITY_THRESHOLD} to any paragraph in {year_old} filing.")

    if not new_sigs:
        st.success("No new risk signals detected. Disclosures are highly similar to prior year.")
        st.stop()

    # Controls row
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        search_q = st.text_input("🔍 Search signals", placeholder="e.g. AI, cybersecurity, climate…")
    with col2:
        cat_filter = st.multiselect(
            "Filter by category",
            options=sorted(set(s.get("risk_category", "General") for s in new_sigs)),
            default=[]
        )
    with col3:
        sort_by = st.selectbox("Sort by", ["Novelty Score ↓", "Risk Category", "Default"])

    # Filter
    filtered = new_sigs
    if search_q:
        filtered = [s for s in filtered if search_q.lower() in s["paragraph"].lower()]
    if cat_filter:
        filtered = [s for s in filtered if s.get("risk_category") in cat_filter]
    if sort_by == "Novelty Score ↓":
        filtered = sorted(filtered, key=lambda x: -x["novelty_score"])
    elif sort_by == "Risk Category":
        filtered = sorted(filtered, key=lambda x: x.get("risk_category", ""))

    st.markdown(f"Showing **{len(filtered)}** of **{len(new_sigs)}** new signals")

    # Novelty score distribution
    if len(filtered) > 1:
        with st.expander("📊 Novelty Score Distribution"):
            fig_hist, ax_hist = plt.subplots(figsize=(7, 2.5))
            novelty_scores = [s["novelty_score"] for s in filtered]
            ax_hist.hist(novelty_scores, bins=15, color="#28a745", edgecolor="white", alpha=0.85)
            ax_hist.set_xlabel("Novelty Score (1 - cosine similarity)")
            ax_hist.set_ylabel("Count")
            ax_hist.set_title("Distribution of Novelty Scores — New Signals")
            ax_hist.spines["top"].set_visible(False)
            ax_hist.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_hist)
            plt.close()

    # Signal cards
    for i, sig in enumerate(filtered):
        cat   = sig.get("risk_category", "General")
        score = sig.get("novelty_score", 0)
        color = CAT_COLORS.get(cat, "#95a5a6")
        text  = sig["paragraph"]
        if search_q:
            text = text.replace(search_q, f"**{search_q}**")
        with st.expander(f"[{cat}] Signal {i + 1} — Novelty {score:.3f} · {sig['provenance']}"):
            st.markdown(f'<div class="new-signal">{text}</div>', unsafe_allow_html=True)
            st.markdown(f"""
- **Category:** `{cat}`
- **Novelty Score:** `{score:.4f}` (higher = more novel)
- **Closest Old Paragraph (sim={sig['max_sim_to_old']:.3f}):** _{sig.get('closest_old_para', 'N/A')}_
- **Provenance:** {sig['provenance']}
""")

    # CSV download
    df_new = pd.DataFrame([{
        "Risk Category":  s.get("risk_category", "General"),
        "Novelty Score":  s["novelty_score"],
        "Cosine Sim":     s["max_sim_to_old"],
        "Paragraph":      s["paragraph"],
        "Provenance":     s["provenance"],
    } for s in filtered])
    st.download_button(
        "⬇ Download New Signals as CSV",
        data=df_new.to_csv(index=False),
        file_name=f"{ticker}_new_signals_{year_new}.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────────────────────────────────────
# VIEW 4 — REMOVED SIGNALS
# ─────────────────────────────────────────────────────────────────────────────
elif view == "❌ Removed Signals":
    st.header(f"❌ Removed Risk Signals — Dropped from {year_new}")
    st.caption(f"Paragraphs present in {year_old} filing with no close match in {year_new} filing.")

    if not removed_sigs:
        st.info("No removed signals detected. All prior-year disclosures persist in the new filing.")
        st.stop()

    search_r = st.text_input("🔍 Search removed signals")
    cat_filter_r = st.multiselect(
        "Filter by category",
        options=sorted(set(s.get("risk_category", "General") for s in removed_sigs)),
        default=[]
    )

    filtered_r = removed_sigs
    if search_r:
        filtered_r = [s for s in filtered_r if search_r.lower() in s["paragraph"].lower()]
    if cat_filter_r:
        filtered_r = [s for s in filtered_r if s.get("risk_category") in cat_filter_r]

    st.markdown(f"Showing **{len(filtered_r)}** of **{len(removed_sigs)}** removed signals")

    for i, sig in enumerate(filtered_r):
        cat = sig.get("risk_category", "General")
        sim = sig.get("max_sim_to_new", 0)
        with st.expander(f"[{cat}] Removed Signal {i + 1} — Max sim to new: {sim:.3f} · {sig['provenance']}"):
            st.markdown(f'<div class="gone-signal">{sig["paragraph"]}</div>', unsafe_allow_html=True)
            st.markdown(f"""
- **Category:** `{cat}`
- **Best match in new filing (sim={sim:.3f}):** Below threshold — not carried forward
- **Provenance:** {sig['provenance']}
""")

    df_removed = pd.DataFrame([{
        "Risk Category":    s.get("risk_category", "General"),
        "Max Sim to New":   s["max_sim_to_new"],
        "Paragraph":        s["paragraph"],
        "Provenance":       s["provenance"],
    } for s in filtered_r])
    st.download_button(
        "⬇ Download Removed Signals as CSV",
        data=df_removed.to_csv(index=False),
        file_name=f"{ticker}_removed_signals_{year_old}.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────────────────────────────────────
# VIEW 5 — SIMILARITY HEATMAP
# ─────────────────────────────────────────────────────────────────────────────
elif view == "🔥 Similarity Heatmap":
    st.header("🔥 FAISS-Powered Paragraph Similarity Heatmap")
    st.caption(f"Cosine similarity between {year_new} paragraphs (rows) and {year_old} paragraphs (columns). "
               f"FAISS IndexFlatL2 · all-MiniLM-L6-v2 · 384-dim embeddings")

    if sim_matrix.size == 0:
        st.warning("Similarity matrix not available.")
        st.stop()

    # Cap for readability
    n_new = min(sim_matrix.shape[0], 30)
    n_old = min(sim_matrix.shape[1], 30)
    mat   = sim_matrix[:n_new, :n_old]

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        mat, ax=ax,
        cmap="RdYlGn",
        vmin=0, vmax=1,
        linewidths=0.3,
        linecolor="#dee2e6",
        cbar_kws={"label": "Cosine Similarity", "shrink": 0.8},
        xticklabels=[f"Old {i}" for i in range(n_old)],
        yticklabels=[f"New {i}" for i in range(n_new)],
    )
    ax.set_xlabel(f"{year_old} Filing Paragraphs", fontsize=11)
    ax.set_ylabel(f"{year_new} Filing Paragraphs", fontsize=11)
    ax.set_title(f"{ticker} — Paragraph-Level Semantic Similarity Matrix", fontsize=13, fontweight="bold")
    plt.xticks(fontsize=7, rotation=45)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown(f"""
**How to read this:**  
- **Dark green** (sim ≥ 0.85) → paragraph persisted nearly verbatim  
- **Yellow** (sim ~0.5–0.7) → topic similar but language changed  
- **Red** (sim < 0.35) → paragraph is novel or entirely removed  
- Threshold for NEW classification: **{SIMILARITY_THRESHOLD}**
""")

    # Distribution of max similarities
    st.subheader("Max Similarity Distribution (per new paragraph)")
    max_sims = mat.max(axis=1)
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    ax2.hist(max_sims, bins=20, color="#0f3460", edgecolor="white", alpha=0.85)
    ax2.axvline(SIMILARITY_THRESHOLD, color="#dc3545", linestyle="--", linewidth=1.5,
                label=f"Threshold = {SIMILARITY_THRESHOLD}")
    ax2.set_xlabel("Max Cosine Similarity to Prior Year")
    ax2.set_ylabel("Count")
    ax2.set_title("How similar is each new paragraph to the best match in prior filing?")
    ax2.legend()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# VIEW 6 — CROSS-COMPANY BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────
elif view == "🏢 Cross-Company Benchmark":
    st.header("🏢 Cross-Company Risk Disclosure Benchmarking")
    st.caption("Compare drift scores across multiple companies using mock data. Swap to EDGAR API for real data.")

    COMPANIES = {
        "AAPL":  {"sector": "Technology",    "drift": 0.38, "new": 6,  "removed": 4},
        "MSFT":  {"sector": "Technology",    "drift": 0.42, "new": 7,  "removed": 3},
        "AMZN":  {"sector": "E-Commerce",    "drift": 0.51, "new": 9,  "removed": 5},
        "GOOGL": {"sector": "Technology",    "drift": 0.35, "new": 5,  "removed": 4},
        "META":  {"sector": "Social Media",  "drift": 0.58, "new": 11, "removed": 7},
        "JPM":   {"sector": "Banking",       "drift": 0.29, "new": 4,  "removed": 3},
        "GS":    {"sector": "Banking",       "drift": 0.33, "new": 5,  "removed": 2},
        "TSLA":  {"sector": "EV / Auto",     "drift": 0.62, "new": 13, "removed": 8},
        "JNJ":   {"sector": "Healthcare",    "drift": 0.24, "new": 3,  "removed": 2},
        "XOM":   {"sector": "Energy",        "drift": 0.44, "new": 8,  "removed": 6},
    }
    # Overwrite with actual result for the selected ticker
    if ticker in COMPANIES:
        COMPANIES[ticker]["drift"]   = drift_score
        COMPANIES[ticker]["new"]     = len(new_sigs)
        COMPANIES[ticker]["removed"] = len(removed_sigs)

    df_bench = pd.DataFrame([
        {"Ticker": t, "Sector": v["sector"], "Drift Score": v["drift"],
         "New Signals": v["new"], "Removed Signals": v["removed"]}
        for t, v in COMPANIES.items()
    ]).sort_values("Drift Score", ascending=False)

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    bar_colors = ["#dc3545" if t == ticker else "#0f3460" for t in df_bench["Ticker"]]
    bars = ax.bar(df_bench["Ticker"], df_bench["Drift Score"], color=bar_colors, edgecolor="white", linewidth=1.2)
    ax.axhline(df_bench["Drift Score"].mean(), color="#f39c12", linestyle="--", linewidth=1.5,
               label=f"Avg = {df_bench['Drift Score'].mean():.1%}")
    for bar, val in zip(bars, df_bench["Drift Score"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.0%}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("Drift Score (% new paragraphs)")
    ax.set_title("Year-over-Year Risk Disclosure Drift Score — Fortune 500 Sample")
    ax.set_ylim(0, df_bench["Drift Score"].max() + 0.12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown(f"**{ticker}** (highlighted in red) · Sector: {COMPANIES.get(ticker, {}).get('sector', 'N/A')}")

    # Data table
    st.subheader("Benchmark Table")
    st.dataframe(
        df_bench.style
            .background_gradient(subset=["Drift Score"], cmap="RdYlGn_r")
            .format({"Drift Score": "{:.1%}"}),
        use_container_width=True,
    )

    # Sector comparison
    st.subheader("Average Drift by Sector")
    sector_avg = df_bench.groupby("Sector")["Drift Score"].mean().sort_values(ascending=False)
    fig2, ax2 = plt.subplots(figsize=(7, 3.5))
    ax2.barh(sector_avg.index, sector_avg.values,
             color=[CAT_COLORS.get("Market", "#2980b9")] * len(sector_avg))
    ax2.set_xlabel("Average Drift Score")
    ax2.set_title("Average Drift Score by Sector")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# VIEW 7 — TEMPORAL TREND
# ─────────────────────────────────────────────────────────────────────────────
elif view == "📈 Temporal Trend":
    st.header("📈 Multi-Year Risk Disclosure Drift Trend")
    st.caption("Year-over-year drift scores with major macro-event annotations.")

    with st.spinner("Computing multi-year trend..."):
        trend_key = f"trend_{ticker}_{use_real}"
        if trend_key not in st.session_state:
            trend = compute_temporal_trend(ticker, use_real=use_real)
            st.session_state[trend_key] = trend
        trend = st.session_state[trend_key]

    if not trend:
        st.warning("Not enough filing years to compute trend.")
        st.stop()

    years_t  = [t["year"] for t in trend]
    scores_t = [t["drift_score"] for t in trend]
    new_cnts = [t["new_count"] for t in trend]
    rem_cnts = [t["removed_count"] for t in trend]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(years_t, scores_t, marker="o", linewidth=2.5, color="#0f3460",
            markersize=8, markerfacecolor="white", markeredgewidth=2, label="Drift Score")
    ax.fill_between(years_t, scores_t, alpha=0.12, color="#0f3460")

    # Macro-event annotations
    macro_events = {
        "2021": ("COVID-19\npandemic risks", 0.08),
        "2023": ("AI regulation +\ncybersecurity escalation", 0.08),
    }
    for y_annot, (label, offset) in macro_events.items():
        if y_annot in years_t:
            y_val = scores_t[years_t.index(y_annot)]
            ax.annotate(
                label,
                xy=(y_annot, y_val),
                xytext=(y_annot, y_val + offset + 0.04),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.2),
                fontsize=8.5, color="#555", ha="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ccc", alpha=0.9),
            )

    ax.set_xlabel("Year")
    ax.set_ylabel("Annual Drift Score")
    ax.set_title(f"{ticker} — Year-over-Year Risk Disclosure Drift Score")
    ax.set_ylim(0, max(scores_t) + 0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # New vs Removed counts bar chart
    st.subheader("New vs Removed Signal Counts per Year")
    x       = np.arange(len(years_t))
    width   = 0.35
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    ax2.bar(x - width / 2, new_cnts, width, label="New Signals",     color="#28a745", alpha=0.85)
    ax2.bar(x + width / 2, rem_cnts, width, label="Removed Signals", color="#dc3545", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels([t["year_pair"] for t in trend], rotation=10)
    ax2.set_ylabel("Paragraph Count")
    ax2.set_title("New vs Removed Risk Paragraphs by Year Pair")
    ax2.legend()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    # Table
    df_trend = pd.DataFrame(trend)
    df_trend.columns = ["Year Pair", "Year", "Drift Score", "New Signals", "Removed Signals"]
    df_trend["Drift Score"] = df_trend["Drift Score"].map(lambda x: f"{x:.1%}")
    st.dataframe(df_trend.set_index("Year Pair"), use_container_width=True)

    st.markdown("""
**Interpretation:**  
Higher drift scores correspond to major macro events forcing disclosure of new risk categories:
- **2021:** COVID-19 supply chain and operational risk escalation  
- **2023:** AI governance, LLM regulation, geopolitical export controls, and cybersecurity escalation
""")


# ─────────────────────────────────────────────────────────────────────────────
# VIEW 8 — EVALUATION LOG
# ─────────────────────────────────────────────────────────────────────────────
elif view == "📋 Evaluation Log":
    st.header("📋 Evaluation Output Log")
    st.caption("Structured, reproducible evaluation metadata for every analysis run. Exportable as JSON.")

    st.json(eval_log)

    # JSON download
    st.download_button(
        "⬇ Download Evaluation Log as JSON",
        data=json.dumps(eval_log, indent=2),
        file_name=f"{ticker}_eval_log_{year_old}_{year_new}.json",
        mime="application/json",
    )

    # Summary table
    st.subheader("Run Summary")
    summary_data = {
        "Field": [
            "Run Timestamp (UTC)", "Model", "Similarity Threshold",
            "Filing Period", "Paragraphs (Old)", "Paragraphs (New)",
            "New Signals", "Removed Signals", "Persisted Signals", "Drift Score",
        ],
        "Value": [
            eval_log.get("run_timestamp", "N/A"),
            eval_log.get("model", "N/A"),
            str(eval_log.get("similarity_threshold", "N/A")),
            f"{year_old} → {year_new}",
            str(eval_log.get("paragraphs_old", "N/A")),
            str(eval_log.get("paragraphs_new", "N/A")),
            str(eval_log.get("new_signals", "N/A")),
            str(eval_log.get("removed_signals", "N/A")),
            str(eval_log.get("persisted_signals", "N/A")),
            f"{eval_log.get('drift_score', 0):.1%}",
        ],
    }
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

    # Category breakdown tables
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("New Signal Categories")
        cat_new = eval_log.get("category_breakdown_new", {})
        if cat_new:
            df_cat = pd.DataFrame(list(cat_new.items()), columns=["Category", "Count"])
            st.dataframe(df_cat.sort_values("Count", ascending=False), hide_index=True, use_container_width=True)
    with col2:
        st.subheader("Removed Signal Categories")
        cat_rem = eval_log.get("category_breakdown_removed", {})
        if cat_rem:
            df_cat2 = pd.DataFrame(list(cat_rem.items()), columns=["Category", "Count"])
            st.dataframe(df_cat2.sort_values("Count", ascending=False), hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown(f"*SEC Risk Signal Drift Analyzer · Anisha Tiwary · KIIT University 2026*")
