# Fresta Finance — S&P 500 Structural Entropy Ranking

> Applies the [Fresta Lens Framework](https://doi.org/10.5281/zenodo.18251304) to financial markets.  
> Ranks every S&P 500 company by **structural entropy** across **five orders**.

---

## What It Does

Most financial risk models score companies in isolation — debt ratio, volatility, margins — and aggregate them additively. This misses critical layers:

- **How metrics interact** — a company with great margins but extreme supply chain concentration may be more fragile than its individual scores suggest
- **What the system inherits** — sector dependencies, macro stress, and infrastructure concentration that no balance sheet will show you
- **Where the roots are** — geographic and structural roots of the supply chain, single points of failure, and circular dependencies
- **What politics decides** — tariff exposure, regulatory risk, geopolitical concentration, and policy dependency

Fresta Finance computes a **five-order entropy score** for every S&P 500 company:

```
E_unified = 0.40 × E_total + 0.35 × E_tree + 0.25 × E_political

where E_total = E0 + E_upstream + E_inherited
```

| Order | Script | What it captures |
|---|---|---|
| **E0** (1st) | `fresta_finance.py` | Local financial health — margins, debt, volatility |
| **E_upstream** (2nd) | `fresta_finance.py` | Propagated entropy through sector dependency graph |
| **E_inherited** (3rd) | `fresta_finance.py` | Systemic stress from macro, rates, and concentration |
| **E_tree** (4th) | `fresta_tree.py` | Supply chain dependency tree — root diversity, SPOFs, circular dependencies |
| **E_political** (5th) | `fresta_political.py` | Political-economic risk — tariffs, regulation, geopolitical concentration |

**Lower score = less structural entropy = more resilient company.**

---

## Pipeline

The three scripts run in sequence. Each builds on the previous output:

```
fresta_finance.py   →  sp500_entropy_ranked.csv
        ↓
fresta_tree.py      →  sp500_tree_analysis.csv
        ↓
fresta_political.py →  sp500_unified_report.html
```

---

## Usage

### Requirements

```bash
pip install pandas numpy yfinance requests anthropic
```

For orders 4 and 5, you need either a local LLM via [LM Studio](https://lmstudio.ai) (API-compatible, any model) or the [Anthropic API](https://console.anthropic.com). Set your key if using Claude:

```bash
set ANTHROPIC_API_KEY=sk-ant-...    # Windows
export ANTHROPIC_API_KEY=sk-ant-... # Linux/Mac
```

### Run (automated)

```bash
run_fresta.bat   # Windows — runs all three scripts in sequence
```

### Run (manual)

```bash
python fresta_finance.py    # ~15–30 min (downloads market data)
python fresta_tree.py       # ~10–15 min with Claude API / ~2–4h with local LLM
python fresta_political.py  # ~10–15 min with Claude API / ~1–2h with local LLM
```

The first run of `fresta_finance.py` downloads price data for ~500 tickers. Results are cached locally for 7 days. Tree and political analyses are cached per company as individual JSON files — runs can be interrupted and will resume exactly where they stopped.

---

## Output

| File | Description |
|---|---|
| `output/sp500_entropy_ranked.csv` | 1st–3rd order financial scores |
| `output/sp500_entropy_report.html` | Interactive 3-order financial report |
| `output/sp500_tree_analysis.csv` | 4th order supply chain tree metrics |
| `output/sp500_unified_report.html` | Full interactive 5-order unified report |

Open `sp500_unified_report.html` in any browser. It is fully self-contained — sortable by any column, filterable by text or single-root risk flag, with rank change indicators showing which companies look structurally different when supply chain and political risk are added.

---

## How Scoring Works

### E0 — 1st Order (Local Entropy)

Three blocks, each scored as a percentile rank across all S&P 500 companies:

| Block | Metrics | Weight |
|---|---|---|
| Recycling capacity | Profit margins, operating margins, FCF, ROE | 45% |
| Structural fragility | Debt/equity, current ratio | 35% |
| Noise/stress | Annualised volatility, max drawdown | 20% |

### E_upstream — 2nd Order (Propagated Entropy)

Builds a weighted dependency graph from sector groupings, market cap, and price correlations (threshold: r > 0.70). Runs iterative propagation (α = 0.30, 2 iterations): each company inherits a fraction of its dependencies' entropy. Adds a Herfindahl concentration penalty and a circular dependency penalty.

### E_inherited — 3rd Order (Systemic Stress)

Models macro overlays (SPY, VIX, RATES) as virtual nodes. Financial sector companies inherit rate stress; communication companies inherit macro stress. Adds a saturation penalty when infrastructure entropy exceeds a critical threshold (65.0).

### E_tree — 4th Order (Supply Chain Tree)

For each company, an LLM extracts the real dependency tree:

- **Customers** — who buys, with sector and geography
- **Suppliers** — who sells, with type, geography, and number of alternatives
- **Geographic roots** — countries/regions weighted by actual exposure
- **Single points of failure** — critical dependencies with no viable alternative
- **Circular dependencies** — feedback loops in the supply chain

Key metrics computed from the tree:

| Metric | What it measures |
|---|---|
| `root_diversity` | Shannon entropy of geographic root distribution |
| `single_root_risk` | True if any single root > 60% of supply chain weight |
| `supplier_hhi` | Herfindahl index of supplier concentration |
| `customer_hhi` | Herfindahl index of customer concentration |
| `spof_count` | Number of single points of failure |
| `circular_count` | Number of circular dependency loops |

**Adaptive depth:** before calling the LLM, each company is assigned a complexity score (0–10) from its sector, market cap, and business description. Simple companies (local utilities, domestic retailers) get a short shallow prompt and small token budget (~350 tokens). Complex companies (semiconductors, global tech, defense) get a deep structured prompt (~850 tokens). This makes the analysis 3–5× faster on average without sacrificing quality where it matters.

**Anti-hallucination:** all LLM outputs are validated against a strict JSON schema. Invalid or out-of-range fields are replaced with conservative defaults — never invented. Up to 3 retry attempts per company before falling back to safe defaults. Results are saved as individual validated JSON files and never re-analysed.

### E_political — 5th Order (Political-Economic Risk)

Two-stage analysis. First, sector-level political risk is assessed once and cached (~12 LLM calls total). Then each company is scored against that baseline using its tree data as context:

| Component | What it captures |
|---|---|
| `tariff_risk` | Exposure to import/export tariffs and trade wars |
| `regulatory_risk` | Antitrust, data privacy, environmental, financial regulation |
| `geopolitical_risk` | Conflict zones, political instability, sanctions exposure |
| `china_exposure` | Supply chain or revenue dependence on China |
| `taiwan_exposure` | Supply chain dependence on Taiwan (critical for semiconductors) |
| `policy_dependency` | Reliance on government subsidies, contracts, or favorable policy |

The company-level score adjusts the sector baseline using tree data as context — a semiconductor company with TSMC as its sole supplier is scored worse than a peer with diversified Asian manufacturing, even within the same sector.

---

## Key Findings (S&P 500, March 2026)

The five-order analysis reveals structural fragilities invisible to financial-only models.

**Most resilient (unified):** BRK-B, CBOE, CME, BLK, V — financial market infrastructure with diversified roots, regulatory moats, and no geographic concentration.

**Biggest rank falls when supply chain and political risk are added:**

| Ticker | Financial rank | Unified rank | Δ | Why |
|---|---|---|---|---|
| AMD | #3 | #495 | −492 | 100% TSMC dependency, Taiwan existential risk |
| SMCI | #331 | #503 | −172 | NVIDIA allocation + Taiwan components + accounting risk |
| ALB | #456 | #502 | −46 | Chile Atacama lithium concentration, China processing monopoly |
| QCOM | top quartile | bottom quartile | — | Fabless + ~60% China revenue exposure |

**Biggest rank improvements:**

| Ticker | Why |
|---|---|
| VICI | Domestic real estate, no supply chain concentration |
| NTRS, AMP, CINF | Domestic financial services, no physical supply chain risk |
| BRK-B | Most diversified company in the index — confirmed at all five orders |

**The NVDA paradox:** NVDA ranks #1 by financial health (E0 = 13.6 — best in the entire index). But it carries one of the highest E_tree scores. The most financially resilient company is simultaneously one of the most structurally fragile — a cash machine whose entire product depends on a single foundry (TSMC) in a single geopolitical hotspot (Taiwan Strait). The unified score captures what financial analysis cannot.

---

## Theoretical Grounding

This tool is a financial application of the **Fresta Lens Framework** — a five-volume theoretical work (~500 pages) on structural evaluation, entropy, and system coherence.

- Full framework: [doi.org/10.5281/zenodo.18251304](https://doi.org/10.5281/zenodo.18251304)
- EDGE (domain evaluation tool): [github.com/EviAmarates/fresta-edge](https://github.com/EviAmarates/fresta-edge)

---

## License

MIT

## Support

If this is useful: [ko-fi.com/tiagosantos20582](https://ko-fi.com/tiagosantos20582)
