# The Theory Behind Fresta Finance

> A non-technical introduction to the structural entropy approach applied to financial markets.  
> For the full theoretical framework: [doi.org/10.5281/zenodo.18251304](https://doi.org/10.5281/zenodo.18251304)

---

## 1. The Problem With How We Evaluate Companies

Most financial risk models do something like this:

```
Risk Score = (Debt Ratio × 0.3) + (Volatility × 0.4) + (Margins × 0.3)
```

This is additive scoring. It assumes metrics are independent — that you can score each one separately and combine them linearly. It's intuitive, easy to compute, and **structurally wrong**.

Consider two companies:

- **Company A**: great margins, moderate debt, low volatility. Score: 72.
- **Company B**: great margins, moderate debt, low volatility. Score: 72.

They look identical. But Company A has 100% of its manufacturing concentrated in a single country under geopolitical risk, supplied by a single foundry with no viable alternative. Company B operates with distributed supply chains and diversified geographic roots. Under stress, A collapses and B holds.

Additive models cannot see this. They operate entirely at the **1st order** — local financial properties — and are blind to the four structural layers that actually determine systemic resilience.

This is not a calibration problem. It is not solved by adding more financial ratios. It is a structural problem: the model is missing entire dimensions of reality.

---

## 2. Five Orders of Financial Reality

Fresta Finance decomposes company risk into five structural orders, each capturing a distinct layer of reality that the previous orders cannot see.

### 1st Order — E0: What the Company Is

Local, intrinsic financial properties:

- **Recycling capacity** (45%): Can the company generate and reinvest surplus? (profit margins, operating margins, FCF, ROE)
- **Structural fragility** (35%): How leveraged and illiquid is it? (debt/equity, current ratio)
- **Noise** (20%): How volatile is its market behavior? (annualised volatility, max drawdown)

Each block is scored as a percentile rank across all S&P 500 companies — so the score always reflects relative structural position, not absolute accounting values.

This is what traditional models measure. It's necessary but insufficient.

### 2nd Order — E_upstream: What the Company Depends On

No company exists in isolation. Every company inherits entropy from its dependencies — sector peers, supply chain partners, market infrastructure.

Fresta Finance builds a **weighted dependency graph** from sector groupings, market capitalization weights, and price correlations (threshold: r > 0.70). It then propagates entropy iteratively:

```
E_upstream(t) = E0(t) + α × Σ [w(dep) × E(dep)]
```

Where α = 0.30 controls how much upstream entropy each company absorbs per iteration.

Two additional penalties apply at this order:

- **Concentration penalty** (Herfindahl index): companies that depend heavily on a single peer or sector are more fragile than those with distributed dependencies
- **Cycle penalty**: circular dependencies (A depends on B, B depends on A) create feedback loops that amplify stress rather than absorbing it

### 3rd Order — E_inherited: What the System Inherits

Above the sector level, all companies are embedded in a macro infrastructure. This layer is invisible to balance sheet analysis but determines the ceiling of systemic risk.

Fresta Finance models this through **virtual nodes** — abstract representations of macro forces:

| Virtual Node | What it represents |
|---|---|
| `SPY` | Broad market stress (S&P 500 volatility + drawdown) |
| `VIX` | Implied volatility regime |
| `RATES` | Interest rate stress (via TLT) |
| `MACRO` | Combined macro environment (SPY + VIX weighted) |

Different sectors inherit different macro stresses:
- **Financial Services** → inherits from `RATES` (rate sensitivity)
- **Communication Services** → inherits from `MACRO` (macro cyclicality)
- **Everything else** → inherits from `SPY` (general market)

A **saturation penalty** applies when infrastructure entropy exceeds a critical threshold (65.0) — modeling the non-linear amplification that occurs when base infrastructure is itself under stress. If Utilities and Energy are saturated, every company above them inherits that stress regardless of how good their individual numbers look.

### 4th Order — E_tree: Where the Roots Are

The first three orders work entirely with financial and market data. The fourth order asks a different question: **what physical and structural reality sustains this company?**

For each company, a dependency tree is extracted — not from balance sheets but from real-world supply chain knowledge:

```
Company
  └── Customers (who buys, sector, geography, criticality)
  └── Suppliers (who sells, what, geography, alternatives count)
        └── Geographic roots (regions weighted by real exposure)
              └── Single points of failure (alternatives = 0)
              └── Circular dependencies (feedback loops)
```

Key structural metrics computed from the tree:

| Metric | What it measures |
|---|---|
| `root_diversity` | Shannon entropy of geographic root distribution |
| `single_root_risk` | True if any single root > 60% of supply chain weight |
| `supplier_hhi` | Herfindahl index of supplier concentration |
| `customer_hhi` | Herfindahl index of customer concentration |
| `spof_count` | Explicit single points of failure (no viable alternative) |
| `circular_count` | Feedback loops in the supply chain |

The E_tree score penalises geographic concentration, supplier monopolies, critical dependencies, and circular structures:

```
E_tree = max_root_weight × 40      # geographic concentration
       + supplier_hhi × 20         # supplier monopoly
       + critical_ratio × 20       # dependency without alternatives
       + customer_hhi × 10         # customer concentration
       + spof_count × 5            # explicit SPOFs (capped at 30)
       + circular_count × 8        # circular dependencies (capped at 24)
```

**Why this order matters:** The 4th order makes visible the fragility that financial metrics actively conceal. NVIDIA has E0 = 13.6 — the best in the entire S&P 500 index. Its margins, FCF, and ROE are extraordinary. And yet NVIDIA is simultaneously one of the most structurally fragile companies in the analysis: 100% dependent on TSMC for advanced chips, TSMC located in Taiwan, Taiwan under active geopolitical threat. No financial ratio captures this. The dependency tree does.

**Adaptive depth:** before calling the LLM, each company is assigned a complexity score (0–10) from sector, market cap, and business description — with no LLM needed. Simple companies (local utilities, domestic retailers) get a short shallow prompt (~350 tokens). Complex companies (semiconductors, global tech, defense) get a deep structured prompt (~850 tokens). Known highly complex tickers (NVDA, AAPL, BA, TSMC, etc.) are forced directly to the COMPLEX tier. This reduces total analysis time by 3–5× without reducing quality where it matters.

**Anti-hallucination:** all LLM outputs are validated against a strict JSON schema. Numeric fields are clamped to valid ranges. Lists are type-checked. Weights are renormalised. Invalid or implausible values are replaced with conservative defaults — never invented. Up to 3 retry attempts per company. Results are stored as individual validated JSON files (one per company) and never re-queried once validated.

### 5th Order — E_political: What Politics Decides

The fourth order extracts the dependency tree. The fifth order asks: **what does that tree's geographic and structural position mean politically?**

A geographic root is not just a location — it is an embedded political risk. Taiwan exposure is not the same as German exposure. China revenue is not the same as US revenue. A single-root dependency in the Taiwan Strait carries a different risk profile than a single-root dependency in the Netherlands.

**Two-stage analysis:**

*Stage 1 — Sector baseline:* sector-level political risk is assessed once and cached. ~12 LLM calls cover all sectors, establishing context for:

| Dimension | What it captures |
|---|---|
| `tariff_risk` | Import/export tariff and trade war exposure |
| `regulatory_risk` | Antitrust, data privacy, environmental, financial regulation |
| `geopolitical_risk` | Conflict zones, instability, sanctions exposure |
| `china_exposure` | Supply chain or revenue dependence on China |
| `taiwan_exposure` | Supply chain dependence on Taiwan |
| `policy_dependency` | Government subsidies, contracts, or favorable policy reliance |

*Stage 2 — Company adjustment:* each company's score is adjusted against the sector baseline using its tree data as context. A semiconductor company with TSMC as its sole supplier (alternatives = 0) scores significantly worse on `taiwan_exposure` than a peer with diversified Asian manufacturing — even if both are in the same sector. The tree provides the structural evidence; the political model translates it into risk.

---

## 3. Why Non-Additivity Matters

The key theoretical claim underpinning all five orders is **structural non-additivity**:

> A system's total entropy is not the sum of its parts. A critical weakness at any order can nullify strengths at lower orders.

This is captured in the unified formula:

```
E_unified = 0.40 × E_total + 0.35 × E_tree + 0.25 × E_political
```

A company with excellent E_total (strong financial health) can have a poor E_unified if its supply chain is concentrated in a single geopolitical root. The financial strength is real — but it is *fragile* in the structural sense: one disruption at the root level can collapse everything built above it.

**A concrete example from the S&P 500 data (March 2026):**

- **AMD** (financial rank: #3, unified rank: #495, Δ = −492): AMD has strong financial health — lean operations, good margins, strong FCF. Financially it looks like one of the most resilient companies in the index. But AMD is 100% fabless, entirely dependent on TSMC for all advanced chips. No alternative foundry can manufacture at 5nm or below at scale. The Taiwan Strait is not a tail risk — it is AMD's primary structural dependency. Financial metrics cannot see this. Orders 4 and 5 do.

- **BRK-B** (financial rank: #2, unified rank: #1): Berkshire Hathaway's financial strength is matched by structural resilience at every order. Insurance, rail, energy, retail, financial holdings — no single geographic root, no SPOF, no China/Taiwan exposure, no policy dependency. The unified score confirms what the financial score already suggested: BRK-B is structurally coherent across all five orders.

- **NVDA** (financial rank: #1, unified rank: significantly lower): the most striking case in the dataset. NVDA has the best E0 in the entire index. And yet it has one of the highest E_tree scores. The company that looks most financially resilient is simultaneously one of the most structurally fragile. This is the core claim of the framework made visible: financial health and structural resilience are not the same thing, and conflating them is precisely the kind of systematic error that the five-order model is designed to correct.

The difference is not just that these companies are "riskier" or "safer" by traditional metrics. It's that their structural position across all five orders either amplifies or absorbs stress — and that amplification structure is invisible to financial analysis alone.

---

## 4. What the Score Means in Practice

The E_unified score is **not a prediction** — it is a structural diagnostic.

A high score does not mean a company will fail. It means the company is in a structural position where stress, if it arrives, will propagate and amplify rather than attenuate. A low score means the company's structural position tends to absorb and dissipate stress across all five orders.

**Score interpretation (E_unified):**

| Range | Structural interpretation |
|---|---|
| < 60 | Low systemic entropy — resilient across all five orders |
| 60 – 80 | Moderate entropy — some amplification risk at upper orders |
| 80 – 100 | High entropy — significant supply chain or political fragility |
| > 100 | Critical entropy — multiple amplification layers active simultaneously |

**What makes a score move across runs:**

Financial scores (E_total) change as market data changes — volatility regimes shift, sector correlations evolve, macro stress rises or falls. Tree scores (E_tree) change slowly — supply chains evolve over months or years. Political scores (E_political) change with geopolitical events — tariff escalations, regulatory actions, election outcomes, sanctions. This is by design: the unified score reflects the current structural state of the system, not a fixed property of the company.

**The scores also interact.** A company with rising financial stress (E_total increasing) in a sector with high political risk (E_political high) and concentrated supply chains (E_tree high) is in a qualitatively different position than three companies each with only one of those problems. The unified formula does not make this interaction explicit — that is future work — but the component scores together provide the diagnostic picture.

---

## 5. The Fresta Lens Framework

Fresta Finance is one application of a broader theoretical framework developed over five volumes (~500 pages) addressing fundamental problems in evaluation theory, system coherence, and structural incompleteness.

The core insight of the framework — applicable far beyond finance — is that **any evaluation system that ignores upstream interdependencies and inherited structural stress is not merely incomplete: it is systematically biased toward the forces that benefit most from that blindness**.

In financial markets, those forces are: marketing narratives built around individual company metrics, benchmark-hugging behavior that ignores sector-level fragility, and risk models calibrated on historical correlations that fail precisely when those correlations break down under stress.

The five-order decomposition is the framework's answer: make each invisible layer explicit, quantify it with the best available method for that layer (financial data for orders 1–3, dependency tree extraction for order 4, political risk assessment for order 5), and let the structural reality speak.

Orders 4 and 5 extend this logic beyond financial data into the physical and political reality that financial data is supposed to represent — but systematically fails to capture at the structural level. A stock price reflects expectations about future cash flows. It does not reflect the probability that the foundry producing the company's only product is located 180km from a military adversary with stated territorial ambitions. The five-order model does.

**Further reading:**
- Full framework (5 volumes): [doi.org/10.5281/zenodo.18251304](https://doi.org/10.5281/zenodo.18251304)
- EDGE — the domain evaluation tool: [github.com/EviAmarates/fresta-edge](https://github.com/EviAmarates/fresta-edge)
- Academic paper on Structural Domain Analysis: [PAPER.md in fresta-edge](https://github.com/EviAmarates/fresta-edge/blob/main/PAPER.md)
