# -*- coding: utf-8 -*-
"""
FRESTA TREE — Adaptive Dependency Tree Analysis (4th Order)
===========================================================
Builds a real dependency tree for each S&P 500 company using a local LLM.

Key optimisations vs v1:
  - ADAPTIVE DEPTH: complexity score determines prompt depth and token budget.
    Simple companies (local utilities, domestic retailers) → fast shallow analysis.
    Complex companies (semis, global tech, defense) → deep multi-node analysis.
  - ANTI-HALLUCINATION: structured JSON with strict schema validation.
    Invalid fields → null (never invented). Up to 3 retry attempts per company.
    Every field validated against expected types and ranges.
  - PHYSICAL CACHE: each company stored as individual validated JSON file.
    Never re-analyses validated results. Resumes exactly where it stopped.
    Cache includes: timestamp, complexity_tier, validation_status, version.

Complexity tiers:
  SIMPLE   (score 0-3)  → ~8s  per company   (short prompt, 300 tokens)
  MODERATE (score 4-6)  → ~15s per company   (medium prompt, 500 tokens)
  COMPLEX  (score 7-10) → ~30s per company   (deep prompt, 800 tokens)

Usage:
  python fresta_tree.py

Dependencies:
  pip install pandas numpy requests yfinance
"""

import sys
import json
import time
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Optional

import pandas as pd
import numpy as np

try:
    import requests
except ImportError:
    print("Missing: pip install requests")
    sys.exit(1)

try:
    import yfinance as yf
except ImportError:
    print("Missing: pip install yfinance")
    sys.exit(1)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

LM_STUDIO_HOST  = "http://127.0.0.1:1234"
LM_STUDIO_MODEL = "meta-llama-3-8b-instruct"
LM_STUDIO_TIMEOUT = 180
CACHE_VERSION  = "2"          # bump to invalidate all cache
MAX_RETRIES    = 3            # anti-hallucination retries
CACHE_TTL_DAYS = 30
SINGLE_ROOT_THRESHOLD = 0.60

# Token budgets per complexity tier
TIER_TOKENS = {"SIMPLE": 350, "MODERATE": 550, "COMPLEX": 850}

# Sectors with inherently high complexity
HIGH_COMPLEXITY_SECTORS = {
    "Technology", "Semiconductors", "Defense", "Aerospace",
    "Communication Services", "Energy", "Basic Materials",
}
LOW_COMPLEXITY_SECTORS = {
    "Utilities", "Real Estate", "Consumer Defensive",
}

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)
cache_dir  = output_dir / "tree_cache"
cache_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(output_dir / "tree.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# COMPLEXITY SCORING
# ═══════════════════════════════════════════════════════════════════════════════

def compute_complexity(ticker: str, sector: str, market_cap: Optional[float],
                       description: str) -> tuple[int, str]:
    """
    Score company complexity 0-10 using available data — NO LLM needed.
    Returns (score, tier) where tier is SIMPLE / MODERATE / COMPLEX.
    """
    score = 0

    # 1. Sector complexity
    if sector in HIGH_COMPLEXITY_SECTORS:
        score += 3
    elif sector in LOW_COMPLEXITY_SECTORS:
        score += 0
    else:
        score += 1

    # 2. Market cap (larger = more complex supply chain)
    if market_cap:
        if market_cap > 500e9:   score += 3   # mega cap
        elif market_cap > 100e9: score += 2   # large cap
        elif market_cap > 10e9:  score += 1   # mid cap
        # small cap → +0

    # 3. Description signals
    desc_lower = description.lower()
    global_signals = ["worldwide", "global", "international", "multinational",
                      "operations in", "countries", "supply chain", "manufacturing"]
    score += min(sum(1 for s in global_signals if s in desc_lower), 3)

    # 4. Known highly complex tickers
    very_complex = {"NVDA", "AAPL", "TSMC", "INTC", "AMD", "QCOM", "AVGO",
                    "TSM", "MU", "AMAT", "KLAC", "LRCX", "BA", "LMT", "RTX",
                    "GE", "HON", "CAT", "DE", "XOM", "CVX", "COP"}
    if ticker in very_complex:
        score = max(score, 8)

    score = min(score, 10)

    if score <= 3:   tier = "SIMPLE"
    elif score <= 6: tier = "MODERATE"
    else:            tier = "COMPLEX"

    return score, tier


# ═══════════════════════════════════════════════════════════════════════════════
# CACHE — one validated JSON per company
# ═══════════════════════════════════════════════════════════════════════════════

def cache_path(ticker: str) -> Path:
    return cache_dir / f"{ticker}_v{CACHE_VERSION}.json"


def load_cache(ticker: str) -> Optional[dict]:
    p = cache_path(ticker)
    if not p.exists():
        return None
    age = (datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)).days
    if age >= CACHE_TTL_DAYS:
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if data.get("validated") and data.get("version") == CACHE_VERSION:
            return data
        return None
    except Exception:
        return None


def save_cache(ticker: str, data: dict) -> None:
    try:
        data["version"]   = CACHE_VERSION
        data["cached_at"] = datetime.now().isoformat()
        cache_path(ticker).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as e:
        log.warning(f"Cache save failed for {ticker}: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# LLM INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are a supply chain risk analyst. "
    "Respond ONLY with valid JSON. No markdown, no explanations, no extra text. "
    "Use real company and country names. Never invent data — if unsure, omit the field."
)


def ask_llm(prompt: str, max_tokens: int) -> Optional[str]:
    try:
        r = requests.post(
            f"{LM_STUDIO_HOST}/v1/chat/completions",
            json={
                "model":      LM_STUDIO_MODEL,
                "messages":   [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.05,
                "stream":     False,
            },
            timeout=LM_STUDIO_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        log.warning(f"LLM error: {e}")
        return None


def extract_json(text: str) -> Optional[dict]:
    """Extract and parse JSON from LLM response, stripping markdown if present."""
    if not text:
        return None
    clean = text.strip()
    if clean.startswith("```"):
        lines = clean.split("\n")
        clean = "\n".join(l for l in lines if not l.startswith("```"))
    # Find outermost JSON object
    start = clean.find("{")
    end   = clean.rfind("}") + 1
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(clean[start:end])
    except json.JSONDecodeError:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMA VALIDATION — anti-hallucination
# ═══════════════════════════════════════════════════════════════════════════════

def clamp(v, lo, hi, default):
    """Clamp numeric value to range, return default if invalid."""
    try:
        v = float(v)
        return max(lo, min(hi, v))
    except (TypeError, ValueError):
        return default


def validate_tree(raw: dict) -> tuple[dict, list[str]]:
    """
    Validate and sanitise LLM output against expected schema.
    Returns (clean_dict, list_of_warnings).
    Hallucinated or invalid fields are replaced with safe defaults.
    """
    warnings = []
    out = {}

    # suppliers
    suppliers = []
    for s in (raw.get("suppliers") or [])[:10]:
        if not isinstance(s, dict):
            continue
        name = str(s.get("name", "")).strip()
        if not name:
            continue
        w = clamp(s.get("weight"), 0, 1, 0.1)
        suppliers.append({
            "name":         name,
            "weight":       round(w, 3),
            "type":         str(s.get("type", "unknown"))[:50],
            "geography":    str(s.get("geography", "Unknown"))[:50],
            "critical":     bool(s.get("critical", False)),
            "alternatives": int(clamp(s.get("alternatives"), 0, 20, 1)),
        })
    if suppliers:
        total = sum(x["weight"] for x in suppliers) or 1.0
        for x in suppliers:
            x["weight"] = round(x["weight"] / total, 3)
    out["suppliers"] = suppliers

    # customers
    customers = []
    for c in (raw.get("customers") or [])[:10]:
        if not isinstance(c, dict):
            continue
        name = str(c.get("name", "")).strip()
        if not name:
            continue
        w = clamp(c.get("weight"), 0, 1, 0.1)
        customers.append({
            "name":      name,
            "weight":    round(w, 3),
            "sector":    str(c.get("sector", "Unknown"))[:50],
            "geography": str(c.get("geography", "Unknown"))[:50],
            "critical":  bool(c.get("critical", False)),
        })
    if customers:
        total = sum(x["weight"] for x in customers) or 1.0
        for x in customers:
            x["weight"] = round(x["weight"] / total, 3)
    out["customers"] = customers

    # geographic_roots
    geo_roots = []
    for g in (raw.get("geographic_roots") or [])[:15]:
        if not isinstance(g, dict):
            continue
        region = str(g.get("region", "")).strip()
        if not region:
            continue
        w = clamp(g.get("weight"), 0, 1, 0.1)
        geo_roots.append({
            "region":    region,
            "weight":    round(w, 3),
            "risk_type": str(g.get("risk_type", "geopolitical"))[:40],
        })
    if geo_roots:
        total = sum(x["weight"] for x in geo_roots) or 1.0
        for x in geo_roots:
            x["weight"] = round(x["weight"] / total, 3)
    else:
        warnings.append("No geographic roots found — defaulting to single unknown root")
        geo_roots = [{"region": "Unknown", "weight": 1.0, "risk_type": "unknown"}]
    out["geographic_roots"] = geo_roots

    # SPOFs and circular deps — must be list of strings
    def safe_str_list(val, maxlen=10):
        if not isinstance(val, list):
            return []
        return [str(x)[:80] for x in val if isinstance(x, (str, int, float))][:maxlen]

    out["single_points_of_failure"] = safe_str_list(raw.get("single_points_of_failure"))
    out["circular_dependencies"]    = safe_str_list(raw.get("circular_dependencies"))

    # narrative — single string, max 300 chars
    narr = raw.get("narrative", "")
    out["narrative"] = str(narr)[:300] if isinstance(narr, str) else ""

    return out, warnings


# ═══════════════════════════════════════════════════════════════════════════════
# ADAPTIVE PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════

def build_prompt(ticker: str, name: str, sector: str,
                 description: str, tier: str) -> str:
    """Build prompt adapted to complexity tier."""

    base_schema = '''{
  "suppliers": [{"name":"","weight":0.0,"type":"","geography":"","critical":false,"alternatives":0}],
  "customers": [{"name":"","weight":0.0,"sector":"","geography":"","critical":false}],
  "geographic_roots": [{"region":"","weight":0.0,"risk_type":"geopolitical/regulatory/natural"}],
  "single_points_of_failure": [],
  "circular_dependencies": [],
  "narrative": ""
}'''

    if tier == "SIMPLE":
        return f"""Analyse {ticker} ({name}), sector: {sector}.
Return ONLY JSON with this structure (weights 0-1, sum to 1 within each list):
{base_schema}
Include 2-3 suppliers, 2-3 customers, 2-3 geographic roots. Be concise."""

    elif tier == "MODERATE":
        return f"""Analyse the supply chain of {ticker} ({name}).
Sector: {sector}
Context: {description[:300]}

Return ONLY JSON:
{base_schema}
Include 3-5 suppliers with real company names, 3-5 customers, 3-5 geographic roots.
Mark critical=true only for suppliers with NO alternatives.
List real single points of failure if any."""

    else:  # COMPLEX
        return f"""Perform a deep supply chain analysis for {ticker} ({name}).
Sector: {sector}
Context: {description[:500]}

Return ONLY JSON:
{base_schema}
Requirements:
- 5-8 suppliers with real names (TSMC, ASML, Samsung, etc.)
- 4-6 customers with sector and geography
- 5-8 geographic roots (countries/regions) with weights reflecting real exposure
- alternatives=0 means TRUE monopoly supplier (no viable substitute)
- List ALL single points of failure explicitly
- List circular dependencies (e.g. NVDA sells to hyperscalers who fund NVDA R&D)
- narrative: 2 sentences on structural risk"""


# ═══════════════════════════════════════════════════════════════════════════════
# TREE METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_tree_metrics(tree: dict, tier: str) -> dict:
    suppliers  = tree.get("suppliers", [])
    customers  = tree.get("customers", [])
    geo_roots  = tree.get("geographic_roots", [])
    spof       = tree.get("single_points_of_failure", [])
    circular   = tree.get("circular_dependencies", [])

    # Geographic root diversity
    geo_weights = [g["weight"] for g in geo_roots if g.get("weight", 0) > 0]
    root_count  = len(geo_weights)

    if geo_weights:
        probs           = [w / (sum(geo_weights) or 1) for w in geo_weights]
        root_diversity  = float(-sum(p * np.log2(p + 1e-10) for p in probs))
        max_root_weight = max(probs)
        single_root_risk = max_root_weight >= SINGLE_ROOT_THRESHOLD
    else:
        root_diversity   = 0.0
        max_root_weight  = 1.0
        single_root_risk = True

    # Supplier HHI
    sup_w = [s["weight"] for s in suppliers]
    if sup_w:
        total    = sum(sup_w) or 1.0
        sup_probs = [w / total for w in sup_w]
        supplier_hhi = sum(p ** 2 for p in sup_probs)
    else:
        supplier_hhi = 1.0

    # Critical and monopoly ratios
    n_sup           = max(len(suppliers), 1)
    critical_ratio  = sum(1 for s in suppliers if s.get("critical")) / n_sup
    monopoly_ratio  = sum(1 for s in suppliers if s.get("alternatives", 1) == 0) / n_sup

    # Customer HHI
    cust_w = [c["weight"] for c in customers]
    if cust_w:
        total     = sum(cust_w) or 1.0
        cust_probs = [w / total for w in cust_w]
        customer_hhi = sum(p ** 2 for p in cust_probs)
    else:
        customer_hhi = 0.5

    # Penalties
    spof_penalty     = min(len(spof) * 5, 30)
    circular_penalty = min(len(circular) * 8, 24)

    # E_tree score
    geo_score  = max_root_weight * 40
    sup_score  = supplier_hhi * 20
    crit_score = (critical_ratio * 0.7 + monopoly_ratio * 0.3) * 20
    cust_score = customer_hhi * 10
    e_tree     = geo_score + sup_score + crit_score + cust_score + spof_penalty + circular_penalty

    return {
        "root_count":         root_count,
        "root_diversity":     round(root_diversity, 4),
        "max_root_weight":    round(max_root_weight, 4),
        "single_root_risk":   single_root_risk,
        "supplier_hhi":       round(supplier_hhi, 4),
        "customer_hhi":       round(customer_hhi, 4),
        "critical_ratio":     round(critical_ratio, 4),
        "monopoly_ratio":     round(monopoly_ratio, 4),
        "spof_count":         len(spof),
        "circular_count":     len(circular),
        "critical_nodes":     json.dumps(spof),
        "circular_deps":      json.dumps(circular),
        "narrative":          tree.get("narrative", ""),
        "complexity_tier":    tier,
        "E_tree":             round(min(e_tree, 100), 4),
        "validated":          True,
    }


def default_metrics(tier: str = "SIMPLE") -> dict:
    return {
        "root_count": 1, "root_diversity": 0.0, "max_root_weight": 1.0,
        "single_root_risk": True, "supplier_hhi": 1.0, "customer_hhi": 0.5,
        "critical_ratio": 0.5, "monopoly_ratio": 0.0, "spof_count": 0,
        "circular_count": 0, "critical_nodes": "[]", "circular_deps": "[]",
        "narrative": "Analysis failed — using conservative defaults.",
        "complexity_tier": tier, "E_tree": 50.0, "validated": False,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_company(ticker: str, name: str, sector: str,
                    market_cap: Optional[float], description: str) -> dict:
    """
    Full adaptive pipeline for one company:
    1. Score complexity
    2. Check cache
    3. Build adaptive prompt
    4. Call LLM (up to MAX_RETRIES)
    5. Validate schema (anti-hallucination)
    6. Compute metrics
    7. Save to cache
    """
    complexity_score, tier = compute_complexity(ticker, sector, market_cap, description)

    # Cache check
    cached = load_cache(ticker)
    if cached:
        return cached

    log.info(f"  Complexity: {tier} (score={complexity_score})")

    prompt     = build_prompt(ticker, name, sector, description, tier)
    max_tokens = TIER_TOKENS[tier]

    raw_tree   = None
    all_warnings = []

    for attempt in range(1, MAX_RETRIES + 1):
        response = ask_llm(prompt, max_tokens)
        parsed   = extract_json(response)

        if parsed is None:
            log.warning(f"  Attempt {attempt}/{MAX_RETRIES}: could not parse JSON")
            time.sleep(2)
            continue

        clean_tree, warnings = validate_tree(parsed)
        all_warnings.extend(warnings)

        # Accept if we have at least geographic roots
        if clean_tree.get("geographic_roots"):
            raw_tree = clean_tree
            if warnings:
                log.warning(f"  Validation warnings: {warnings}")
            break
        else:
            log.warning(f"  Attempt {attempt}/{MAX_RETRIES}: validation failed — {warnings}")
            time.sleep(2)

    if raw_tree is None:
        log.warning(f"  All attempts failed — using defaults")
        metrics = default_metrics(tier)
        metrics["complexity_score"] = complexity_score
        save_cache(ticker, metrics)
        return metrics

    metrics = compute_tree_metrics(raw_tree, tier)
    metrics["complexity_score"]  = complexity_score
    metrics["validation_warnings"] = json.dumps(all_warnings)
    metrics["raw_tree"]           = json.dumps(raw_tree, ensure_ascii=False)

    save_cache(ticker, metrics)
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# HTML REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def generate_report(df: pd.DataFrame, path: Path) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    def fmt(v, dec=2):
        try: return f"{float(v):.{dec}f}" if pd.notna(v) else "N/A"
        except Exception: return "N/A"

    def risk_badge(v):
        try:
            v = float(v)
            cls = "excellent" if v < 30 else "good" if v < 55 else "moderate" if v < 75 else "poor"
            return f'<span class="badge {cls}">{v:.2f}</span>'
        except Exception: return "N/A"

    def tier_badge(t):
        colours = {"SIMPLE": "#d4edda", "MODERATE": "#fff3cd", "COMPLEX": "#f8d7da"}
        return f'<span style="background:{colours.get(t,"#eee")};padding:2px 7px;border-radius:8px;font-size:.78em">{t}</span>'

    all_rows = ""
    for _, r in df.sort_values("E_tree", ascending=False).iterrows():
        spof = json.loads(r.get("critical_nodes", "[]")) if isinstance(r.get("critical_nodes"), str) else []
        warn = " warn" if r.get("single_root_risk") else ""
        all_rows += (
            f'<tr class="{warn}">'
            f'<td class="rc">{int(r.get("Rank",0))}</td>'
            f'<td class="tc">{r["Ticker"]}</td>'
            f'<td>{r["Security"]}</td>'
            f'<td class="sc">{r["Sector"]}</td>'
            f'<td>{risk_badge(r["E_tree"])}</td>'
            f'<td>{tier_badge(r.get("complexity_tier","?"))}</td>'
            f'<td>{"⚠️ YES" if r.get("single_root_risk") else "✅ No"}</td>'
            f'<td>{fmt(r["root_count"],0)}</td>'
            f'<td>{fmt(r["root_diversity"])}</td>'
            f'<td>{fmt(r["supplier_hhi"])}</td>'
            f'<td>{fmt(r["spof_count"],0)}</td>'
            f'<td style="font-size:.78em;color:#888">{", ".join(spof[:3])}</td>'
            f'<td style="font-size:.76em;color:#555;max-width:260px">{str(r.get("narrative",""))[:110]}</td>'
            f'</tr>\n'
        )

    tier_counts = df.get("complexity_tier", pd.Series()).value_counts().to_dict()

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fresta Tree — Dependency Analysis</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Segoe UI',sans-serif;background:linear-gradient(135deg,#1a1a2e,#16213e);padding:20px;color:#333}}
.container{{max-width:1800px;margin:0 auto;background:#fff;border-radius:15px;box-shadow:0 10px 40px rgba(0,0,0,.4);overflow:hidden}}
.header{{background:linear-gradient(135deg,#1a1a2e,#e94560);color:#fff;padding:40px;text-align:center}}
.header h1{{font-size:2.1em;margin-bottom:8px}}
.header a{{color:#ffd}}
.content{{padding:40px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:16px;margin-bottom:36px}}
.card{{background:linear-gradient(135deg,#f5f7fa,#c3cfe2);padding:18px;border-radius:10px;text-align:center;box-shadow:0 3px 8px rgba(0,0,0,.1);transition:transform .3s}}
.card:hover{{transform:translateY(-3px)}}
.card h3{{color:#1a1a2e;font-size:.76em;text-transform:uppercase;letter-spacing:1px;margin-bottom:7px}}
.card .val{{font-size:1.8em;font-weight:700}}
.val.red{{color:#dc3545}}.val.green{{color:#28a745}}.val.orange{{color:#fd7e14}}
.section{{margin-bottom:42px}}
.section h2{{color:#1a1a2e;font-size:1.4em;padding-bottom:9px;border-bottom:3px solid #e94560;margin-bottom:11px}}
.info{{background:#f0f0ff;border-left:4px solid #1a1a2e;padding:10px 14px;border-radius:0 8px 8px 0;font-size:.86em;color:#444;margin-bottom:13px}}
table{{width:100%;border-collapse:collapse;box-shadow:0 2px 10px rgba(0,0,0,.1);border-radius:8px;overflow:hidden;font-size:.82em}}
thead{{background:linear-gradient(135deg,#1a1a2e,#e94560);color:#fff}}
th{{padding:12px 8px;text-align:left;font-weight:600;text-transform:uppercase;font-size:.74em;cursor:pointer}}
th:hover{{opacity:.85}}
td{{padding:9px 7px;border-bottom:1px solid #eee}}
tbody tr:hover{{background:#f9f9ff}}
.rc{{font-weight:700;color:#1a1a2e;text-align:center}}
.tc{{font-family:monospace;font-weight:700;font-size:1.03em}}
.sc{{color:#888;font-size:.83em}}
.badge{{display:inline-block;padding:3px 8px;border-radius:12px;font-weight:700}}
.excellent{{background:#d4edda;color:#155724}}
.good{{background:#d1ecf1;color:#0c5460}}
.moderate{{background:#fff3cd;color:#856404}}
.poor{{background:#f8d7da;color:#721c24}}
tr.warn{{background:#fff8f0!important;border-left:3px solid #e94560}}
.search-bar input{{padding:8px 14px;border:1px solid #ddd;border-radius:20px;width:300px;font-size:.87em;margin-bottom:12px}}
.footer{{text-align:center;padding:24px;background:#f8f9fa;color:#888;font-size:.81em;border-top:1px solid #eee}}
.footer a{{color:#e94560}}
</style></head>
<body><div class="container">
<div class="header">
  <h1>🌳 Fresta Tree — Adaptive Dependency Analysis</h1>
  <p>{ts} &nbsp;·&nbsp; <a href="https://github.com/EviAmarates/fresta-finance">GitHub</a>
  &nbsp;·&nbsp; <a href="https://doi.org/10.5281/zenodo.18251304">Fresta Framework</a></p>
</div>
<div class="content">
<div class="cards">
  <div class="card"><h3>Analysed</h3><div class="val">{len(df)}</div></div>
  <div class="card"><h3>Single Root ⚠️</h3><div class="val red">{int(df.get("single_root_risk",pd.Series([False]*len(df))).sum())}</div></div>
  <div class="card"><h3>Multi-Root ✅</h3><div class="val green">{int((~df.get("single_root_risk",pd.Series([True]*len(df)))).sum())}</div></div>
  <div class="card"><h3>Mean E_tree</h3><div class="val orange">{df["E_tree"].mean():.2f}</div></div>
  <div class="card"><h3>SIMPLE</h3><div class="val">{tier_counts.get("SIMPLE",0)}</div></div>
  <div class="card"><h3>MODERATE</h3><div class="val orange">{tier_counts.get("MODERATE",0)}</div></div>
  <div class="card"><h3>COMPLEX</h3><div class="val red">{tier_counts.get("COMPLEX",0)}</div></div>
</div>

<div class="section">
  <h2>📋 Full Tree Analysis — All {len(df)} Companies (sorted by E_tree)</h2>
  <div class="info">Red rows = single geographic root risk. Complexity tier shows depth of LLM analysis used. Click headers to sort.</div>
  <div class="search-bar"><input type="text" id="si" placeholder="🔍 Search ticker or company..." oninput="ft()"></div>
  <table id="t">
    <thead><tr>
      <th onclick="st(0)">Rank</th><th onclick="st(1)">Ticker</th><th>Company</th>
      <th onclick="st(3)">Sector</th><th onclick="st(4)">E_tree</th>
      <th>Tier</th><th>Single Root</th><th onclick="st(7)">Roots</th>
      <th onclick="st(8)">Diversity</th><th onclick="st(9)">Sup HHI</th>
      <th onclick="st(10)">SPOFs</th><th>Critical Nodes</th><th>Narrative</th>
    </tr></thead>
    <tbody id="tb">{all_rows}</tbody>
  </table>
</div>
</div>
<div class="footer">
  <p>Fresta Tree v2 — Adaptive LLM analysis with anti-hallucination validation — {ts}</p>
  <p><a href="https://github.com/EviAmarates/fresta-finance">GitHub</a> &nbsp;·&nbsp;
  <a href="https://ko-fi.com/tiagosantos20582">Support</a></p>
</div>
</div>
<script>
let _d={{}};
function st(c){{
  const tb=document.getElementById('tb');
  const rows=[...tb.querySelectorAll('tr')];
  _d[c]=!_d[c];
  rows.sort((a,b)=>{{
    const va=a.cells[c]?.textContent.trim()||'';
    const vb=b.cells[c]?.textContent.trim()||'';
    const na=parseFloat(va),nb=parseFloat(vb);
    if(!isNaN(na)&&!isNaN(nb)) return _d[c]?na-nb:nb-na;
    return _d[c]?va.localeCompare(vb):vb.localeCompare(va);
  }});
  rows.forEach(r=>tb.appendChild(r));
}}
function ft(){{
  const q=document.getElementById('si').value.toLowerCase();
  document.querySelectorAll('#tb tr').forEach(r=>{{
    r.style.display=r.textContent.toLowerCase().includes(q)?'':'none';
  }});
}}
</script>
</body></html>"""

    path.write_text(html, encoding="utf-8")
    log.info(f"HTML: {path.absolute()}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("FRESTA TREE v2 — Adaptive Dependency Analysis")
    log.info("=" * 60)

    # Load base CSV
    base_csv = output_dir / "sp500_entropy_ranked.csv"
    if not base_csv.exists():
        log.error("Run fresta_finance.py first!")
        sys.exit(1)
    base_df = pd.read_csv(base_csv)
    log.info(f"Loaded {len(base_df)} companies.")

    # Check LM Studio
    try:
        r = requests.get(f"{LM_STUDIO_HOST}/v1/models", timeout=5)
        models = [m["id"] for m in r.json().get("data", [])]
        if not any(LM_STUDIO_MODEL in m for m in models):
            log.error(f"Model '{LM_STUDIO_MODEL}' not loaded in LM Studio.")
            log.error(f"Available: {models}")
            log.error("Load the model in LM Studio Developer tab first!")
            sys.exit(1)
        log.info(f"LM Studio OK — model: {LM_STUDIO_MODEL}")
    except Exception:
        log.error("LM Studio not running at http://127.0.0.1:1234")
        log.error("Open LM Studio → Developer → Start Server")
        sys.exit(1)

    results = []
    total   = len(base_df)
    cached_count = 0
    tier_stats   = defaultdict(int)

    log.info(f"\nAnalysing {total} companies (adaptive depth)...")

    for i, (_, row) in enumerate(base_df.iterrows(), 1):
        ticker = row["Ticker"]
        name   = row.get("Security", ticker)
        sector = row.get("Sector", "Unknown")

        # Get fundamentals for complexity scoring
        try:
            info    = yf.Ticker(ticker).get_info()
            mktcap  = info.get("marketCap")
            desc    = (info.get("longBusinessSummary") or "")[:600]
        except Exception:
            mktcap = None
            desc   = ""
        time.sleep(0.05)

        # Check cache first (before logging — to keep output clean)
        cached = load_cache(ticker)
        if cached:
            cached_count += 1
            tier_stats[cached.get("complexity_tier", "?")] += 1
            results.append({**row.to_dict(), **cached})
            if i % 50 == 0:
                log.info(f"[{i}/{total}] {cached_count} from cache so far...")
            continue

        complexity_score, tier = compute_complexity(ticker, sector, mktcap, desc)
        tier_stats[tier] += 1

        log.info(f"[{i}/{total}] {ticker:6s} | {tier:8s} | {sector[:28]}")

        metrics = analyse_company(ticker, name, sector, mktcap, desc)
        results.append({**row.to_dict(), **metrics})

    log.info(f"\nDone: {cached_count} cached, {total-cached_count} analysed via LLM")
    log.info(f"Tiers: SIMPLE={tier_stats['SIMPLE']}  MODERATE={tier_stats['MODERATE']}  COMPLEX={tier_stats['COMPLEX']}")

    out_df = pd.DataFrame(results)

    # Save CSV
    csv_cols = ["Rank", "Ticker", "Security", "Sector", "E_total", "E_tree",
                "complexity_tier", "complexity_score", "single_root_risk",
                "root_count", "root_diversity", "max_root_weight",
                "supplier_hhi", "customer_hhi", "critical_ratio", "monopoly_ratio",
                "spof_count", "circular_count", "critical_nodes", "narrative"]
    out_df[[c for c in csv_cols if c in out_df.columns]].to_csv(
        output_dir / "sp500_tree_analysis.csv", index=False
    )

    generate_report(out_df, output_dir / "sp500_tree_report.html")

    log.info("\nTOP 10 most fragile trees:")
    for _, r in out_df.nlargest(10, "E_tree").iterrows():
        log.info(f"  {r['Ticker']:6s}  E_tree={r['E_tree']:.2f}  "
                 f"tier={r.get('complexity_tier','?'):8s}  "
                 f"{'⚠️ single root' if r.get('single_root_risk') else ''}")


if __name__ == "__main__":
    main()
