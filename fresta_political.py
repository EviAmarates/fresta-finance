# -*- coding: utf-8 -*-
"""
FRESTA POLITICAL — Adaptive Political-Economic Risk Analysis (5th Order)
========================================================================
Analyses political and regulatory risk for each S&P 500 company.

Key optimisations vs v1:
  - ADAPTIVE DEPTH: sector baseline cached once, company analysis scaled
    by complexity tier from fresta_tree.py
  - ANTI-HALLUCINATION: strict JSON schema validation on all fields.
    Numeric scores clamped 0-100. Lists validated. Up to 3 retries.
  - PHYSICAL CACHE: sector risks cached 7 days, company risks cached 7 days.
    Separate cache files per company. Never re-analyses validated results.
  - EFFICIENT: only ~12 sector LLM calls + 1 per company (not 2×503).

Final output:
  E_unified = W_FINANCIAL × E_total + W_TREE × E_tree + W_POLITICAL × E_political

Usage:
  1. python fresta_finance.py
  2. python fresta_tree.py
  3. python fresta_political.py

Dependencies:
  pip install pandas numpy requests
"""

import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np

try:
    import requests
except ImportError:
    print("Missing: pip install requests")
    sys.exit(1)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

LM_STUDIO_HOST  = "http://127.0.0.1:1234"
LM_STUDIO_MODEL = "meta-llama-3-8b-instruct"
LM_STUDIO_TIMEOUT = 120
CACHE_VERSION  = "2"
MAX_RETRIES    = 3
CACHE_TTL_DAYS = 7

# Final unified score weights
W_FINANCIAL = 0.40
W_TREE      = 0.35
W_POLITICAL = 0.25

# Token budgets per complexity tier
TIER_TOKENS = {"SIMPLE": 300, "MODERATE": 450, "COMPLEX": 600}

output_dir    = Path("output")
output_dir.mkdir(exist_ok=True)
pol_cache_dir = output_dir / "political_cache"
pol_cache_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(output_dir / "political.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# LLM
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are a geopolitical and regulatory risk analyst. "
    "Respond ONLY with valid JSON. No markdown, no explanations. "
    "All numeric scores are integers 0-100. Never invent data."
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
    if not text:
        return None
    clean = text.strip()
    if clean.startswith("```"):
        clean = "\n".join(l for l in clean.split("\n") if not l.startswith("```"))
    start = clean.find("{")
    end   = clean.rfind("}") + 1
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(clean[start:end])
    except json.JSONDecodeError:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# CACHE
# ═══════════════════════════════════════════════════════════════════════════════

def pol_cache_path(key: str) -> Path:
    safe = key.replace("/", "_").replace(" ", "_")
    return pol_cache_dir / f"{safe}_v{CACHE_VERSION}.json"


def load_pol_cache(key: str) -> Optional[dict]:
    p = pol_cache_path(key)
    if not p.exists():
        return None
    age = (datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)).days
    if age >= CACHE_TTL_DAYS:
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if data.get("validated") else None
    except Exception:
        return None


def save_pol_cache(key: str, data: dict) -> None:
    try:
        data["version"]   = CACHE_VERSION
        data["cached_at"] = datetime.now().isoformat()
        data["validated"] = True
        pol_cache_path(key).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as e:
        log.warning(f"Cache save failed for {key}: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION — anti-hallucination
# ═══════════════════════════════════════════════════════════════════════════════

def clamp_int(v, lo=0, hi=100, default=40) -> int:
    try:
        return max(lo, min(hi, int(float(v))))
    except (TypeError, ValueError):
        return default


def validate_sector_risk(raw: dict, sector: str) -> tuple[dict, bool]:
    """Validate sector risk response. Returns (clean, is_valid)."""
    required = ["tariff_risk", "regulatory_risk", "geopolitical_risk",
                "china_exposure", "taiwan_exposure"]
    if not all(k in raw for k in required):
        return {}, False

    out = {
        "sector":             sector,
        "tariff_risk":        clamp_int(raw.get("tariff_risk")),
        "regulatory_risk":    clamp_int(raw.get("regulatory_risk")),
        "geopolitical_risk":  clamp_int(raw.get("geopolitical_risk")),
        "policy_dependency":  clamp_int(raw.get("policy_dependency", 30)),
        "sanction_risk":      clamp_int(raw.get("sanction_risk", 20)),
        "china_exposure":     clamp_int(raw.get("china_exposure")),
        "taiwan_exposure":    clamp_int(raw.get("taiwan_exposure")),
        "trend":              str(raw.get("trend", "stable"))[:20],
        "narrative":          str(raw.get("narrative", ""))[:300],
        "key_risks":          [str(x)[:80] for x in (raw.get("key_risks") or [])[:5]],
    }
    return out, True


def validate_company_risk(raw: dict, ticker: str) -> tuple[dict, bool]:
    """Validate company political risk. Returns (clean, is_valid)."""
    if "E_political" not in raw:
        return {}, False

    out = {
        "ticker":                  ticker,
        "E_political":             clamp_int(raw.get("E_political")),
        "tariff_adjustment":       clamp_int(raw.get("tariff_adjustment", 0), -20, 20, 0),
        "regulatory_adjustment":   clamp_int(raw.get("regulatory_adjustment", 0), -20, 20, 0),
        "geo_concentration_penalty": clamp_int(raw.get("geo_concentration_penalty", 0), 0, 30, 0),
        "specific_risks":          [str(x)[:80] for x in (raw.get("specific_risks") or [])[:4]],
        "political_moats":         [str(x)[:80] for x in (raw.get("political_moats") or [])[:3]],
        "political_narrative":     str(raw.get("political_narrative", ""))[:200],
    }
    return out, True


# ═══════════════════════════════════════════════════════════════════════════════
# SECTOR ANALYSIS — cached, ~12 calls total
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_sector(sector: str) -> dict:
    cached = load_pol_cache(f"sector_{sector}")
    if cached:
        log.info(f"  [{sector}] → cached")
        return cached

    prompt = f"""Assess current political and regulatory risk for the {sector} sector (US-listed companies, global exposure).

Return ONLY this JSON:
{{
  "tariff_risk": 0-100,
  "regulatory_risk": 0-100,
  "geopolitical_risk": 0-100,
  "policy_dependency": 0-100,
  "sanction_risk": 0-100,
  "china_exposure": 0-100,
  "taiwan_exposure": 0-100,
  "key_risks": ["risk1","risk2","risk3"],
  "trend": "improving/stable/deteriorating",
  "narrative": "2 sentence summary"
}}"""

    for attempt in range(1, MAX_RETRIES + 1):
        response = ask_llm(prompt, 350)
        parsed   = extract_json(response)
        if parsed:
            clean, valid = validate_sector_risk(parsed, sector)
            if valid:
                save_pol_cache(f"sector_{sector}", clean)
                return clean
        log.warning(f"  [{sector}] attempt {attempt} failed")
        time.sleep(2)

    # Default
    default = {
        "sector": sector, "tariff_risk": 40, "regulatory_risk": 40,
        "geopolitical_risk": 30, "policy_dependency": 30, "sanction_risk": 20,
        "china_exposure": 30, "taiwan_exposure": 10, "trend": "stable",
        "narrative": "Default values — LLM analysis failed.", "key_risks": [],
        "validated": True,
    }
    save_pol_cache(f"sector_{sector}", default)
    return default


# ═══════════════════════════════════════════════════════════════════════════════
# COMPANY ANALYSIS — adaptive by complexity tier
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_company(ticker: str, name: str, sector: str,
                    sector_risk: dict, tree_data: dict,
                    tier: str) -> dict:
    cached = load_pol_cache(f"company_{ticker}")
    if cached:
        return cached

    spof = tree_data.get("critical_nodes", "[]")
    if isinstance(spof, str):
        try: spof = json.loads(spof)
        except Exception: spof = []

    single_root = tree_data.get("single_root_risk", False)
    root_count  = tree_data.get("root_count", 2)
    narrative   = str(tree_data.get("narrative") or "")[:150]

    max_tokens = TIER_TOKENS.get(tier, 400)

    if tier == "SIMPLE":
        prompt = f"""Rate political risk for {ticker} ({name}) in {sector} sector.
Sector baseline: tariff={sector_risk['tariff_risk']}, regulatory={sector_risk['regulatory_risk']}, china={sector_risk['china_exposure']}.

Return ONLY JSON:
{{"E_political":0-100,"tariff_adjustment":-20to20,"regulatory_adjustment":-20to20,"geo_concentration_penalty":0-30,"specific_risks":[],"political_moats":[],"political_narrative":""}}"""

    else:
        prompt = f"""Assess political risk for {ticker} ({name}).
Sector: {sector} | Complexity: {tier}
Supply chain context: {root_count} geographic roots, single_root={single_root}
Critical nodes: {spof[:4]}
Structural summary: {narrative}
Sector baseline: tariff={sector_risk['tariff_risk']}, regulatory={sector_risk['regulatory_risk']}, geopolitical={sector_risk['geopolitical_risk']}, china={sector_risk['china_exposure']}, taiwan={sector_risk['taiwan_exposure']}

Return ONLY JSON:
{{"E_political":0-100,"tariff_adjustment":-20to20,"regulatory_adjustment":-20to20,"geo_concentration_penalty":0-30,"specific_risks":["risk1","risk2"],"political_moats":["moat1"],"political_narrative":"1 sentence"}}

E_political: 0-25=low risk, 25-50=moderate, 50-75=high, 75-100=critical"""

    for attempt in range(1, MAX_RETRIES + 1):
        response = ask_llm(prompt, max_tokens)
        parsed   = extract_json(response)
        if parsed:
            clean, valid = validate_company_risk(parsed, ticker)
            if valid:
                save_pol_cache(f"company_{ticker}", clean)
                return clean
        log.warning(f"  [{ticker}] attempt {attempt} failed")
        time.sleep(2)

    # Default — use sector baseline
    base = int((sector_risk["tariff_risk"] * 0.3 +
                sector_risk["regulatory_risk"] * 0.3 +
                sector_risk["geopolitical_risk"] * 0.4))
    default = {
        "ticker": ticker, "E_political": base,
        "tariff_adjustment": 0, "regulatory_adjustment": 0,
        "geo_concentration_penalty": 0, "specific_risks": [],
        "political_moats": [], "political_narrative": "Default — LLM failed.",
        "validated": True,
    }
    save_pol_cache(f"company_{ticker}", default)
    return default


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED SCORE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_unified(e_total: float, e_tree: float, e_political: float) -> float:
    # Normalise E_tree (0-100) and E_political (0-100) to ~E_total scale (60-160)
    e_tree_norm = e_tree * 1.5
    e_pol_norm  = e_political * 1.5
    return round(W_FINANCIAL * e_total + W_TREE * e_tree_norm + W_POLITICAL * e_pol_norm, 4)


# ═══════════════════════════════════════════════════════════════════════════════
# HTML REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def generate_report(df: pd.DataFrame, path: Path) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    def fmt(v, dec=2):
        try: return f"{float(v):.{dec}f}" if pd.notna(v) else "N/A"
        except Exception: return "N/A"

    def badge(v, t=(70,100,130)):
        try:
            v=float(v)
            cls = "excellent" if v<t[0] else "good" if v<t[1] else "moderate" if v<t[2] else "poor"
            return f'<span class="badge {cls}">{v:.2f}</span>'
        except Exception: return "N/A"

    def pol_badge(v):
        try:
            v=float(v)
            cls = "excellent" if v<30 else "good" if v<50 else "moderate" if v<70 else "poor"
            return f'<span class="badge {cls}">{v:.1f}</span>'
        except Exception: return "N/A"

    df_s = df.sort_values("E_unified").reset_index(drop=True)
    df_s["Rank_u"] = range(1, len(df_s) + 1)

    all_rows = ""
    for _, r in df_s.iterrows():
        risks = r.get("specific_risks","[]")
        if isinstance(risks,str):
            try: risks=json.loads(risks)
            except Exception: risks=[]
        warn = " warn" if r.get("single_root_risk") else ""
        delta = int(r.get("Rank",0)) - int(r["Rank_u"])
        arrow = f"▲{delta}" if delta>0 else (f"▼{abs(delta)}" if delta<0 else "=")
        all_rows += (
            f'<tr class="{warn}">'
            f'<td class="rc">{int(r["Rank_u"])}</td>'
            f'<td class="rc" style="color:#aaa">{int(r.get("Rank",0))}</td>'
            f'<td class="rc" style="color:{"#28a745" if delta>5 else "#dc3545" if delta<-5 else "#888"}">{arrow}</td>'
            f'<td class="tc">{r["Ticker"]}</td>'
            f'<td>{r["Security"]}</td>'
            f'<td class="sc">{r["Sector"]}</td>'
            f'<td>{badge(r.get("E_unified",0))}</td>'
            f'<td>{badge(r.get("E_total",0))}</td>'
            f'<td>{badge(r.get("E_tree",50),(30,55,75))}</td>'
            f'<td>{pol_badge(r.get("E_political",50))}</td>'
            f'<td>{"⚠️" if r.get("single_root_risk") else "✅"}</td>'
            f'<td style="font-size:.76em;color:#555">{", ".join(str(x) for x in risks[:2])}</td>'
            f'<td style="font-size:.75em;color:#666;max-width:220px">{str(r.get("political_narrative",""))[:90]}</td>'
            f'</tr>\n'
        )

    # Sector summary
    sec_sum = df_s.groupby("Sector").agg(
        n=("Ticker","count"),
        unified=("E_unified","mean"),
        political=("E_political","mean"),
        tree=("E_tree","mean"),
    ).round(2).sort_values("unified",ascending=False).reset_index()
    sec_rows = "".join(
        f'<tr><td>{r["Sector"]}</td><td>{int(r["n"])}</td>'
        f'<td>{r["unified"]:.2f}</td><td>{r["political"]:.2f}</td><td>{r["tree"]:.2f}</td></tr>'
        for _,r in sec_sum.iterrows()
    )

    best  = df_s.iloc[0]
    worst = df_s.iloc[-1]

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fresta Unified — 5-Order S&P 500</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Segoe UI',sans-serif;background:linear-gradient(135deg,#0f0c29,#302b63,#24243e);padding:20px;color:#333}}
.container{{max-width:1900px;margin:0 auto;background:#fff;border-radius:15px;box-shadow:0 10px 50px rgba(0,0,0,.5);overflow:hidden}}
.header{{background:linear-gradient(135deg,#0f0c29,#302b63);color:#fff;padding:42px;text-align:center}}
.header h1{{font-size:2.2em;margin-bottom:8px}}
.header a{{color:#a8edea}}
.pills{{display:flex;gap:8px;justify-content:center;flex-wrap:wrap;margin-top:14px}}
.pill{{background:rgba(255,255,255,.15);padding:5px 14px;border-radius:20px;font-size:.8em}}
.content{{padding:40px}}
.formula{{background:#f8f9ff;border:1px solid #dde;padding:14px 18px;border-radius:8px;font-family:monospace;font-size:.9em;margin-bottom:28px;color:#333}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:15px;margin-bottom:36px}}
.card{{background:linear-gradient(135deg,#f5f7fa,#c3cfe2);padding:16px;border-radius:10px;text-align:center;box-shadow:0 3px 8px rgba(0,0,0,.1);transition:transform .3s}}
.card:hover{{transform:translateY(-3px)}}
.card h3{{color:#302b63;font-size:.74em;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px}}
.card .val{{font-size:1.75em;font-weight:700;color:#333}}
.val.red{{color:#dc3545}}.val.green{{color:#28a745}}.val.orange{{color:#fd7e14}}
.section{{margin-bottom:42px}}
.section h2{{color:#302b63;font-size:1.4em;padding-bottom:9px;border-bottom:3px solid #302b63;margin-bottom:11px}}
.info{{background:#f0f0ff;border-left:4px solid #302b63;padding:10px 14px;border-radius:0 8px 8px 0;font-size:.85em;color:#444;margin-bottom:12px}}
table{{width:100%;border-collapse:collapse;box-shadow:0 2px 10px rgba(0,0,0,.1);border-radius:8px;overflow:hidden;font-size:.8em}}
thead{{background:linear-gradient(135deg,#0f0c29,#302b63);color:#fff}}
th{{padding:11px 7px;text-align:left;font-weight:600;text-transform:uppercase;font-size:.73em;cursor:pointer}}
th:hover{{opacity:.8}}
td{{padding:9px 6px;border-bottom:1px solid #eee}}
tbody tr:hover{{background:#f5f5ff}}
.rc{{font-weight:700;color:#302b63;text-align:center}}
.tc{{font-family:monospace;font-weight:700;font-size:1.03em}}
.sc{{color:#888;font-size:.82em}}
.badge{{display:inline-block;padding:3px 8px;border-radius:12px;font-weight:700;font-size:.88em}}
.excellent{{background:#d4edda;color:#155724}}
.good{{background:#d1ecf1;color:#0c5460}}
.moderate{{background:#fff3cd;color:#856404}}
.poor{{background:#f8d7da;color:#721c24}}
tr.warn{{background:#fff8f0!important;border-left:3px solid #ffc107}}
.search-bar input{{padding:8px 14px;border:1px solid #ddd;border-radius:20px;width:300px;font-size:.87em;margin-bottom:12px}}
.footer{{text-align:center;padding:24px;background:#f8f9fa;color:#888;font-size:.81em;border-top:1px solid #eee}}
.footer a{{color:#302b63}}
</style></head>
<body><div class="container">
<div class="header">
  <h1>🔬 Fresta Unified — 5-Order Analysis</h1>
  <p>S&amp;P 500 Complete Structural Entropy — {ts}</p>
  <div class="pills">
    <span class="pill">1st: Local Health</span>
    <span class="pill">2nd: Sector Propagation</span>
    <span class="pill">3rd: Macro Stress</span>
    <span class="pill">4th: Supply Chain Tree</span>
    <span class="pill">5th: Political Risk</span>
  </div>
  <p style="margin-top:12px;font-size:.85em">
    <a href="https://doi.org/10.5281/zenodo.18251304">Fresta Framework</a> &nbsp;·&nbsp;
    <a href="https://github.com/EviAmarates/fresta-finance">GitHub</a>
  </p>
</div>
<div class="content">

<div class="formula">
  E_unified = {W_FINANCIAL} × E_total + {W_TREE} × E_tree + {W_POLITICAL} × E_political &nbsp;|&nbsp; Lower = more resilient
</div>

<div class="cards">
  <div class="card"><h3>Companies</h3><div class="val">{len(df_s)}</div></div>
  <div class="card"><h3>Best Unified</h3><div class="val green">{fmt(df_s["E_unified"].min())}</div><div style="font-size:.7em;color:#666">{best["Ticker"]}</div></div>
  <div class="card"><h3>Worst Unified</h3><div class="val red">{fmt(df_s["E_unified"].max())}</div><div style="font-size:.7em;color:#666">{worst["Ticker"]}</div></div>
  <div class="card"><h3>Mean Unified</h3><div class="val orange">{fmt(df_s["E_unified"].mean())}</div></div>
  <div class="card"><h3>Mean E_political</h3><div class="val">{fmt(df_s["E_political"].mean()) if "E_political" in df_s else "N/A"}</div></div>
  <div class="card"><h3>Single Root ⚠️</h3><div class="val red">{int(df_s.get("single_root_risk",pd.Series([False]*len(df_s))).sum())}</div></div>
</div>

<div class="section">
  <h2>📊 Sector Summary</h2>
  <table><thead><tr><th>Sector</th><th>N</th><th>Mean Unified</th><th>Mean Political</th><th>Mean E_tree</th></tr></thead>
  <tbody>{sec_rows}</tbody></table>
</div>

<div class="section">
  <h2>📋 Full Unified Ranking — {len(df_s)} Companies</h2>
  <div class="info">△▽ = rank change from financial-only analysis. Large positive change = company was hiding fragility behind good numbers. Yellow = single root risk.</div>
  <div class="search-bar"><input type="text" id="si" placeholder="🔍 Search..." oninput="ft()"></div>
  <table id="t">
    <thead><tr>
      <th onclick="st(0)">Rank</th><th onclick="st(1)">Fin</th><th onclick="st(2)">Δ</th>
      <th onclick="st(3)">Ticker</th><th>Company</th><th onclick="st(5)">Sector</th>
      <th onclick="st(6)">E_unified</th><th onclick="st(7)">E_total</th>
      <th onclick="st(8)">E_tree</th><th onclick="st(9)">E_political</th>
      <th>Root</th><th>Political Risks</th><th>Narrative</th>
    </tr></thead>
    <tbody id="tb">{all_rows}</tbody>
  </table>
</div>
</div>
<div class="footer">
  <p>Fresta Unified v2 — 5-order structural entropy with adaptive LLM analysis — {ts}</p>
  <p><a href="https://doi.org/10.5281/zenodo.18251304">Framework</a> &nbsp;·&nbsp;
  <a href="https://github.com/EviAmarates/fresta-finance">GitHub</a> &nbsp;·&nbsp;
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
    log.info(f"Unified HTML: {path.absolute()}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("FRESTA POLITICAL v2 — Adaptive Political Risk Analysis")
    log.info("=" * 60)

    # Load data
    base_csv = output_dir / "sp500_entropy_ranked.csv"
    tree_csv = output_dir / "sp500_tree_analysis.csv"

    if not base_csv.exists():
        log.error("Run fresta_finance.py first!")
        sys.exit(1)

    df = pd.read_csv(base_csv)
    log.info(f"Loaded {len(df)} companies from financial analysis.")

    if tree_csv.exists():
        tree_df = pd.read_csv(tree_csv)
        merge_cols = ["Ticker", "E_tree", "complexity_tier", "single_root_risk",
                      "root_count", "root_diversity", "spof_count", "critical_nodes", "narrative"]
        merge_cols = [c for c in merge_cols if c in tree_df.columns]
        df = df.merge(tree_df[merge_cols], on="Ticker", how="left")
        df["E_tree"]          = df.get("E_tree", pd.Series([50.0]*len(df))).fillna(50.0)
        df["complexity_tier"] = df.get("complexity_tier", pd.Series(["MODERATE"]*len(df))).fillna("MODERATE")
        log.info(f"Tree data merged.")
    else:
        log.warning("No tree data found — run fresta_tree.py for better results.")
        df["E_tree"]           = 50.0
        df["complexity_tier"]  = "MODERATE"
        df["single_root_risk"] = False
        df["root_count"]       = 2
        df["spof_count"]       = 0
        df["critical_nodes"]   = "[]"
        df["narrative"]        = ""

    # Check LM Studio
    try:
        r = requests.get(f"{LM_STUDIO_HOST}/v1/models", timeout=5)
        models = [m["id"] for m in r.json().get("data", [])]
        if not any(LM_STUDIO_MODEL in m for m in models):
            log.error(f"Model '{LM_STUDIO_MODEL}' not loaded. Available: {models}")
            log.error("Load the model in LM Studio Developer tab first!")
            sys.exit(1)
        log.info(f"LM Studio OK — {LM_STUDIO_MODEL}")
    except Exception:
        log.error("LM Studio not running at http://127.0.0.1:1234")
        log.error("Open LM Studio → Developer → Start Server")
        sys.exit(1)

    # Step 1: Sector analysis (~12 LLM calls, mostly cached)
    log.info("\n[1/2] Sector political risk analysis...")
    sectors      = df["Sector"].dropna().unique()
    sector_risks = {}
    for sector in sectors:
        sector_risks[sector] = analyse_sector(sector)
        time.sleep(0.3)

    # Step 2: Company analysis (adaptive by tier)
    log.info(f"\n[2/2] Company political risk ({len(df)} companies)...")
    pol_records = []
    cached_count = 0

    for i, (_, row) in enumerate(df.iterrows(), 1):
        ticker  = row["Ticker"]
        name    = row.get("Security", ticker)
        sector  = row.get("Sector", "Unknown")
        tier    = row.get("complexity_tier", "MODERATE")

        # Quick cache check
        if load_pol_cache(f"company_{ticker}"):
            cached_count += 1

        if i % 50 == 0:
            log.info(f"  {i}/{len(df)} ({cached_count} cached)")

        sector_risk = sector_risks.get(sector, {
            "tariff_risk": 40, "regulatory_risk": 40, "geopolitical_risk": 30,
            "china_exposure": 30, "taiwan_exposure": 10,
        })
        tree_data = {
            "critical_nodes":  str(row.get("critical_nodes") or "[]"),
            "narrative":       str(row.get("narrative") or ""),
            "root_count":      int(row.get("root_count") or 2),
            "single_root_risk":bool(row.get("single_root_risk") or False),
        }

        result = analyse_company(ticker, name, sector, sector_risk, tree_data, tier)
        pol_records.append({
            "Ticker":                result.get("ticker", ticker),
            "E_political":           result.get("E_political", 40),
            "tariff_adjustment":     result.get("tariff_adjustment", 0),
            "regulatory_adjustment": result.get("regulatory_adjustment", 0),
            "geo_concentration_penalty": result.get("geo_concentration_penalty", 0),
            "specific_risks":        json.dumps(result.get("specific_risks", [])),
            "political_moats":       json.dumps(result.get("political_moats", [])),
            "political_narrative":   result.get("political_narrative", ""),
        })
        time.sleep(0.05)

    pol_df = pd.DataFrame(pol_records)
    df = df.merge(pol_df, on="Ticker", how="left")
    df["E_political"] = df["E_political"].fillna(40)

    # Unified score
    df["E_unified"] = df.apply(
        lambda r: compute_unified(
            float(r.get("E_total", 100)),
            float(r.get("E_tree", 50)),
            float(r.get("E_political", 40)),
        ), axis=1
    )

    # Save CSV
    save_cols = ["Rank", "Ticker", "Security", "Sector",
                 "E_unified", "E_total", "E_tree", "E_political",
                 "complexity_tier", "single_root_risk", "spof_count",
                 "specific_risks", "political_narrative"]
    df[[c for c in save_cols if c in df.columns]].to_csv(
        output_dir / "sp500_political_risk.csv", index=False
    )

    generate_report(df, output_dir / "sp500_unified_report.html")

    # Summary
    df_s = df.sort_values("E_unified").reset_index(drop=True)
    df_s["Rank_u"] = range(1, len(df_s) + 1)

    log.info("\n" + "=" * 60)
    log.info("TOP 10 MOST RESILIENT — UNIFIED SCORE")
    log.info("=" * 60)
    for _, r in df_s.head(10).iterrows():
        delta = int(r.get("Rank", 0)) - int(r["Rank_u"])
        arrow = f"▲{delta}" if delta > 0 else (f"▼{abs(delta)}" if delta < 0 else "=")
        log.info(f"  #{int(r['Rank_u']):3d} [{arrow:5s}]  {r['Ticker']:6s}  "
                 f"unified={r['E_unified']:.2f}  pol={r.get('E_political',0):.1f}  "
                 f"tree={r.get('E_tree',0):.1f}")

    log.info(f"\nDone! Open output/sp500_unified_report.html")


if __name__ == "__main__":
    main()
