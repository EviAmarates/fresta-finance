# -*- coding: utf-8 -*-
"""
FRESTA FINANCE — S&P 500 Structural Entropy Ranking
=====================================================
Applies the Fresta Lens Framework to financial markets.
Ranks every S&P 500 company by structural entropy across three orders:

    E_total = E0 + E_upstream + E_inherited

Where:
  E0          = 1st order — local financial health (recycling, fragility, noise)
  E_upstream  = 2nd order — propagated entropy through sector + price-correlation graph
  E_inherited = 3rd order — systemic stress delta from macro infrastructure

Lower score = less structural entropy = more coherent, resilient company.

Usage:
  python fresta_finance.py

Output (in ./output/):
  - sp500_entropy_ranked.csv   — full ranked table
  - sp500_entropy_report.html  — interactive HTML report (all 500 rows, sortable)

Dependencies:
  pip install pandas numpy yfinance requests
"""

import sys
import time
import logging
import random
from datetime import datetime
from pathlib import Path
from io import StringIO
from collections import defaultdict

import pandas as pd
import numpy as np

try:
    import yfinance as yf
except ImportError:
    print("Missing: pip install yfinance")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("Missing: pip install requests")
    sys.exit(1)

# ── Windows encoding fix ──────────────────────────────────────────────────────
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# 1st order block weights (must sum to 1.0)
# NOTE: dilution block removed — cap/shares is just stock price, not real dilution.
# Future work: compute % change in sharesOutstanding over 4 quarters.
W_RECYCLING  = 0.45   # Profit margins, FCF, ROE
W_FRAGILITY  = 0.35   # Debt ratios, liquidity
W_NOISE      = 0.20   # Volatility, drawdown

# 2nd order propagation
UPSTREAM_ITERATIONS  = 2
UPSTREAM_ALPHA       = 0.30   # Upstream influence weight per iteration
CONCENTRATION_WEIGHT = 0.10   # Herfindahl concentration penalty
CYCLE_WEIGHT         = 0.05   # Circular dependency penalty
CORR_THRESHOLD       = 0.70   # Minimum price correlation to add as dependency
CORR_WEIGHT          = 0.20   # Weight of correlation-based edges

# 3rd order systemic stress
INFRA_BETA           = 0.30   # Infrastructure entropy inheritance weight
INFRA_GAMMA          = 0.15   # Saturated-base penalty weight
INFRA_CRITICAL       = 65.0   # Saturation threshold (score units)

# Download settings
CHUNK_SIZE       = 15
MAX_DEPS         = 10
MIN_DEP_WEIGHT   = 0.03
PRICE_PERIOD     = "6mo"
CACHE_TTL_DAYS   = 7

# ── Output & logging ──────────────────────────────────────────────────────────
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(output_dir / "run.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# In-memory caches
_price_cache   = {}
_virtual_cache = {}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA ACQUISITION
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_sp500_tickers() -> pd.DataFrame:
    """Return DataFrame with columns: Symbol, Security, Sector."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; FrestaFinance/1.0)"}
    sources = [
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv",
    ]
    for url in sources:
        try:
            log.info(f"Fetching tickers from: {url}")
            r = requests.get(url, headers=headers, timeout=30)
            r.raise_for_status()
            if "wikipedia" in url:
                tables = pd.read_html(StringIO(r.text), attrs={"id": "constituents"})
                df = tables[0] if tables else pd.read_html(StringIO(r.text))[0]
            else:
                df = pd.read_csv(StringIO(r.text))

            df.columns = [c.strip().lower() for c in df.columns]
            sym_col  = next((c for c in df.columns if c in ["symbol", "ticker"]), None)
            name_col = next((c for c in df.columns if c in ["security", "name"]), None)
            sec_col  = next((c for c in df.columns if "sector" in c), None)

            if not sym_col:
                continue

            out = pd.DataFrame()
            out["Symbol"]   = df[sym_col].astype(str).str.strip().str.replace(".", "-", regex=False)
            out["Security"] = df[name_col].astype(str).str.strip() if name_col else out["Symbol"]
            out["Sector"]   = df[sec_col].astype(str).str.strip() if sec_col else "Unknown"
            out = out[out["Symbol"].notna() & (out["Symbol"] != "")]
            log.info(f"Loaded {len(out)} tickers.")
            return out.reset_index(drop=True)
        except Exception as e:
            log.warning(f"Source failed: {e}")

    local = Path("sp500_tickers.txt")
    if local.exists():
        tickers = [l.strip().replace(".", "-") for l in local.read_text().splitlines() if l.strip()]
        log.info(f"Loaded {len(tickers)} tickers from local file.")
        return pd.DataFrame({"Symbol": tickers, "Security": tickers, "Sector": "Unknown"})

    log.error("Could not fetch tickers. Create sp500_tickers.txt with one ticker per line.")
    sys.exit(1)


def _load_disk_cache(ticker: str) -> pd.Series | None:
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(exist_ok=True)
    path = cache_dir / f"{ticker}.csv"
    if not path.exists():
        return None
    age = (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)).days
    if age >= CACHE_TTL_DAYS:
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df["Close"] if "Close" in df.columns else None
    except Exception:
        return None


def _save_disk_cache(ticker: str, prices: pd.Series) -> None:
    try:
        path = output_dir / "cache" / f"{ticker}.csv"
        pd.DataFrame({"Close": prices}).to_csv(path)
    except Exception:
        pass


def download_prices(tickers: list[str]) -> dict[str, pd.Series]:
    """Download closing prices with memory cache, disk cache, and chunked downloads."""
    result = {}

    missing = []
    for t in tickers:
        if t in _price_cache:
            result[t] = _price_cache[t]
        else:
            cached = _load_disk_cache(t)
            if cached is not None:
                result[t] = cached
                _price_cache[t] = cached
            else:
                missing.append(t)

    if not missing:
        log.info(f"All {len(tickers)} tickers loaded from cache.")
        return result

    log.info(f"Downloading {len(missing)} tickers in chunks of {CHUNK_SIZE}...")
    chunks = [missing[i:i + CHUNK_SIZE] for i in range(0, len(missing), CHUNK_SIZE)]

    for idx, chunk in enumerate(chunks, 1):
        log.info(f"  Chunk {idx}/{len(chunks)} ({len(chunk)} tickers)...")
        for t in chunk:
            try:
                hist = yf.Ticker(t).history(period=PRICE_PERIOD, auto_adjust=True)
                if not hist.empty and "Close" in hist.columns:
                    prices = hist["Close"].dropna()
                    if len(prices) > 20:
                        result[t] = prices
                        _price_cache[t] = prices
                        _save_disk_cache(t, prices)
                time.sleep(0.05)
            except Exception:
                pass

    log.info(f"Download complete: {len(result)}/{len(tickers)} tickers.")
    return result


def get_fundamentals(ticker: str) -> dict:
    """Fetch key financial fundamentals from Yahoo Finance."""
    try:
        info = yf.Ticker(ticker).get_info()
        return {
            "profitMargins":    info.get("profitMargins"),
            "operatingMargins": info.get("operatingMargins"),
            "freeCashflow":     info.get("freeCashflow"),
            "returnOnEquity":   info.get("returnOnEquity"),
            "debtToEquity":     info.get("debtToEquity"),
            "currentRatio":     info.get("currentRatio"),
            "marketCap":        info.get("marketCap"),
            "sector":           info.get("sector", "Unknown"),
        }
    except Exception:
        return {}


def get_virtual_score(node: str) -> float:
    """Score for macro virtual nodes: SPY, QQQ, VIX, MACRO, RATES."""
    if node in _virtual_cache:
        return _virtual_cache[node]
    symbol_map = {"SPY": "SPY", "QQQ": "QQQ", "VIX": "^VIX", "RATES": "TLT"}
    sym = symbol_map.get(node)
    score = 50.0
    if sym:
        try:
            hist = yf.Ticker(sym).history(period="2y", auto_adjust=True)
            if not hist.empty:
                ret = hist["Close"].pct_change().dropna()
                vol = ret.std() * np.sqrt(252)
                dd  = abs((hist["Close"] / hist["Close"].expanding().max() - 1).min())
                score = min((vol * 0.6 + dd * 0.4) * 100, 100)
        except Exception:
            pass
    if node == "MACRO":
        score = get_virtual_score("SPY") * 0.7 + get_virtual_score("VIX") * 0.3
    _virtual_cache[node] = score
    return score


# ═══════════════════════════════════════════════════════════════════════════════
# ENTROPY CALCULATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def shannon_entropy(series: pd.Series) -> float:
    vals = series.dropna()
    if len(vals) < 10:
        return np.nan
    n_bins = max(2, int(np.ceil(np.log2(len(vals)) + 1)))
    hist, _ = np.histogram(vals, bins=n_bins)
    hist = hist[hist > 0]
    probs = hist / hist.sum()
    return float(-np.sum(probs * np.log2(probs + 1e-10)))


def tail_risk(returns: pd.Series) -> float:
    """CVaR 5% — average loss in worst 5% of days."""
    r = returns.dropna()
    if len(r) < 20:
        return np.nan
    idx = max(1, int(len(r) * 0.05))
    cvar = float(abs(np.mean(np.sort(r)[:idx])))
    # Add skew penalty: negative skew amplifies tail risk
    skew = float(r.skew()) if len(r) > 10 else 0.0
    skew_penalty = max(0, -skew) * 0.1  # only penalise negative skew
    return cvar + skew_penalty


def price_metrics(prices: pd.Series) -> dict:
    """Compute volatility, max drawdown, Shannon entropy, tail risk."""
    if prices is None or len(prices) < 20:
        return {}
    ret = prices.pct_change().dropna()
    cum = (1 + ret).cumprod()
    dd  = abs(((cum - cum.expanding().max()) / cum.expanding().max()).min())
    return {
        "volatility":   float(ret.std() * np.sqrt(252)),
        "max_drawdown": float(dd),
        "shannon":      shannon_entropy(ret),
        "tail_risk":    tail_risk(ret),
    }


def percentile_rank(value, series: pd.Series) -> float:
    """Return 0-1 percentile of value within series."""
    s = series.dropna()
    if pd.isna(value) or len(s) == 0:
        return 0.5
    return float((s < value).sum() / len(s))


def compute_E0(row: dict, all_data: pd.DataFrame) -> float:
    """
    1st order entropy score (0–100). Lower = better.
    Three blocks: recycling capacity, structural fragility, noise.
    Dilution block removed — cap/shares is just stock price, not dilution.
    """
    def score(col, invert=False):
        v = row.get(col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return 0.5
        p = percentile_rank(v, all_data[col]) if col in all_data.columns else 0.5
        return (1 - p) if invert else p

    recycling = np.mean([
        score("profitMargins",    invert=True),
        score("operatingMargins", invert=True),
        score("freeCashflow",     invert=True),
        score("returnOnEquity",   invert=True),
    ])
    fragility = np.mean([
        score("debtToEquity"),
        score("currentRatio", invert=True),
    ])
    noise = np.mean([
        score("volatility"),
        score("max_drawdown"),
        score("tail_risk"),   # added: CVaR with skew penalty
    ])

    e0 = (recycling * W_RECYCLING + fragility * W_FRAGILITY + noise * W_NOISE)
    return round(e0 * 100, 4)


def compute_price_correlations(prices: dict[str, pd.Series],
                                tickers: list[str]) -> dict[str, list[tuple]]:
    """
    Build correlation-based edges: if two tickers have price correlation > CORR_THRESHOLD,
    add a weighted dependency edge. Uses only tickers with sufficient price history.
    Groups by sector first to limit computation (O(n) within sector rather than O(n²)).
    """
    corr_deps = defaultdict(list)

    # Build price matrix per sector for efficiency
    # We use the full set but only check within-sector pairs to keep it tractable
    # For cross-sector we rely on the sector graph
    try:
        valid = {t: p for t, p in prices.items() if p is not None and len(p) > 40}
        if not valid:
            return corr_deps

        # Align all series to common index
        price_df = pd.DataFrame(valid).ffill().dropna(axis=1, thresh=40)
        ret_df   = price_df.pct_change().dropna()

        if ret_df.shape[1] < 2:
            return corr_deps

        corr_matrix = ret_df.corr()

        for t in corr_matrix.columns:
            row = corr_matrix[t]
            # Take tickers with correlation above threshold, excluding self
            high_corr = row[(row >= CORR_THRESHOLD) & (row.index != t)]
            if high_corr.empty:
                continue
            # Normalise correlation values as weights
            total = high_corr.sum()
            for dep, corr_val in high_corr.items():
                weight = (corr_val / total) * CORR_WEIGHT
                if weight >= MIN_DEP_WEIGHT:
                    corr_deps[t].append((dep, weight))

        log.info(f"Correlation edges built: {sum(len(v) for v in corr_deps.values())} total")
    except Exception as e:
        log.warning(f"Correlation computation failed: {e}")

    return corr_deps


def build_dependency_graph(tickers: list[str], fundamentals: dict,
                           prices: dict) -> dict[str, list[tuple]]:
    """
    Build weighted dependency graph combining:
    - Sector peer weights (market-cap weighted)
    - Macro virtual node overlays (SPY, RATES, MACRO)
    - Price correlation edges (correlation > CORR_THRESHOLD)
    """
    sector_groups = defaultdict(list)
    for t in tickers:
        sec = fundamentals.get(t, {}).get("sector", "Unknown")
        sector_groups[sec].append(t)

    # Price correlation edges
    log.info("Computing price correlations for dependency graph...")
    corr_edges = compute_price_correlations(prices, tickers)

    deps = {}
    for t in tickers:
        fund   = fundamentals.get(t, {})
        sector = fund.get("sector", "Unknown")
        peers  = [p for p in sector_groups.get(sector, []) if p != t][:MAX_DEPS]

        # Sector peers weighted by market cap
        peer_caps = [(p, fundamentals.get(p, {}).get("marketCap") or 1.0) for p in peers]
        total_cap = sum(c for _, c in peer_caps) or 1.0
        dep_list  = [(p, c / total_cap * (1 - CORR_WEIGHT)) for p, c in peer_caps
                     if c / total_cap * (1 - CORR_WEIGHT) >= MIN_DEP_WEIGHT]

        # Add correlation edges
        for dep, w in corr_edges.get(t, []):
            dep_list.append((dep, w))

        # Macro overlays
        if "financial" in sector.lower():
            dep_list.append(("RATES", 0.4))
        elif "communication" in sector.lower():
            dep_list.append(("MACRO", 0.4))
        else:
            dep_list.append(("SPY", 0.2))

        # Normalize and cap
        total = sum(w for _, w in dep_list) or 1.0
        dep_list = sorted([(d, w / total) for d, w in dep_list], key=lambda x: -x[1])[:MAX_DEPS]
        deps[t] = dep_list

    return deps


def herfindahl(weights: list[float]) -> float:
    return sum(w ** 2 for w in weights) if weights else 0.0


def propagate_entropy(E0_scores: dict, dep_graph: dict) -> dict[str, float]:
    """
    2nd order: propagate entropy through dependency graph.
    Returns DELTA added at this order (E_upstream = result - E0).
    """
    scores = dict(E0_scores)

    for _ in range(UPSTREAM_ITERATIONS):
        new_scores = {}
        for t, deps in dep_graph.items():
            base     = scores.get(t, 50.0)
            upstream = 0.0
            for dep, w in deps:
                dep_score = (get_virtual_score(dep)
                             if dep in ("SPY", "QQQ", "VIX", "MACRO", "RATES")
                             else scores.get(dep, 50.0))
                upstream += w * dep_score

            conc  = herfindahl([w for _, w in deps])
            cycle = any(t in [d for d, _ in dep_graph.get(dep, [])] for dep, _ in deps)

            new_scores[t] = (base
                             + UPSTREAM_ALPHA * upstream
                             + CONCENTRATION_WEIGHT * conc * 100
                             + (CYCLE_WEIGHT * 100 if cycle else 0))
        scores = new_scores

    # Return delta only
    return {t: scores[t] - E0_scores.get(t, 0) for t in scores}


def compute_E_inherited(E0_scores: dict, E_upstream_deltas: dict,
                        dep_graph: dict) -> dict[str, float]:
    """
    3rd order: systemic stress DELTA from macro infrastructure.
    Operates on full upstream scores (E0 + E_upstream) to assess
    how much macro amplification adds on top.
    """
    full_upstream = {t: E0_scores.get(t, 0) + E_upstream_deltas.get(t, 0)
                     for t in E0_scores}
    result = {}
    for t in E0_scores:
        deps = dep_graph.get(t, [])
        infra_stress = 0.0
        for dep, w in deps:
            dep_score = (get_virtual_score(dep)
                         if dep in ("SPY", "QQQ", "VIX", "MACRO", "RATES")
                         else full_upstream.get(dep, 50.0))
            infra_stress += w * dep_score

        saturation_penalty = max(0, infra_stress - INFRA_CRITICAL) * INFRA_GAMMA
        result[t] = INFRA_BETA * infra_stress + saturation_penalty

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# HTML REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def generate_html_report(df: pd.DataFrame, path: Path) -> None:
    """Generate a rich, self-contained HTML report — all rows, sortable."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    def score_class(v):
        if v < 80:   return "excellent"
        if v < 100:  return "good"
        if v < 120:  return "moderate"
        return "poor"

    def fmt(v, dec=2):
        try:
            return f"{float(v):.{dec}f}" if pd.notna(v) and v is not None else "N/A"
        except Exception:
            return "N/A"

    # Sector stats
    sector_stats = (df.groupby("Sector")["E_total"]
                    .agg(["count", "mean", "median", "std", "min", "max"])
                    .round(2).sort_values("mean", ascending=False).reset_index())

    def make_rows(subset):
        rows = ""
        for _, r in subset.iterrows():
            sc   = score_class(r["E_total"])
            warn = " warn" if r.get("E_inherited", 0) > 20 else ""
            rows += (f'<tr class="{warn}">'
                     f'<td class="rc">{int(r["Rank"])}</td>'
                     f'<td class="tc">{r["Ticker"]}</td>'
                     f'<td class="nc">{r["Security"]}</td>'
                     f'<td class="sc">{r["Sector"]}</td>'
                     f'<td><span class="badge {sc}">{fmt(r["E_total"])}</span></td>'
                     f'<td>{fmt(r["E0"])}</td>'
                     f'<td>{fmt(r["E_upstream"])}</td>'
                     f'<td>{fmt(r["E_inherited"])}</td>'
                     f'<td>{fmt(r.get("volatility"), 3)}</td>'
                     f'<td>{fmt(r.get("max_drawdown"), 3)}</td>'
                     f'</tr>\n')
        return rows

    sector_rows = "".join(
        f'<tr><td>{r["Sector"]}</td><td>{int(r["count"])}</td>'
        f'<td>{r["mean"]:.2f}</td><td>{r["median"]:.2f}</td>'
        f'<td>{"N/A" if pd.isna(r["std"]) else f"{r[chr(115)+chr(116)+chr(100)]:.2f}"}</td>'
        f'<td>{r["min"]:.2f}</td><td>{r["max"]:.2f}</td></tr>'
        for _, r in sector_stats.iterrows()
    )

    best  = df.iloc[0]
    worst = df.iloc[-1]
    all_rows = make_rows(df)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fresta Finance — S&P 500 Entropy Report</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;background:linear-gradient(135deg,#667eea,#764ba2);padding:20px;color:#333}}
.container{{max-width:1600px;margin:0 auto;background:#fff;border-radius:15px;box-shadow:0 10px 40px rgba(0,0,0,.2);overflow:hidden}}
.header{{background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;padding:40px;text-align:center}}
.header h1{{font-size:2.2em;margin-bottom:10px;text-shadow:2px 2px 4px rgba(0,0,0,.3)}}
.header p{{font-size:1.05em;opacity:.9}}
.header a{{color:#ffe}}
.content{{padding:40px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:20px;margin-bottom:40px}}
.card{{background:linear-gradient(135deg,#f5f7fa,#c3cfe2);padding:22px;border-radius:10px;text-align:center;box-shadow:0 4px 6px rgba(0,0,0,.1);transition:transform .3s}}
.card:hover{{transform:translateY(-4px)}}
.card h3{{color:#667eea;font-size:.8em;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px}}
.card .val{{font-size:2em;font-weight:700;color:#333}}
.card .sub{{font-size:.72em;color:#666;margin-top:4px}}
.card .val.green{{color:#28a745}}.card .val.red{{color:#dc3545}}
.section{{margin-bottom:50px}}
.section h2{{color:#667eea;font-size:1.5em;padding-bottom:10px;border-bottom:3px solid #667eea;margin-bottom:10px}}
.info{{background:#f0f4ff;border-left:4px solid #667eea;padding:10px 14px;border-radius:0 8px 8px 0;font-size:.88em;color:#444;margin-bottom:14px}}
.warn-box{{background:#fff8e1;border-left:4px solid #ffc107;padding:10px 14px;border-radius:0 8px 8px 0;font-size:.88em;color:#555;margin-bottom:14px}}
.legend{{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:14px;font-size:.83em}}
.legend span{{display:flex;align-items:center;gap:5px}}
.dot{{width:11px;height:11px;border-radius:3px;display:inline-block}}
table{{width:100%;border-collapse:collapse;box-shadow:0 2px 10px rgba(0,0,0,.1);border-radius:8px;overflow:hidden;font-size:.86em}}
thead{{background:linear-gradient(135deg,#667eea,#764ba2);color:#fff}}
th{{padding:13px 9px;text-align:left;font-weight:600;text-transform:uppercase;font-size:.78em;letter-spacing:.4px;cursor:pointer;user-select:none}}
th:hover{{background:rgba(255,255,255,.15)}}
td{{padding:10px 9px;border-bottom:1px solid #eee}}
tbody tr:hover{{background:#f5f7fa}}
.rc{{font-weight:700;color:#667eea;text-align:center}}
.tc{{font-family:monospace;font-size:1.05em;font-weight:700}}
.nc{{color:#555;max-width:200px}}
.sc{{color:#888;font-size:.86em}}
.badge{{display:inline-block;padding:3px 9px;border-radius:12px;font-weight:700;font-size:.92em}}
.excellent{{background:#d4edda;color:#155724}}
.good{{background:#d1ecf1;color:#0c5460}}
.moderate{{background:#fff3cd;color:#856404}}
.poor{{background:#f8d7da;color:#721c24}}
tr.warn{{background:#fffbf0!important;border-left:3px solid #ffc107}}
.search-bar{{margin-bottom:14px}}
.search-bar input{{padding:8px 14px;border:1px solid #ddd;border-radius:20px;width:300px;font-size:.9em;outline:none}}
.search-bar input:focus{{border-color:#667eea}}
.footer{{text-align:center;padding:28px;background:#f8f9fa;color:#888;font-size:.83em;border-top:1px solid #eee}}
.footer a{{color:#667eea}}
</style>
</head>
<body>
<div class="container">

<div class="header">
  <h1>🔬 Fresta Finance</h1>
  <p>S&amp;P 500 Structural Entropy Report &mdash; {ts}</p>
  <p style="margin-top:8px;font-size:.88em">
    Based on the <a href="https://doi.org/10.5281/zenodo.18251304">Fresta Lens Framework</a>
    &nbsp;·&nbsp; <a href="https://github.com/EviAmarates/fresta-finance">GitHub</a>
  </p>
</div>

<div class="content">

<!-- CARDS -->
<div class="cards">
  <div class="card"><h3>Total Processed</h3><div class="val">{len(df)}</div><div class="sub">S&amp;P 500 companies</div></div>
  <div class="card"><h3>Best Score</h3><div class="val green">{fmt(df['E_total'].min())}</div><div class="sub">{best['Ticker']} &mdash; {best['Security'][:22]}</div></div>
  <div class="card"><h3>Worst Score</h3><div class="val red">{fmt(df['E_total'].max())}</div><div class="sub">{worst['Ticker']} &mdash; {worst['Security'][:22]}</div></div>
  <div class="card"><h3>Mean E_total</h3><div class="val">{fmt(df['E_total'].mean())}</div><div class="sub">S&amp;P 500 average</div></div>
  <div class="card"><h3>Mean E0</h3><div class="val">{fmt(df['E0'].mean())}</div><div class="sub">1st order avg</div></div>
  <div class="card"><h3>Mean E_upstream</h3><div class="val">{fmt(df['E_upstream'].mean())}</div><div class="sub">2nd order avg</div></div>
</div>

<!-- TOP 10 BEST -->
<div class="section">
  <h2>🏆 Top 10 — Most Resilient</h2>
  <div class="info">Lowest total structural entropy. Lower score = more coherent structural position under stress.</div>
  <div class="legend">
    <span><span class="dot" style="background:#d4edda"></span>Excellent (&lt;80)</span>
    <span><span class="dot" style="background:#d1ecf1"></span>Good (80–100)</span>
    <span><span class="dot" style="background:#fff3cd"></span>Moderate (100–120)</span>
    <span><span class="dot" style="background:#f8d7da"></span>High risk (&gt;120)</span>
  </div>
  <table>
    <thead><tr><th>Rank</th><th>Ticker</th><th>Company</th><th>Sector</th><th>E_total</th><th>E0</th><th>E_upstream</th><th>E_inherited</th><th>Volatility</th><th>Max DD</th></tr></thead>
    <tbody>{make_rows(df.head(10))}</tbody>
  </table>
</div>

<!-- TOP 10 WORST -->
<div class="section">
  <h2>⚠️ Top 10 — Most Fragile</h2>
  <div class="info">Highest total structural entropy — multiple amplification layers active simultaneously.</div>
  <table>
    <thead><tr><th>Rank</th><th>Ticker</th><th>Company</th><th>Sector</th><th>E_total</th><th>E0</th><th>E_upstream</th><th>E_inherited</th><th>Volatility</th><th>Max DD</th></tr></thead>
    <tbody>{make_rows(df.tail(10))}</tbody>
  </table>
</div>

<!-- SECTOR STATS -->
<div class="section">
  <h2>📊 Statistics by Sector</h2>
  <div class="info">Aggregated entropy by GICS sector — sorted by mean entropy (highest first).</div>
  <table>
    <thead><tr><th>Sector</th><th>Companies</th><th>Mean</th><th>Median</th><th>Std Dev</th><th>Min</th><th>Max</th></tr></thead>
    <tbody>{sector_rows}</tbody>
  </table>
</div>

<!-- FULL RANKING -->
<div class="section">
  <h2>📋 Full Ranking — All {len(df)} Companies</h2>
  <div class="warn-box">⚠️ Yellow rows have significant inherited macro stress (E_inherited &gt; 20). Click any column header to sort.</div>
  <div class="search-bar"><input type="text" id="searchInput" placeholder="🔍  Search ticker or company..." oninput="filterTable()"></div>
  <table id="mainTable">
    <thead><tr>
      <th onclick="sortTable(0)">Rank</th>
      <th onclick="sortTable(1)">Ticker</th>
      <th>Company</th>
      <th onclick="sortTable(3)">Sector</th>
      <th onclick="sortTable(4)">E_total</th>
      <th onclick="sortTable(5)">E0</th>
      <th onclick="sortTable(6)">E_upstream</th>
      <th onclick="sortTable(7)">E_inherited</th>
      <th onclick="sortTable(8)">Volatility</th>
      <th onclick="sortTable(9)">Max DD</th>
    </tr></thead>
    <tbody id="tableBody">{all_rows}</tbody>
  </table>
</div>

</div><!-- /content -->

<div class="footer">
  <p>Generated by <strong>Fresta Finance</strong> &mdash; {ts}</p>
  <p>S&amp;P 500 Structural Entropy · 3-Order Lens Framework</p>
  <p style="margin-top:8px">
    <a href="https://doi.org/10.5281/zenodo.18251304">Fresta Lens Framework</a> &nbsp;·&nbsp;
    <a href="https://github.com/EviAmarates/fresta-finance">GitHub</a> &nbsp;·&nbsp;
    <a href="https://ko-fi.com/tiagosantos20582">Support on Ko-fi</a>
  </p>
</div>

</div><!-- /container -->

<script>
// Sort
let _dir = {{}};
function sortTable(col) {{
  const tb = document.getElementById('tableBody');
  const rows = [...tb.querySelectorAll('tr')];
  _dir[col] = !_dir[col];
  rows.sort((a,b) => {{
    const va = a.cells[col]?.textContent.trim() ?? '';
    const vb = b.cells[col]?.textContent.trim() ?? '';
    const na = parseFloat(va), nb = parseFloat(vb);
    if (!isNaN(na) && !isNaN(nb)) return _dir[col] ? na-nb : nb-na;
    return _dir[col] ? va.localeCompare(vb) : vb.localeCompare(va);
  }});
  rows.forEach(r => tb.appendChild(r));
}}

// Search
function filterTable() {{
  const q = document.getElementById('searchInput').value.toLowerCase();
  document.querySelectorAll('#tableBody tr').forEach(row => {{
    const text = row.textContent.toLowerCase();
    row.style.display = text.includes(q) ? '' : 'none';
  }});
}}
</script>
</body></html>"""

    path.write_text(html, encoding="utf-8")
    log.info(f"HTML report: {path.absolute()}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("FRESTA FINANCE — S&P 500 Structural Entropy Analysis")
    log.info("=" * 60)

    # 1. Tickers
    tickers_df = fetch_sp500_tickers()
    tickers    = tickers_df["Symbol"].tolist()

    # 2. Prices
    log.info("\n[1/5] Downloading price data...")
    prices = download_prices(tickers)

    # 3. Fundamentals
    log.info(f"\n[2/5] Fetching fundamentals for {len(tickers)} tickers...")
    fundamentals = {}
    for i, t in enumerate(tickers, 1):
        if i % 50 == 0:
            log.info(f"  {i}/{len(tickers)}")
        fundamentals[t] = get_fundamentals(t)
        time.sleep(0.1)

    # 4. Build base dataframe + E0
    log.info("\n[3/5] Computing 1st order entropy (E0)...")
    records = []
    for t in tickers:
        fund = fundamentals.get(t, {})
        pm   = price_metrics(prices.get(t))
        rec  = {
            "Ticker":           t,
            "Security":         tickers_df.loc[tickers_df["Symbol"] == t, "Security"].values[0]
                                if t in tickers_df["Symbol"].values else t,
            "Sector":           fund.get("sector", "Unknown"),
            "profitMargins":    fund.get("profitMargins"),
            "operatingMargins": fund.get("operatingMargins"),
            "freeCashflow":     fund.get("freeCashflow"),
            "returnOnEquity":   fund.get("returnOnEquity"),
            "debtToEquity":     fund.get("debtToEquity"),
            "currentRatio":     fund.get("currentRatio"),
            "marketCap":        fund.get("marketCap"),
            **pm,
        }
        records.append(rec)

    df = pd.DataFrame(records)
    df["E0"] = df.apply(lambda row: compute_E0(row.to_dict(), df), axis=1)

    # 5. Dependency graph + propagation
    log.info("\n[4/5] Building dependency graph & propagating entropy...")
    dep_graph = build_dependency_graph(tickers, fundamentals, prices)

    E0_dict       = dict(zip(df["Ticker"], df["E0"]))
    E_up_dict     = propagate_entropy(E0_dict, dep_graph)
    E_inh_dict    = compute_E_inherited(E0_dict, E_up_dict, dep_graph)

    df["E_upstream"]  = df["Ticker"].map(E_up_dict)
    df["E_inherited"] = df["Ticker"].map(E_inh_dict)
    df["E_total"]     = df["E0"] + df["E_upstream"] + df["E_inherited"]

    # Rank
    df = df.sort_values("E_total").reset_index(drop=True)
    df["Rank"] = range(1, len(df) + 1)

    # 6. Outputs
    log.info("\n[5/5] Saving outputs...")

    col_order = ["Rank", "Ticker", "Security", "Sector",
                 "E_total", "E0", "E_upstream", "E_inherited",
                 "volatility", "max_drawdown", "shannon", "tail_risk",
                 "profitMargins", "operatingMargins", "freeCashflow",
                 "returnOnEquity", "debtToEquity", "currentRatio", "marketCap"]
    out_cols = [c for c in col_order if c in df.columns]

    csv_path = output_dir / "sp500_entropy_ranked.csv"
    df[out_cols].to_csv(csv_path, index=False)
    log.info(f"CSV: {csv_path.absolute()}")

    html_path = output_dir / "sp500_entropy_report.html"
    generate_html_report(df[out_cols], html_path)

    # Summary
    log.info("\n" + "=" * 60)
    log.info("TOP 10 — LOWEST STRUCTURAL ENTROPY (Most Resilient)")
    log.info("=" * 60)
    for _, r in df.head(10).iterrows():
        log.info(f"  #{int(r['Rank']):3d}  {r['Ticker']:6s}  {r['Sector']:30s}  E_total={r['E_total']:.2f}")

    log.info("\n" + "=" * 60)
    log.info("TOP 10 — HIGHEST STRUCTURAL ENTROPY (Most Fragile)")
    log.info("=" * 60)
    for _, r in df.tail(10).iterrows():
        log.info(f"  #{int(r['Rank']):3d}  {r['Ticker']:6s}  {r['Sector']:30s}  E_total={r['E_total']:.2f}")

    log.info(f"\nDone. Open {html_path.absolute()} in your browser.")


if __name__ == "__main__":
    main()
