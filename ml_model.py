"""
Customer Risk Analysis — ML Engine (Advanced 6-Table Version)
═══════════════════════════════════════════════════════════════
All six SAP tables are mandatory:

  BSID  — Open customer invoices          (core AR data)
  BSAD  — Cleared customer invoices       (historical payment behavior)
  BKPF  — FI document headers             (document-type filtering)
  KNA1  — Customer general master         (name, country, city)
  KNB1  — Customer company-code data      (payment terms, blocking status)
  KNKK  — Customer credit management      (credit limit, open balance, SAP risk class)

Because all tables are present, the ML feature set is fully activated:
  • Credit utilization  (SKFOR ÷ KLIMK)              from KNKK
  • Historical late-payment rate + avg late days      from BSAD
  • SAP risk class (CTLPC) as numeric baseline signal from KNKK
  • Document-type filtered invoice set via BKPF
  • Payment terms (credit days) parsed from ZTERM     from KNB1 / BSID
  • Country / company-code enrichment                 from KNA1 / KNB1

K-Means clusters customers into 4 tiers: Low → Medium → High → Critical
Risk score 0-100 derived from centroid geometry within each tier band.
"""

import os
import re
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def run_customer_risk_analysis(
    bsid_path: str,
    bsad_path: str,
    bkpf_path: str,
    kna1_path: str,
    knb1_path: str,
    knkk_path: str,
) -> dict:
    """
    Full 6-table pipeline:
      load → filter → clean → age → engineer features → cluster → score → output

    All six paths are required.
    """

    # 1. Load all six tables
    bsid = _load_bsid(bsid_path)
    bsad = _load_bsad(bsad_path)
    bkpf = _load_bkpf(bkpf_path)
    kna1 = _load_kna1(kna1_path)
    knb1 = _load_knb1(knb1_path)
    knkk = _load_knkk(knkk_path)

    # 2. Filter BSID to valid AR document types using BKPF
    bsid = _filter_by_bkpf(bsid, bkpf)

    # 3. Resolve aging reference date
    bsid, aging_ref = _resolve_aging_reference(bsid)

    # 4. Compute overdue days + aging buckets on raw invoice lines
    bsid = _compute_aging(bsid, aging_ref)

    # 5. Build payment history features from BSAD
    payment_history = _build_payment_history(bsad)

    # 6. Build credit features from KNKK
    credit_features = _build_credit_features(knkk)

    # 7. Customer-level feature engineering from BSID
    customer_df = _engineer_features(bsid, aging_ref)

    # 8. Merge all master + enrichment tables
    customer_df = _merge_master(customer_df, kna1, knb1, payment_history, credit_features)

    # 9. K-Means clustering + risk scoring
    customer_df, scaler, centroids = _kmeans_cluster_and_score(customer_df)

    # 10. Build structured result payload
    return _build_result(customer_df, bsid, knkk)


# ═══════════════════════════════════════════════════════════════════════════════
# AGING REFERENCE DATE RESOLUTION
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_aging_reference(bsid: pd.DataFrame):
    """
    Priority:
      1. AGING_REFERENCE_DATE env var (YYYY-MM-DD)
      2. Latest NETDT / ZFBDT / BUDAT / BLDAT in BSID → end of that month
      3. datetime.today()
    """
    env_val = os.environ.get("AGING_REFERENCE_DATE", "").strip()
    if env_val:
        try:
            return bsid, datetime.strptime(env_val, "%Y-%m-%d")
        except ValueError:
            pass

    for col in ("NETDT", "ZFBDT", "BUDAT", "BLDAT"):
        if col in bsid.columns and bsid[col].notna().any():
            max_date = bsid[col].dropna().max()
            if pd.notna(max_date):
                ts = pd.Timestamp(max_date)
                if ts.month < 12:
                    eom = pd.Timestamp(year=ts.year, month=ts.month + 1, day=1) - pd.Timedelta(days=1)
                else:
                    eom = pd.Timestamp(year=ts.year + 1, month=1, day=1) - pd.Timedelta(days=1)
                return bsid, eom.to_pydatetime()

    return bsid, datetime.today()


# ═══════════════════════════════════════════════════════════════════════════════
# COLUMN REMAPPING UTILITY
# ═══════════════════════════════════════════════════════════════════════════════

def _remap_columns(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    rename = {}
    for canonical, aliases in col_map.items():
        if canonical in df.columns:
            continue
        for alias in aliases:
            if alias in df.columns:
                rename[alias] = canonical
                break
    return df.rename(columns=rename)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ═══════════════════════════════════════════════════════════════════════════════

def _load_bsid(path: str) -> pd.DataFrame:
    """BSID — Customer Open Items (core AR table)."""
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip().upper() for c in df.columns]
    df = _remap_columns(df, {
        "KUNNR": ["KUNNR", "CUSTOMER", "CUSTOMER_ID", "CUSTOMER_NO", "CUST_NO"],
        "BLDAT": ["BLDAT", "INVOICE_DATE", "DOCUMENT_DATE", "DOC_DATE"],
        "BUDAT": ["BUDAT", "POSTING_DATE", "POST_DATE"],
        "ZFBDT": ["ZFBDT", "BASELINE_DATE", "BASE_DATE", "VALUE_DATE"],
        "NETDT": ["NETDT", "NET_DUE_DATE", "DUE_DATE", "MATURITY_DATE"],
        "WRBTR": ["WRBTR", "AMOUNT", "INVOICE_AMOUNT", "DMBTR", "LC_AMOUNT"],
        "ZTERM": ["ZTERM", "PAYMENT_TERMS", "PAYMENT_TERM", "PAY_TERMS"],
        "BELNR": ["BELNR", "DOCUMENT_NO", "DOC_NUMBER", "FI_DOC_NO"],
        "BUKRS": ["BUKRS", "COMPANY_CODE", "CO_CODE"],
        "WAERS": ["WAERS", "CURRENCY", "CURR"],
    })
    for col in ["BLDAT", "BUDAT", "ZFBDT", "NETDT"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "WRBTR" in df.columns:
        df["WRBTR"] = pd.to_numeric(df["WRBTR"], errors="coerce").fillna(0).abs()
    df = df.dropna(subset=[c for c in ["KUNNR", "WRBTR"] if c in df.columns])
    if "WRBTR" in df.columns:
        df = df[df["WRBTR"] > 0]
    return df.reset_index(drop=True)


def _load_bsad(path: str) -> pd.DataFrame:
    """BSAD — Customer Cleared Items (historical payment behavior)."""
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip().upper() for c in df.columns]
    df = _remap_columns(df, {
        "KUNNR": ["KUNNR", "CUSTOMER", "CUSTOMER_ID", "CUST_NO"],
        "BLDAT": ["BLDAT", "INVOICE_DATE", "DOCUMENT_DATE"],
        "ZFBDT": ["ZFBDT", "BASELINE_DATE", "BASE_DATE"],
        "NETDT": ["NETDT", "NET_DUE_DATE", "DUE_DATE"],
        "AUGDT": ["AUGDT", "CLEARING_DATE", "PAYMENT_DATE", "CLEAR_DATE"],
        "WRBTR": ["WRBTR", "AMOUNT", "INVOICE_AMOUNT", "DMBTR"],
        "BELNR": ["BELNR", "DOCUMENT_NO", "DOC_NUMBER"],
    })
    for col in ["BLDAT", "ZFBDT", "NETDT", "AUGDT"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "WRBTR" in df.columns:
        df["WRBTR"] = pd.to_numeric(df["WRBTR"], errors="coerce").fillna(0).abs()
    return df.reset_index(drop=True)


def _load_bkpf(path: str) -> pd.DataFrame:
    """BKPF — FI Document Headers (used to filter BSID to AR-only records)."""
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip().upper() for c in df.columns]
    return _remap_columns(df, {
        "BELNR": ["BELNR", "DOCUMENT_NO", "DOC_NUMBER", "FI_DOC_NO"],
        "BUDAT": ["BUDAT", "POSTING_DATE"],
        "BLART": ["BLART", "DOCUMENT_TYPE", "DOC_TYPE"],
        "BUKRS": ["BUKRS", "COMPANY_CODE"],
        "GJAHR": ["GJAHR", "FISCAL_YEAR"],
    })


def _load_kna1(path: str) -> pd.DataFrame:
    """KNA1 — Customer General Master (name, country, city)."""
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip().upper() for c in df.columns]
    return _remap_columns(df, {
        "KUNNR": ["KUNNR", "CUSTOMER", "CUSTOMER_ID", "CUST_NO"],
        "NAME1": ["NAME1", "CUSTOMER_NAME", "NAME", "CUST_NAME"],
        "LAND1": ["LAND1", "COUNTRY", "COUNTRY_CODE"],
        "ORT01": ["ORT01", "CITY", "TOWN"],
        "KTOKD": ["KTOKD", "ACCOUNT_GROUP"],
    })


def _load_knb1(path: str) -> pd.DataFrame:
    """KNB1 — Customer Company-Code Data (payment terms, blocking status)."""
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip().upper() for c in df.columns]
    df = _remap_columns(df, {
        "KUNNR": ["KUNNR", "CUSTOMER", "CUSTOMER_ID"],
        "BUKRS": ["BUKRS", "COMPANY_CODE", "CO_CODE"],
        "ZTERM": ["ZTERM", "PAYMENT_TERMS", "PAYMENT_TERM"],
        "SPERR": ["SPERR", "PAYMENT_BLOCK", "BLOCK_INDICATOR"],
    })
    if "SPERR" in df.columns:
        df["PAYMENT_BLOCKED"] = df["SPERR"].apply(
            lambda x: 1 if pd.notna(x) and str(x).strip() not in ("", "0", "nan") else 0
        )
    else:
        df["PAYMENT_BLOCKED"] = 0
    return df


def _load_knkk(path: str) -> pd.DataFrame:
    """KNKK — Customer Credit Management (credit limit, open balance, SAP risk class)."""
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip().upper() for c in df.columns]
    df = _remap_columns(df, {
        "KUNNR": ["KUNNR", "CUSTOMER", "CUSTOMER_ID"],
        "KLIMK": ["KLIMK", "CREDIT_LIMIT", "CRED_LIMIT"],
        "SKFOR": ["SKFOR", "OPEN_BALANCE", "RECEIVABLE_BALANCE", "OPEN_ITEMS"],
        "CTLPC": ["CTLPC", "RISK_CLASS", "SAP_RISK_CLASS", "RISK_CATEGORY"],
    })
    for col in ["KLIMK", "SKFOR"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).abs()
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# BKPF FILTER
# ═══════════════════════════════════════════════════════════════════════════════

_AR_DOC_TYPES = {"RV", "DR", "DG", "DA", "AB", "DZ", "DF"}

def _filter_by_bkpf(bsid: pd.DataFrame, bkpf: pd.DataFrame) -> pd.DataFrame:
    """Keep only BSID rows with BELNR matching valid AR document types in BKPF."""
    if "BELNR" not in bsid.columns or "BELNR" not in bkpf.columns or "BLART" not in bkpf.columns:
        return bsid
    valid_docs = bkpf[bkpf["BLART"].isin(_AR_DOC_TYPES)]["BELNR"].unique()
    filtered   = bsid[bsid["BELNR"].isin(valid_docs)]
    return filtered if len(filtered) > 0 else bsid   # fallback: keep all


# ═══════════════════════════════════════════════════════════════════════════════
# AGING CALCULATION
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_aging(df: pd.DataFrame, aging_ref: datetime) -> pd.DataFrame:
    """
    Overdue Days = aging_ref − due_date (clipped ≥ 0).
    Due-date priority: NETDT > ZFBDT > BUDAT > BLDAT.

    6 aging buckets:
      Current   ≤ 0   Not yet due        No Risk
      1-30      1-30  Mild delay         Low
      31-60    31-60  Moderate delay     Medium
      61-90    61-90  High delay         High
      91-120  91-120  Severe delay       Severe
      120+    >120    Critical           Critical
    """
    due_col = None
    for c in ("NETDT", "ZFBDT", "BUDAT", "BLDAT"):
        if c in df.columns and df[c].notna().any():
            due_col = c
            break

    ref_ts = pd.Timestamp(aging_ref)
    df["DAYS_OVERDUE"] = (ref_ts - df[due_col]).dt.days.clip(lower=0) if due_col else 0

    bins   = [-1, 0, 30, 60, 90, 120, float("inf")]
    labels = ["Current", "1-30", "31-60", "61-90", "91-120", "120+"]
    df["AGING_BUCKET"] = pd.cut(df["DAYS_OVERDUE"], bins=bins, labels=labels)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# PAYMENT HISTORY — BSAD
# ═══════════════════════════════════════════════════════════════════════════════

def _build_payment_history(bsad: pd.DataFrame) -> pd.DataFrame:
    """
    Per-customer late-payment behavior from cleared invoices.
    Late Days = AUGDT − NETDT (or ZFBDT / BLDAT as fallback).
    """
    if "AUGDT" not in bsad.columns or "KUNNR" not in bsad.columns:
        return pd.DataFrame()

    bsad = bsad.copy()
    due_col = next((c for c in ("NETDT", "ZFBDT", "BLDAT") if c in bsad.columns and bsad[c].notna().any()), None)
    bsad["LATE_DAYS"] = (bsad["AUGDT"] - bsad[due_col]).dt.days.fillna(0).clip(lower=0) if due_col else 0

    amount_col = "WRBTR" if "WRBTR" in bsad.columns else None

    agg_dict = {
        "HIST_TOTAL_INVOICES": ("LATE_DAYS", "count"),
        "HIST_LATE_INVOICES":  ("LATE_DAYS", lambda x: (x > 0).sum()),
        "HIST_AVG_LATE_DAYS":  ("LATE_DAYS", "mean"),
        "HIST_MAX_LATE_DAYS":  ("LATE_DAYS", "max"),
    }
    if amount_col:
        agg_dict["HIST_TOTAL_PAID"]      = (amount_col, "sum")
        agg_dict["HIST_AVG_PAYMENT_AMT"] = (amount_col, "mean")

    hist = bsad.groupby("KUNNR").agg(**agg_dict).reset_index()
    hist["HIST_LATE_RATE"] = (
        hist["HIST_LATE_INVOICES"] / hist["HIST_TOTAL_INVOICES"].clip(lower=1)
    ).round(4)
    for col in ["HIST_AVG_LATE_DAYS", "HIST_MAX_LATE_DAYS", "HIST_TOTAL_PAID", "HIST_AVG_PAYMENT_AMT"]:
        if col in hist.columns:
            hist[col] = hist[col].round(2)
    return hist


# ═══════════════════════════════════════════════════════════════════════════════
# CREDIT FEATURES — KNKK
# ═══════════════════════════════════════════════════════════════════════════════

def _build_credit_features(knkk: pd.DataFrame) -> pd.DataFrame:
    """
    Credit Utilization = SKFOR / KLIMK × 100.
    SAP Risk Class: 001→1, 002→2, 003→3.
    Flags customers over / near their credit limit.
    """
    if "KUNNR" not in knkk.columns:
        return pd.DataFrame()

    knkk = knkk.copy()

    if "KLIMK" in knkk.columns and "SKFOR" in knkk.columns:
        knkk["CREDIT_UTILIZATION"]    = np.where(
            knkk["KLIMK"] > 0,
            (knkk["SKFOR"] / knkk["KLIMK"] * 100).round(2), 0.0
        )
        knkk["CREDIT_LIMIT_BREACHED"] = (knkk["CREDIT_UTILIZATION"] > 100).astype(int)
    else:
        knkk["CREDIT_UTILIZATION"]    = 0.0
        knkk["CREDIT_LIMIT_BREACHED"] = 0

    if "CTLPC" in knkk.columns:
        def _parse_ctlpc(val):
            try:
                return int(str(val).strip().lstrip("0") or "0")
            except (ValueError, TypeError):
                return 0
        knkk["SAP_RISK_NUM"] = knkk["CTLPC"].apply(_parse_ctlpc).clip(0, 3)
    else:
        knkk["SAP_RISK_NUM"] = 0

    cols = ["KUNNR", "CREDIT_UTILIZATION", "CREDIT_LIMIT_BREACHED", "SAP_RISK_NUM"]
    for c in ["KLIMK", "SKFOR", "CTLPC"]:
        if c in knkk.columns:
            cols.append(c)
    return knkk[cols].drop_duplicates("KUNNR")


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_credit_days(zterm) -> int:
    """Extract numeric credit days from ZTERM code (e.g. NT30 → 30)."""
    if pd.isna(zterm):
        return 30
    m = re.search(r"(\d+)", str(zterm).strip().upper())
    if m:
        v = int(m.group(1))
        return v if 0 < v <= 365 else 30
    return 30


def _engineer_features(bsid: pd.DataFrame, aging_ref: datetime) -> pd.DataFrame:
    """Build per-customer feature matrix from open BSID invoice lines."""
    return (
        bsid.groupby("KUNNR")
        .apply(_customer_feature_row, aging_ref=aging_ref)
        .reset_index()
    )


def _customer_feature_row(grp: pd.DataFrame, aging_ref) -> pd.Series:
    amounts   = grp["WRBTR"]
    days      = grp["DAYS_OVERDUE"]
    buckets   = grp["AGING_BUCKET"].astype(str)
    total_amt = float(amounts.sum())
    n         = len(grp)

    overdue_mask   = days > 0
    overdue_amount = float(amounts[overdue_mask].sum())
    pct_critical   = float((days >= 90).sum() / max(n, 1))

    b1_30   = float(amounts[buckets == "1-30"].sum())
    b31_60  = float(amounts[buckets == "31-60"].sum())
    b61_90  = float(amounts[buckets == "61-90"].sum())
    b91_120 = float(amounts[buckets == "91-120"].sum())
    b120p   = float(amounts[buckets == "120+"].sum())

    overdue_ratio  = (overdue_amount / total_amt) if total_amt > 0 else 0.0
    shares         = (amounts / total_amt) if total_amt > 0 else amounts * 0
    concentration  = float((shares ** 2).sum())

    due_col = next((c for c in ("NETDT", "ZFBDT", "BUDAT", "BLDAT")
                    if c in grp.columns and grp[c].notna().any()), None)
    recency = float((pd.Timestamp(aging_ref) - grp[due_col].dropna().min()).days) \
              if due_col and grp[due_col].notna().any() else 0.0

    credit_days = 30
    if "ZTERM" in grp.columns and grp["ZTERM"].notna().any():
        credit_days = _parse_credit_days(grp["ZTERM"].dropna().iloc[0])

    return pd.Series({
        "TOTAL_RECEIVABLE_AMOUNT": round(total_amt, 2),
        "OVERDUE_AMOUNT":          round(overdue_amount, 2),
        "TOTAL_INVOICES":          int(n),
        "MAX_DAYS_OVERDUE":        float(days.max()),
        "AVG_DAYS_OVERDUE":        round(float(days.mean()), 2),
        "PCT_CRITICAL_INVOICES":   round(pct_critical, 4),
        "BUCKET_1_30_AMT":         round(b1_30, 2),
        "BUCKET_31_60_AMT":        round(b31_60, 2),
        "BUCKET_61_90_AMT":        round(b61_90, 2),
        "BUCKET_91_120_AMT":       round(b91_120, 2),
        "BUCKET_120PLUS_AMT":      round(b120p, 2),
        "OVERDUE_RATIO":           round(float(overdue_ratio), 4),
        "AMOUNT_CONCENTRATION":    round(concentration, 4),
        "RECENCY_DAYS":            round(recency, 1),
        "CREDIT_DAYS_USED":        int(credit_days),
    })


# ═══════════════════════════════════════════════════════════════════════════════
# MERGE ALL TABLES
# ═══════════════════════════════════════════════════════════════════════════════

def _merge_master(agg, kna1, knb1, payment_history, credit_features):
    # KNA1
    kna1_cols = ["KUNNR"] + [c for c in ["NAME1", "LAND1", "ORT01", "KTOKD"] if c in kna1.columns]
    agg = agg.merge(kna1[kna1_cols].drop_duplicates("KUNNR"), on="KUNNR", how="left")

    # KNB1
    knb1_cols = ["KUNNR"] + [c for c in ["BUKRS", "ZTERM", "PAYMENT_BLOCKED"] if c in knb1.columns]
    agg = agg.merge(knb1[knb1_cols].drop_duplicates("KUNNR"), on="KUNNR", how="left")

    # BSAD payment history
    if not payment_history.empty and "KUNNR" in payment_history.columns:
        agg = agg.merge(payment_history, on="KUNNR", how="left")
        for col in ["HIST_LATE_RATE", "HIST_AVG_LATE_DAYS", "HIST_MAX_LATE_DAYS",
                    "HIST_TOTAL_INVOICES", "HIST_LATE_INVOICES"]:
            if col in agg.columns:
                agg[col] = agg[col].fillna(0)

    # KNKK credit features
    if not credit_features.empty and "KUNNR" in credit_features.columns:
        agg = agg.merge(credit_features, on="KUNNR", how="left")
        for col in ["CREDIT_UTILIZATION", "SAP_RISK_NUM", "CREDIT_LIMIT_BREACHED"]:
            if col in agg.columns:
                agg[col] = agg[col].fillna(0)

    agg["NAME1"]           = agg["NAME1"].fillna("Unknown Customer") if "NAME1" in agg.columns else "Unknown Customer"
    agg["PAYMENT_BLOCKED"] = agg.get("PAYMENT_BLOCKED", pd.Series(0, index=agg.index)).fillna(0).astype(int)
    return agg


# ═══════════════════════════════════════════════════════════════════════════════
# K-MEANS CLUSTERING + RISK SCORING (full 6-table feature set)
# ═══════════════════════════════════════════════════════════════════════════════

_CLUSTER_FEATURES = [
    # BSID-derived
    "TOTAL_RECEIVABLE_AMOUNT",   # absolute AR exposure
    "OVERDUE_AMOUNT",            # overdue-only exposure          ← primary signal
    "TOTAL_INVOICES",            # transaction volume
    "MAX_DAYS_OVERDUE",          # worst single breach
    "AVG_DAYS_OVERDUE",          # chronic lateness pattern
    "PCT_CRITICAL_INVOICES",     # 90+ day concentration
    "BUCKET_120PLUS_AMT",        # critical bucket absolute amount
    "OVERDUE_RATIO",             # overdue share of total AR
    "AMOUNT_CONCENTRATION",      # Herfindahl risk concentration
    "RECENCY_DAYS",              # staleness of oldest open item
    # BSAD-derived
    "HIST_LATE_RATE",            # historical chronic late-payment rate
    "HIST_AVG_LATE_DAYS",        # avg days late on cleared invoices
    "HIST_MAX_LATE_DAYS",        # worst historical late payment
    # KNKK-derived
    "CREDIT_UTILIZATION",        # % of credit limit consumed
    "CREDIT_LIMIT_BREACHED",     # binary: limit exceeded
    "SAP_RISK_NUM",              # SAP's own risk class (0-3)
]

_DIRECTION_WEIGHTS = np.array([
    0.11,   # TOTAL_RECEIVABLE_AMOUNT
    0.14,   # OVERDUE_AMOUNT
    0.04,   # TOTAL_INVOICES
    0.10,   # MAX_DAYS_OVERDUE
    0.10,   # AVG_DAYS_OVERDUE
    0.07,   # PCT_CRITICAL_INVOICES
    0.06,   # BUCKET_120PLUS_AMT
    0.07,   # OVERDUE_RATIO
    0.02,   # AMOUNT_CONCENTRATION
    0.01,   # RECENCY_DAYS
    0.08,   # HIST_LATE_RATE
    0.05,   # HIST_AVG_LATE_DAYS
    0.02,   # HIST_MAX_LATE_DAYS
    0.07,   # CREDIT_UTILIZATION
    0.03,   # CREDIT_LIMIT_BREACHED
    0.03,   # SAP_RISK_NUM
])

_RISK_LABELS = ["Low", "Medium", "High", "Critical"]


def _kmeans_cluster_and_score(df: pd.DataFrame):
    """
    K-Means(k=4) clustering with geometric 0-100 risk scoring.

    Each cluster gets a 25-point band (Low=0-25, …, Critical=75-100).
    Within-band position = distance to own centroid / max distance in row.
    """
    active_feats   = [f for f in _CLUSTER_FEATURES if f in df.columns]
    active_weights = np.array([_DIRECTION_WEIGHTS[_CLUSTER_FEATURES.index(f)] for f in active_feats])
    active_weights = active_weights / active_weights.sum()

    X_raw    = df[active_feats].fillna(0)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    n_clusters = min(4, len(df))
    kmeans     = KMeans(n_clusters=n_clusters, random_state=42, n_init=20, max_iter=500)
    labels     = kmeans.fit_predict(X_scaled)

    df = df.copy()
    df["CLUSTER"] = labels

    centroids       = kmeans.cluster_centers_
    centroid_danger = centroids @ active_weights
    cluster_rank    = np.argsort(np.argsort(centroid_danger))

    risk_label_map       = {c: _RISK_LABELS[cluster_rank[c]] for c in range(n_clusters)}
    df["PREDICTED_RISK"] = df["CLUSTER"].map(risk_label_map)

    distances  = cdist(X_scaled, centroids, metric="euclidean")
    cust_idx   = np.arange(len(df))
    band_size  = 100.0 / n_clusters
    band_floor = cluster_rank[df["CLUSTER"].values] * band_size
    own_dist   = distances[cust_idx, df["CLUSTER"].values]
    max_dist   = distances.max(axis=1).clip(min=1e-9)
    within_pos = (own_dist / max_dist) * band_size

    df["RISK_SCORE"] = np.clip(band_floor + within_pos, 0, 100).round(2)
    return df, scaler, centroids


# ═══════════════════════════════════════════════════════════════════════════════
# RESULT PAYLOAD BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

_RENAME = {
    "KUNNR": "customer_id", "NAME1": "customer_name",
    "LAND1": "country", "ORT01": "city",
    "BUKRS": "company_code", "ZTERM": "payment_terms",
    "PAYMENT_BLOCKED": "payment_blocked",
    "TOTAL_RECEIVABLE_AMOUNT": "total_receivable",
    "OVERDUE_AMOUNT": "overdue_amount",
    "TOTAL_INVOICES": "total_invoices",
    "MAX_DAYS_OVERDUE": "max_days_overdue",
    "AVG_DAYS_OVERDUE": "avg_days_overdue",
    "PCT_CRITICAL_INVOICES": "pct_critical_invoices",
    "BUCKET_1_30_AMT": "bucket_1_30",
    "BUCKET_31_60_AMT": "bucket_31_60",
    "BUCKET_61_90_AMT": "bucket_61_90",
    "BUCKET_91_120_AMT": "bucket_91_120",
    "BUCKET_120PLUS_AMT": "bucket_120plus",
    "OVERDUE_RATIO": "overdue_ratio",
    "AMOUNT_CONCENTRATION": "amount_concentration",
    "RECENCY_DAYS": "recency_days",
    "CREDIT_DAYS_USED": "credit_days",
    "HIST_LATE_RATE": "hist_late_rate",
    "HIST_AVG_LATE_DAYS": "hist_avg_late_days",
    "HIST_MAX_LATE_DAYS": "hist_max_late_days",
    "HIST_TOTAL_INVOICES": "hist_total_invoices",
    "HIST_LATE_INVOICES": "hist_late_invoices",
    "HIST_TOTAL_PAID": "hist_total_paid",
    "CREDIT_UTILIZATION": "credit_utilization",
    "CREDIT_LIMIT_BREACHED": "credit_limit_breached",
    "SAP_RISK_NUM": "sap_risk_num",
    "KLIMK": "credit_limit", "SKFOR": "open_balance",
    "CTLPC": "sap_risk_class",
    "RISK_SCORE": "risk_score", "PREDICTED_RISK": "predicted_risk",
}


def _build_result(customer_df: pd.DataFrame, bsid: pd.DataFrame, knkk: pd.DataFrame) -> dict:

    # ── KPIs ──────────────────────────────────────────────────────────────────
    total_customers  = len(customer_df)
    total_overdue    = float(customer_df["OVERDUE_AMOUNT"].sum())
    total_receivable = float(customer_df["TOTAL_RECEIVABLE_AMOUNT"].sum())
    high_risk        = int((customer_df["PREDICTED_RISK"] == "High").sum())
    critical         = int((customer_df["PREDICTED_RISK"] == "Critical").sum())
    blocked_count    = int(customer_df.get("PAYMENT_BLOCKED", pd.Series(0)).sum()) \
                       if "PAYMENT_BLOCKED" in customer_df.columns else 0
    credit_breached  = int(customer_df.get("CREDIT_LIMIT_BREACHED", pd.Series(0)).sum()) \
                       if "CREDIT_LIMIT_BREACHED" in customer_df.columns else 0

    # ── Aging bucket totals ────────────────────────────────────────────────────
    aging_raw = bsid.groupby("AGING_BUCKET", observed=True)["WRBTR"].sum().to_dict()
    aging = {b: round(float(aging_raw.get(b, 0.0)), 2)
             for b in ["Current", "1-30", "31-60", "61-90", "91-120", "120+"]}

    # ── Risk distribution ──────────────────────────────────────────────────────
    risk_dist = customer_df["PREDICTED_RISK"].value_counts().to_dict()
    for r in _RISK_LABELS:
        risk_dist.setdefault(r, 0)

    # ── Top 10 by risk score ───────────────────────────────────────────────────
    top10_base = ["KUNNR", "NAME1", "OVERDUE_AMOUNT", "TOTAL_RECEIVABLE_AMOUNT",
                  "RISK_SCORE", "PREDICTED_RISK"]
    top10_extra = [c for c in ["CREDIT_UTILIZATION", "LAND1", "HIST_LATE_RATE",
                               "PAYMENT_BLOCKED", "CREDIT_LIMIT_BREACHED"]
                   if c in customer_df.columns]
    top10 = (
        customer_df.nlargest(10, "RISK_SCORE")[top10_base + top10_extra]
        .rename(columns=_RENAME).to_dict(orient="records")
    )

    # ── Scatter data ───────────────────────────────────────────────────────────
    scatter_base = ["KUNNR", "NAME1", "OVERDUE_AMOUNT", "RISK_SCORE", "PREDICTED_RISK"]
    scatter_extra = [c for c in ["CREDIT_UTILIZATION", "TOTAL_RECEIVABLE_AMOUNT"]
                     if c in customer_df.columns]
    scatter = (
        customer_df[scatter_base + scatter_extra]
        .rename(columns=_RENAME).to_dict(orient="records")
    )

    # ── Full customer table ────────────────────────────────────────────────────
    table_base = [
        "KUNNR", "NAME1",
        "TOTAL_RECEIVABLE_AMOUNT", "OVERDUE_AMOUNT", "TOTAL_INVOICES",
        "MAX_DAYS_OVERDUE", "AVG_DAYS_OVERDUE",
        "PCT_CRITICAL_INVOICES", "BUCKET_120PLUS_AMT", "OVERDUE_RATIO",
        "RISK_SCORE", "PREDICTED_RISK",
    ]
    table_extra = [c for c in [
        "CREDIT_UTILIZATION", "CREDIT_LIMIT_BREACHED", "SAP_RISK_NUM",
        "HIST_LATE_RATE", "HIST_AVG_LATE_DAYS",
        "LAND1", "ZTERM", "BUKRS", "PAYMENT_BLOCKED", "KLIMK", "SKFOR",
    ] if c in customer_df.columns]

    customers = (
        customer_df[table_base + table_extra]
        .rename(columns=_RENAME)
        .sort_values("risk_score", ascending=False)
        .to_dict(orient="records")
    )

    # ── AI context ─────────────────────────────────────────────────────────────
    country_dist = (
        customer_df.groupby("LAND1")["OVERDUE_AMOUNT"].sum()
        .sort_values(ascending=False).head(10).round(2).to_dict()
    ) if "LAND1" in customer_df.columns else {}

    zterm_dist = (
        customer_df["ZTERM"].fillna("Unknown").value_counts().head(10).to_dict()
    ) if "ZTERM" in customer_df.columns else {}

    bukrs_dist = (
        customer_df.groupby("BUKRS")["OVERDUE_AMOUNT"].sum()
        .sort_values(ascending=False).round(2).to_dict()
    ) if "BUKRS" in customer_df.columns else {}

    credit_stats = {}
    if "CREDIT_UTILIZATION" in customer_df.columns:
        util = customer_df["CREDIT_UTILIZATION"]
        credit_stats = {
            "customers_over_credit_limit":   int((util > 100).sum()),
            "customers_near_credit_limit":   int(((util > 80) & (util <= 100)).sum()),
            "customers_healthy_utilization": int((util <= 50).sum()),
            "avg_credit_utilization":        round(float(util.mean()), 2),
            "max_credit_utilization":        round(float(util.max()), 2),
        }

    payment_stats = {}
    if "HIST_LATE_RATE" in customer_df.columns:
        lr = customer_df["HIST_LATE_RATE"]
        payment_stats = {
            "chronic_late_payers":      int((lr > 0.5).sum()),
            "avg_historical_late_rate": round(float(lr.mean()), 4),
            "customers_with_history":   int(lr.notna().sum()),
        }

    # Invoice sample: 50 most overdue
    inv_cols = ["KUNNR", "WRBTR", "DAYS_OVERDUE", "AGING_BUCKET"]
    for c in ["BLDAT", "NETDT", "ZFBDT", "BELNR"]:
        if c in bsid.columns:
            inv_cols.append(c)
    invoice_sample = (
        bsid[inv_cols].nlargest(50, "DAYS_OVERDUE")
        .rename(columns={
            "KUNNR": "customer_id", "WRBTR": "amount",
            "DAYS_OVERDUE": "days_overdue", "AGING_BUCKET": "aging_bucket",
            "BLDAT": "invoice_date", "NETDT": "due_date",
            "ZFBDT": "baseline_date", "BELNR": "document_no",
        }).to_dict(orient="records")
    )
    for row in invoice_sample:
        for k in ("invoice_date", "due_date", "baseline_date"):
            if k in row and hasattr(row[k], "strftime"):
                row[k] = row[k].strftime("%Y-%m-%d")

    # Full AI customer list
    ai_base = [
        "KUNNR", "NAME1",
        "TOTAL_RECEIVABLE_AMOUNT", "OVERDUE_AMOUNT", "TOTAL_INVOICES",
        "MAX_DAYS_OVERDUE", "AVG_DAYS_OVERDUE", "PCT_CRITICAL_INVOICES",
        "BUCKET_120PLUS_AMT", "OVERDUE_RATIO", "RISK_SCORE", "PREDICTED_RISK",
    ]
    ai_extra = [c for c in [
        "LAND1", "ZTERM", "BUKRS",
        "CREDIT_UTILIZATION", "CREDIT_LIMIT_BREACHED", "SAP_RISK_NUM",
        "HIST_LATE_RATE", "HIST_AVG_LATE_DAYS", "PAYMENT_BLOCKED",
    ] if c in customer_df.columns]

    all_customers_ai = (
        customer_df[ai_base + ai_extra]
        .rename(columns=_RENAME)
        .sort_values("risk_score", ascending=False)
        .to_dict(orient="records")
    )

    risk_overdue = (
        customer_df.groupby("PREDICTED_RISK")["OVERDUE_AMOUNT"]
        .sum().round(2).to_dict()
    )

    ai_context = {
        "total_invoices_processed":    int(len(bsid)),
        "aging_amounts":               aging,
        "risk_overdue_totals":         risk_overdue,
        "country_distribution":        country_dist,
        "payment_terms_distribution":  zterm_dist,
        "company_code_distribution":   bukrs_dist,
        "credit_stats":                credit_stats,
        "payment_history_stats":       payment_stats,
        "top50_most_overdue_invoices": invoice_sample,
        "all_customers":               all_customers_ai,
    }

    return {
        "kpi": {
            "total_customers":       total_customers,
            "total_overdue":         round(total_overdue, 2),
            "total_receivable":      round(total_receivable, 2),
            "high_risk":             high_risk,
            "critical":              critical,
            "payment_blocked":       blocked_count,
            "credit_limit_breached": credit_breached,
        },
        "aging_buckets":     aging,
        "risk_distribution": risk_dist,
        "top10":             top10,
        "scatter":           scatter,
        "customers":         customers,
        "ai_context":        ai_context,
    }