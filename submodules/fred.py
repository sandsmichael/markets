import sys, os
import pandas as pd
import numpy as np
import requests
import json
# from fredapi import Fred
from pathlib import Path
from typing import List, Tuple
import datetime

parent = Path(__file__).resolve().parent


# Find secrets.json in multiple locations
def find_secrets():
    """Find secrets.json in current dir, parent dir, or common locations."""
    possible_paths = [
        "secrets.json",  # Current directory
        "./secrets.json",  # Current directory (explicit)
        "../secrets.json",  # Parent directory
        "../../secrets.json",  # Two levels up
        # r"C:\Users\msands\OneDrive\Documents\code\submodules\secrets.json",  # Original location
        # r"C:\Users\msands\OneDrive\Documents\code\markets\secrets.json",  # Markets directory
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "secrets.json not found. Searched locations:\n" + "\n".join(f"  - {p}" for p in possible_paths)
    )

SECRET_FP = find_secrets()
with open(SECRET_FP, 'r') as file:
    secrets = json.load(file)
API_KEY = secrets['fred_api_key']



ID_MACRO = ["GDP", "CPIAUCSL", "UNRATE"]
ID_FI = ["BAMLH0A0HYM2"]
ID_FUNDAMENTAL = []
ID_EQTY = ['SP500']
ID_CRYPTO = ['CBBTCUSD']
# IDS_STANDARD = [*IDS_STD_MACRO, *IDS_STD_FI, *IDS_STD_FUNDAMENTAL, *IDS_STD_EQTY]

API = "https://api.stlouisfed.org"


def get_matrix(ids = ["GDP", "CPIAUCSL", "UNRATE"]):
    """ Retrieve multiple Fred data series' and return a dataframe. """
    merged_data = None
    for series_id in ids:
        params = {
            "series_id": series_id,
            "api_key": API_KEY,
            "file_type": "json",  
        }
        FRED_URL = f"{API}/fred/series/observations"
        response = requests.get(FRED_URL, params=params)
        
        if response.status_code == 200:
            data = response.json()
            observations = data["observations"]
            df = pd.DataFrame(observations)
            df["value"] = pd.to_numeric(df["value"], errors="coerce") 
            df = df[["date", "value"]]  
            df.rename(columns={"value": series_id}, inplace=True) 
            df.set_index("date", inplace=True) 
            
            if merged_data is None:
                merged_data = df 
            
            else:
                merged_data = merged_data.merge(
                    df, how="outer", left_index=True, right_index=True
                ) 
        
        else:
            print(f"Failed to fetch data for {series_id}. Status code: {response.status_code}")
            pass
    
    if merged_data is None:
        raise ValueError("No data fetched. Please check the series IDs and API key.")

    merged_data.index = pd.to_datetime(merged_data.index)

    return merged_data


def get_vintages(series_id: str) -> list[str]:
    """A vintage is a historical version of a data series as it existed at a specific point in time. I.e. Prior to revisions.
    Vintage Dates: When the data was published/available 
    Observation Dates: The actual time period the data refers to
    
    Returns:
        list: A list of vintage dates for the specified series ID.
    """
    r = requests.get(f"{API}/fred/series/vintagedates",
                     params={"series_id": series_id, "api_key": API_KEY, "file_type": "json"},
                     timeout=30)
    r.raise_for_status()
    return r.json().get("vintage_dates", [])


def get_series(series_id: str) -> dict:
    """
    Fetch FRED series metadata (first/primary record) for a given series_id.

    Returns a dict similar to the element of the 'seriess' array in FRED's /fred/series endpoint.
    """
    r = requests.get(
        f"{API}/fred/series",
        params={"series_id": series_id, "api_key": API_KEY, "file_type": "json"},
        timeout=30,
    )
    r.raise_for_status()
    js = r.json() or {}
    seriess = js.get("seriess", [])
    return seriess[0] if len(seriess) > 0 else {}


def get_observations(
    series_id: str,
    vintage_date: str | None = None,
    observation_start: str | None = None,
    observation_end: str | None = None,
) -> pd.DataFrame:
    """
    Fetch observations for a series, optionally as of a specific vintage (snapshot) and/or
    with an observation date range. Returns a DataFrame with columns ['date','value'].
    If vintage_date is provided, returns values as of that vintage.
    If not, returns latest available values.
    """
    params = {
        "series_id": series_id,
        "api_key": API_KEY,
        "file_type": "json",
    }
    if observation_start:
        params["observation_start"] = observation_start
    if observation_end:
        params["observation_end"] = observation_end
    if vintage_date:
        params["vintage_dates"] = vintage_date

    r = requests.get(f"{API}/fred/series/observations", params=params, timeout=30)
    r.raise_for_status()
    js = r.json() or {}
    obs = pd.DataFrame(js.get("observations", []))
    if obs.empty:
        return pd.DataFrame(columns=["date", "value"])

    obs["date"] = pd.to_datetime(obs["date"])
    obs["value"] = pd.to_numeric(obs["value"], errors="coerce")
    return obs[["date", "value"]]



def get_latest_revisions(
    series_id: str,
    n_revisions: int = 2,
    vintage_spacing: int = 1,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare multiple vintages of a FRED series to analyze revisions.
    
    Args:
        series_id: FRED series identifier
        n_revisions: Number of vintages to compare (default 2)
        vintage_spacing: How many vintages to skip between comparisons (default 1)
    
    Returns:
        tuple: (observations_df, revisions_df)
            - observations_df: DataFrame with date index and vintage date columns
            - revisions_df: DataFrame with date index and revision amount columns
    """
    # Get available vintages
    v = get_vintages(series_id)
    if len(v) < n_revisions:
        raise ValueError(f"Only {len(v)} vintages available, requested {n_revisions}")
    
    # Select vintages (most recent n_revisions, spaced by vintage_spacing)
    selected_vintages = []
    for i in range(n_revisions):
        vintage_idx = -(1 + i * vintage_spacing)
        if abs(vintage_idx) <= len(v):
            selected_vintages.append(v[vintage_idx])
    
    # Reverse to get chronological order (oldest to newest)
    selected_vintages.reverse()
    
    # Fetch observations for each vintage (already have date as index)
    dfs = []
    for vintage in selected_vintages:
        df = get_observations(series_id, vintage)
        # Ensure index is date and only one column named by vintage
        if "date" in df.columns:
            df = df.set_index("date")
        # Rename value column to the vintage date
        df = df.rename(columns={"value": vintage})
        # Keep only the vintage column
        df = df[[vintage]]
        dfs.append(df)

    # Merge all vintages for observations DataFrame
    # Start with the first, then join each subsequent on index
    observations_df = dfs[0]
    for df in dfs[1:]:
        # Only join on index, drop any duplicate date columns
        observations_df = observations_df.join(df, how="outer")

    # After join, drop any columns named 'date' (should only have index)
    if 'date' in observations_df.columns:
        observations_df = observations_df.drop(columns=['date'])

    # Now columns are value_<vintage_date> for each vintage
    
    # Create revisions DataFrame with same index
    revisions_df = pd.DataFrame(index=observations_df.index)
    
    if len(selected_vintages) > 1:
        # Get vintage columns (all columns since date is already index)
        vintage_cols = list(observations_df.columns)
        
        # Add revision columns (difference from previous vintage)
        for i in range(1, len(vintage_cols)):
            prev_col = vintage_cols[i-1]
            curr_col = vintage_cols[i]
            revision_col = f"{prev_col}_to_{curr_col}"
            revisions_df[revision_col] = observations_df[curr_col] - observations_df[prev_col]
        
        # Add cumulative revision (first to last)
        if len(vintage_cols) > 1:
            first_col = vintage_cols[0]
            last_col = vintage_cols[-1]
            revisions_df["total"] = observations_df[last_col] - observations_df[first_col]
    
    # Sort both DataFrames by date index
    observations_df = observations_df.sort_index()
    revisions_df = revisions_df.sort_index()
    
    # Filter for only rows with revisions
    if not observations_df.empty:
        # For observations: keep rows with differences, excluding all-null rows
        value_cols = list(observations_df.columns)
        if len(value_cols) > 1:
            has_differences = observations_df[value_cols].nunique(axis=1, dropna=True) > 1
            not_all_null = ~observations_df[value_cols].isna().all(axis=1)
            observations_df = observations_df[has_differences & not_all_null]
    
    if not revisions_df.empty:
        # For revisions: exclude rows that are all zeros or all nulls
        revision_cols = list(revisions_df.columns)
        if revision_cols:
            not_all_zero = ~(revisions_df[revision_cols] == 0).all(axis=1)
            not_all_null = ~revisions_df[revision_cols].isna().all(axis=1)
            revisions_df = revisions_df[not_all_zero & not_all_null]
    
    # Add metadata to both DataFrames
    metadata = {
        "series_id": series_id,
        "vintages": selected_vintages,
        "n_revisions": n_revisions,
        "vintage_spacing": vintage_spacing
    }
    
    observations_df.attrs = metadata
    revisions_df.attrs = metadata
    
    # Ensure index name is set
    observations_df.index.name = "date"
    revisions_df.index.name = "date"

    return observations_df, revisions_df




""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ series transformations                                                                                           │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
 """

def _fred_series_info(series_id: str) -> dict:
    """
    Wrapper to fetch basic series metadata. Uses local get_series helper.
    """
    return get_series(series_id)

def _default_periods_for_transform(transform: str, freq_short: str | None) -> int:
    # freq_short is typically 'D','W','M','Q','A' from FRED metadata
    f = (freq_short or "").upper()
    # Defaults by transform
    if transform in ("dod", "daily", "d/d"):
        return 1
    if transform in ("wow", "weekly"):
        return 1
    if transform in ("mom", "monthly"):
        return 1
    if transform in ("qoq", "quarterly"):
        return 1
    if transform in ("yoy", "annual"):
        if f == "M": return 12
        if f == "Q": return 4
        if f == "W": return 52
        if f == "D": return 365
        if f == "A": return 1
        return 12
    # Generic pct/diff default to 1 step
    return 1

def _annualize_factor(freq_short: str | None) -> int:
    f = (freq_short or "").upper()
    return {"D": 365, "W": 52, "ME": 12, "QE": 4, "A": 1}.get(f) 


def _merge_dataframes(df_dict):
    """
    Merge a dictionary of DataFrames (same index) into one DataFrame.
    Each DataFrame's single column is renamed to the dict key.
    
    Args:
        df_dict: dict of {column_name: DataFrame}
                 Each DataFrame must have a single column and the same index.
    Returns:
        pd.DataFrame with all columns merged and renamed.
    """
    out = []
    for col, df in df_dict.items():
        # If DataFrame has more than one column, keep only the first
        if df.shape[1] > 1:
            df = df.iloc[:, [0]]
        # Rename column to the dict key
        df = df.rename(columns={df.columns[0]: col})
        out.append(df)
    # Merge on index
    merged = pd.concat(out, axis=1)
    return merged


def fetch_transform_fred(
    series_id: str,
    observation_start: str = "1950-01-01",
    vintage_date: str | None = None,          # pass a YYYY-MM-DD to snapshot a vintage
    resample_to: str | None = None,           # e.g., "D","W","M","Q","A"
    resample_agg: str = "last",               # "last" or "mean"
    transform: str = "level",                 # "level","diff","pct","dod","wow","mom","qoq","yoy","logdiff","logpct"
    periods: int | None = None,               # override default periods (e.g., 12 for YoY)
    pct_scale: float = 100.0,                 # 1.0 for fraction, 100.0 for percent
    annualize: bool = False,                  # annualize pct changes (compounded)
    dropna: bool = True,
    return_raw: bool = False
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      - date: reference period
      - value: transformed series
      - raw_value: optional original level before transform (if return_raw=True)
    Notes:
      - If resample_to is set, aggregation is applied before transform.
      - For YoY/MoM/QoQ aliases, periods are auto-chosen from FRED frequency unless overridden.
      - For daily YoY you may prefer to resample_to='M' and use transform='yoy'.
    """
    # 1) Fetch observations (latest or specific vintage)
    obs = get_observations(series_id, vintage_date=vintage_date, observation_start=observation_start)
    if obs.empty:
        cols = ["date", "value"] + (["raw_value"] if return_raw else [])
        return pd.DataFrame(columns=cols)

    df = obs[["date", "value"]].copy()
    df = df.sort_values("date").dropna(subset=["value"])
    df = df.set_index("date")

    # 2) Optional resample
    if resample_to:
        agg = {"last": "last", "mean": "mean", 'sum': 'sum'}[resample_agg]
        df = df.resample(resample_to).agg(agg)

    # 3) Determine frequency for defaults/annualization from metadata or resample_to
    info = _fred_series_info(series_id)
    freq_short = (resample_to or info.get("frequency_short") or "").upper() or None

    # 4) Compute transform
    s = df["value"].astype(float)
    tr = transform.lower()

    # Aliases -> base operation
    alias_to_pct = {"dod","wow","mom","qoq","yoy"}
    if tr in alias_to_pct:
        if periods is None:
            periods = _default_periods_for_transform(tr, freq_short)
        # percent change or log-percent change
        if tr in {"yoy","qoq","mom","wow","dod"}:
            # plain pct change
            s_out = s.pct_change(periods=periods)
        else:
            s_out = s.pct_change(periods=periods)
        # annualize if requested
        if annualize:
            ann = _annualize_factor(freq_short)
            s_out = (1.0 + s_out) ** ann - 1.0
        s_out = s_out * pct_scale
    elif tr in ("pct", "percent", "pct_change"):
        p = periods or 1
        s_out = s.pct_change(periods=p)
        if annualize:
            ann = _annualize_factor(freq_short)
            s_out = (1.0 + s_out) ** ann - 1.0
        s_out = s_out * pct_scale
    elif tr in ("logdiff", "log_diff"):
        p = periods or 1
        s_pos = s.where(s > 0)
        s_out = np.log(s_pos).diff(p) * pct_scale
    elif tr in ("logpct", "log_ret", "logreturn"):
        p = periods or 1
        s_pos = s.where(s > 0)
        s_out = (np.log(s_pos) - np.log(s_pos.shift(p)))  # in log points
        if annualize:
            ann = _annualize_factor(freq_short)
            s_out = s_out * ann
        s_out = s_out * pct_scale
    elif tr in ("diff", "difference"):
        p = periods or 1
        s_out = s.diff(p)
    elif tr in ("level", "none", "identity"):
        s_out = s.copy()
    else:
        raise ValueError(f"Unknown transform '{transform}'")

    out = pd.DataFrame(index=s_out.index)
    out["value"] = s_out
    if return_raw:
        out["raw_value"] = s
    out = out.reset_index().rename(columns={"index": "date"})
    if dropna:
        out = out.dropna(subset=["value"])
    return out.set_index('date')



def get_macro_factors():
    """
    Retrieve and merge key macroeconomic indicators from FRED.
    Returns a DataFrame with monthly and quarterly factors, forward-filled.
    """
    factors = {
        # Employment
        "NFP_CHG":      {"series_id": "PAYEMS",        "resample_to": "ME", "transform": "diff", "pct_scale": 100.0, "observation_start": "1959-01-01"},
        "ADP_CHG":      {"series_id": "ADPMNUSNERSA",  "resample_to": "ME", "transform": "diff", "pct_scale": 100.0, "observation_start": "1959-01-01"},
        "UNEMP_CHG":    {"series_id": "UNRATE",        "resample_to": "ME", "transform": "diff", "pct_scale": 1,     "observation_start": "1959-01-01"},
        "UNEMP_LVL":    {"series_id": "UNRATE",        "resample_to": "ME", "transform": "none", "pct_scale": 1,     "observation_start": "1959-01-01"},

        # GDP
        "GDP_QOQ":      {"series_id": "GDP",           "resample_to": "QE", "transform": "qoq",  "annualize": True,  "pct_scale": 1, "observation_start": "1959-01-01"},
        "RGDP_QOQ":     {"series_id": "GDPC1",         "resample_to": "QE", "transform": "qoq",  "annualize": True,  "pct_scale": 1, "observation_start": "1959-01-01"},
        # Inflation
        "CPI_MOM":      {"series_id": "CPIAUCSL",      "resample_to": "ME", "transform": "mom",  "pct_scale": 1,     "observation_start": "1959-01-01"},
        "CPI_YOY":      {"series_id": "CPIAUCSL",      "resample_to": "ME", "transform": "yoy",  "pct_scale": 1,     "observation_start": "1959-01-01"},
        "CCPI_MOM":     {"series_id": "CPILFESL",      "resample_to": "ME", "transform": "mom",  "pct_scale": 1,     "observation_start": "1959-01-01"},
        "CCPI_YOY":     {"series_id": "CPILFESL",      "resample_to": "ME", "transform": "yoy",  "pct_scale": 1,     "observation_start": "1959-01-01"},
        "PCE_MOM":      {"series_id": "PCE",           "resample_to": "ME", "transform": "mom",  "pct_scale": 1,     "observation_start": "1959-01-01"},
        "PCE_YOY":      {"series_id": "PCE",           "resample_to": "ME", "transform": "yoy",  "pct_scale": 1,     "observation_start": "1959-01-01"},
        "CPCE_MOM":     {"series_id": "PCEPILFE",      "resample_to": "ME", "transform": "mom",  "pct_scale": 1,     "observation_start": "1959-01-01"},
        "CPCE_YOY":     {"series_id": "PCEPILFE",      "resample_to": "ME", "transform": "yoy",  "pct_scale": 1,     "observation_start": "1959-01-01"},
        # Money & Production
        "M2_YOY":       {"series_id": "M2SL",          "resample_to": "ME", "transform": "yoy",  "pct_scale": 1,     "observation_start": "1959-01-01"},
        "INDPRO_YOY":   {"series_id": "INDPRO",        "resample_to": "ME", "transform": "yoy",  "pct_scale": 1,     "observation_start": "1959-01-01"},
        # Sentiment & Retail
        "UMICH_CHG":    {"series_id": "UMCSENT",       "resample_to": "ME", "transform": "diff", "pct_scale": 1,     "observation_start": "1959-01-01"},
        "UMICH_LVL":    {"series_id": "UMCSENT",       "resample_to": "ME", "transform": "none", "pct_scale": 1,     "observation_start": "1959-01-01"},
        "RETAIL_MOM":   {"series_id": "RRSFS",         "resample_to": "ME", "transform": "mom",  "pct_scale": 1,     "observation_start": "1959-01-01"},
        # Labor & Wages
        "AHE_MOM":      {"series_id": "CES0500000003", "resample_to": "ME", "transform": "mom",  "pct_scale": 1,     "observation_start": "1959-01-01"},
        "AHE_YOY":      {"series_id": "CES0500000003", "resample_to": "ME", "transform": "yoy",  "pct_scale": 1,     "observation_start": "1959-01-01"},
        "LABOR_PROD_YOY":{"series_id": "OPHNFB",       "resample_to": "QE", "transform": "yoy",  "annualize": False,  "pct_scale": 1, "observation_start": "1959-01-01"},
        # Patents
        "PAT_QOQ":      {"series_id": "CP",            "resample_to": "ME", "transform": "qoq",  "pct_scale": 1,     "observation_start": "1959-01-01"},
        "PAT_YOY":      {"series_id": "CP",            "resample_to": "ME", "transform": "yoy",  "pct_scale": 1,     "observation_start": "1959-01-01"},
        # Rates
        "FEDFUNDS_LVL":     {"series_id": "FEDFUNDS",      "resample_to": "ME", "transform": "none", "observation_start": "1959-01-01"},
        "GS10_LVL":         {"series_id": "GS10",          "resample_to": "ME", "transform": "none", "observation_start": "1959-01-01"},
        "DTB3_LVL":         {"series_id": "DTB3",          "resample_to": "ME", "transform": "none", "observation_start": "1959-01-01"},
        # Housing & Trade
        "HOUST_MOM":    {"series_id": "HOUST",         "resample_to": "ME", "transform": "mom",  "observation_start": "1959-01-01"},
        "NEWHOME_MOM":  {"series_id": "HSN1F",         "resample_to": "ME", "transform": "mom",  "observation_start": "1959-01-01"},
        "TRADE_QOQ":    {"series_id": "BOPGSTB",       "resample_to": "ME", "transform": "qoq",  "observation_start": "1959-01-01"},
        "EXRATE":       {"series_id": "DTWEXBGS",      "resample_to": "ME", "transform": "none", "observation_start": "1959-01-01"},
        "WHOLESALE_INV_LVL":{"series_id": "WHLSLRIRSA",    "resample_to": "ME", "transform": "none",  "observation_start": "1959-01-01"},
        # Credit & Labor
        "CORP_SPREAD_LVL":  {"series_id": "BAA10Y",        "resample_to": "ME", "transform": "none", "observation_start": "1959-01-01"},
        "LABOR_FORCE_MOM":{"series_id": "CLF16OV",     "resample_to": "ME", "transform": "mom",  "observation_start": "1959-01-01"},
        "JOB_OPENINGS_MOM":{"series_id": "JTSJOL",     "resample_to": "ME", "transform": "mom",  "observation_start": "1959-01-01"},
        "LEADING_MOM":  {"series_id": "USSLIND",       "resample_to": "ME", "transform": "mom",  "observation_start": "1959-01-01"},
        # Markets
        "SP500":        {"series_id": "SP500",         "resample_to": "ME", "transform": "yoy",  "observation_start": "1959-01-01"},
        "VIX_LVL":          {"series_id": "VIXCLS",        "resample_to": "ME", "transform": "none", "observation_start": "1990-01-01"},
    }

    res = {}
    for name, params in factors.items():
        res[name] = fetch_transform_fred(**params)

    res['ADP_CHG'] = res['ADP_CHG'].divide(1000)
    res['GS10_LVL'] = res['GS10_LVL'].divide(100)
    res['DTB3_LVL'] = res['DTB3_LVL'].divide(100)
    res['FEDFUNDS_LVL'] = res['FEDFUNDS_LVL'].divide(100)

    return _merge_dataframes(res).ffill()



def get_cycle_feature_matrix(df_fred: pd.DataFrame, cycles: pd.DataFrame, start_col: str = 'start_date', end_col: str = 'end_date') -> pd.DataFrame:
    """
    Compute a feature matrix for each economic cycle period.
    Each row is a cycle, columns are summary statistics for each macro factor.
    Includes a row for the most recent period (last cycle end to present).
    
    Parameters:
        df (pd.DataFrame): Economic data, indexed by datetime.
        cycles (pd.DataFrame): Cycle periods with start/end columns.
        start_col (str): Name of cycle start column.
        end_col (str): Name of cycle end column.
    
    Returns:
        pd.DataFrame: Feature matrix (cycle x features).
    """

    df = df_fred.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    macro_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    def window_features(window: pd.DataFrame, cols):
        feats = {}
        stats = window[cols].agg(["mean", "std", "min", "max", "median"])
        q10 = window[cols].quantile(0.25)
        q90 = window[cols].quantile(0.75)
        for c in cols:
            feats[f"{c}__mean"] = stats.loc["mean", c]
            # feats[f"{c}__median"] = stats.loc["median", c]
            # feats[f"{c}__std"] = stats.loc["std", c]
            # feats[f"{c}__min"] = stats.loc["min", c]
            # feats[f"{c}__max"] = stats.loc["max", c]
            # feats[f"{c}__iqr"] = q90[c] - q10[c]
            s = window[c].dropna()
            if len(s) > 1:
                x = np.arange(len(s))
                feats[f"{c}__slope"] = np.polyfit(x, s.values, 1)[0]
            else:
                feats[f"{c}__slope"] = np.nan
        return pd.Series(feats)

    cycle_rows = []
    labels = []
    for _, row in cycles.iterrows():
        mask = (df.index >= row[start_col]) & (df.index <= row[end_col])
        w = df.loc[mask, macro_cols]
        if w.empty:
            continue
        cycle_rows.append(window_features(w, macro_cols))
        labels.append(f"{row[start_col].date()} - {row[end_col].date()}")

    X_cycles = pd.DataFrame(cycle_rows, index=labels)

    # Add most recent period
    last_row = df.iloc[[-1]][macro_cols]
    if not last_row.empty:
        # Fill nulls in last_row with previous row values
        if last_row.isnull().any(axis=None):
            prev_row = df.iloc[[-2]][macro_cols] if len(df) > 1 else last_row
            last_row = last_row.combine_first(prev_row)
        latest_features = window_features(last_row, macro_cols)
        X_cycles.loc["Latest"] = latest_features

    return X_cycles


""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ recessions                                                                                                       │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
 """

def get_recession_periods(
    series_id: str = "USREC",                # "USREC" (monthly) or "USRECDM" (daily)
    observation_start: str = "1947-01-01",
    vintage_date: str | None = None,         # snapshot as of a vintage date (YYYY-MM-DD) if desired
    close_open: bool = True                  # if last recession is ongoing, close at last obs date
) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per recession episode:
      - start_date: first date with indicator == 1 after 0
      - end_date: last date with indicator == 1 before returning to 0 (NaT if open and close_open=False)
      - n_periods: number of periods (days or months) in the episode
    """
    obs = get_observations(series_id, vintage_date=vintage_date, observation_start=observation_start)
    if obs.empty:
        return pd.DataFrame(columns=["start_date", "end_date", "n_periods"])

    s = (obs.set_index("date")["value"]
            .astype(float)
            .fillna(0.0))
    # Ensure binary (some series may be 0/1 already)
    s = (s >= 1.0).astype(int)

    # Starts: 0 -> 1; Ends: 1 -> 0 (mark last 1)
    starts_mask = (s.shift(1, fill_value=0) == 0) & (s == 1)
    ends_mask   = (s.shift(-1, fill_value=0) == 0) & (s == 1)

    start_dates = s.index[starts_mask]
    end_dates = s.index[ends_mask]

    # Handle open-ended recession (more starts than ends)
    if len(start_dates) > len(end_dates):
        if close_open:
            end_dates = end_dates.append(pd.DatetimeIndex([s.index[s == 1][-1]]))
        else:
            # leave last end_date as NaT
            end_dates = end_dates.append(pd.DatetimeIndex([pd.NaT]))

    # Align lengths
    k = min(len(start_dates), len(end_dates))
    start_dates = start_dates[:k]
    end_dates = end_dates[:k]

    out = pd.DataFrame({
        "start_date": start_dates,
        "end_date": end_dates
    })
    # Count periods inclusive of both endpoints
    # (for daily/monthly indicators this is number of days/months in the episode)
    def _count_periods(a: pd.Timestamp, b: pd.Timestamp) -> int:
        if pd.isna(b):
            return int(s.loc[a:].sum())
        return int(s.loc[a:b].sum())

    out["n_periods"] = [ _count_periods(a, b) for a, b in zip(out["start_date"], out["end_date"]) ]
    return out

def recession_shading_ranges(periods: pd.DataFrame) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Convert recession periods into [(start, end), ...] tuples for chart shading.
    If end_date is NaT, uses start_date for a zero-length range.
    """
    ranges: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for _, r in periods.iterrows():
        a = pd.to_datetime(r["start_date"])
        b = pd.to_datetime(r["end_date"]) if pd.notna(r["end_date"]) else a
        ranges.append((a, b))
    return ranges


def get_economic_cycles(
    series_id: str = "USREC",                # "USREC" (monthly) or "USRECDM" (daily)
    observation_start: str = "1947-01-01",
    vintage_date: str | None = None,
    close_open: bool = True                  # if False, leave the last episode end_date as NaT
) -> pd.DataFrame:
    """
    Returns all contiguous episodes (recession and expansion):
      - start_date, end_date
      - n_periods: number of rows (months/days) in the episode
      - is_recession: 1=recession, 0=expansion
      - label: 'recession'/'expansion'
      - The last cycle is expanded to the most recent available date.
    """
    obs = get_observations(series_id, observation_start=observation_start, vintage_date=vintage_date)
    if obs.empty:
        return pd.DataFrame(columns=["start_date","end_date","n_periods","is_recession","label"])

    s = (obs.set_index("date")["value"].astype(float).fillna(0.0) >= 1.0).astype(int)

    # Run-length encode contiguous states
    grp_id = s.ne(s.shift()).cumsum()
    episodes = (s.to_frame("is_recession")
                  .assign(group=grp_id)
                  .groupby("group", as_index=False)
                  .agg(
                      start_date=("is_recession", lambda x: x.index.min()),
                      end_date=("is_recession", lambda x: x.index.max()),
                      n_periods=("is_recession", "size"),
                      is_recession=("is_recession", "first"),
                  )
               )

    # Expand last cycle to the most recent available date
    if not episodes.empty:
        today = pd.Timestamp(datetime.date.today())
        episodes.loc[episodes.index[-1], "end_date"] = today
        episodes.loc[episodes.index[-1], "n_periods"] = int((today - episodes.loc[episodes.index[-1], "start_date"]).days) + 1

    if not close_open and not episodes.empty:
        episodes.loc[episodes.index[-1], "end_date"] = pd.NaT

    episodes["label"] = episodes["is_recession"].map({1: "recession", 0: "expansion"})
    out = episodes[["start_date","end_date","n_periods","is_recession","label"]].reset_index(drop=True)
    return out



def get_economic_cycle_slices(
    series_id: str = "USREC",
    observation_start: str = "1947-01-01",
    vintage_date: str | None = None,
    n_years: int = 5,
    which: str = "first",           # "first" or "last"
    close_open: bool = True,        # forwarded to get_economic_cycles
    clamp_open_to_today: bool = True
) -> pd.DataFrame:
    """
    Return per-cycle start/end windows for the first/last N years of each economic cycle.

    Args:
        series_id: US recession indicator (monthly: 'USREC', daily: 'USRECDM')
        observation_start: earliest date to pull the indicator
        vintage_date: optional vintage snapshot
        n_years: number of years for each slice
        which: 'first' (start of cycle) or 'last' (end of cycle)
        close_open: whether to close the last (possibly ongoing) cycle at 'today'
        clamp_open_to_today: if an end_date is NaT, clamp to today for slicing

    Returns:
        DataFrame with columns:
            - start_date, end_date: sliced window boundaries for each cycle
            - full_start, full_end: original cycle boundaries
            - is_recession, label: cycle type metadata
            - full_n_periods: original cycle length (from get_economic_cycles)
            - slice_n_days: inclusive day count in the slice
    """
    which = which.lower().strip()
    if which not in ("first", "last"):
        raise ValueError("which must be 'first' or 'last'")

    cycles = get_economic_cycles(
        series_id=series_id,
        observation_start=observation_start,
        vintage_date=vintage_date,
        close_open=close_open,
    ).copy()

    if cycles.empty:
        return pd.DataFrame(columns=[
            "start_date","end_date","full_start","full_end",
            "is_recession","label","full_n_periods","slice_n_days"
        ])

    out_rows = []
    years = int(n_years)
    for _, row in cycles.iterrows():
        a = pd.to_datetime(row["start_date"])
        b = pd.to_datetime(row["end_date"])
        if pd.isna(b) and clamp_open_to_today:
            b = pd.Timestamp.today().normalize()

        # Skip if bounds are not usable
        if pd.isna(a) or pd.isna(b) or a > b:
            continue

        if which == "first":
            slice_start = a
            # inclusive end at (start + n years - 1 day), clamped to cycle end
            slice_end = min(b, a + pd.DateOffset(years=years) - pd.Timedelta(days=1))
        else:  # last
            slice_end = b
            # inclusive start at (end - n years + 1 day), clamped to cycle start
            slice_start = max(a, b - pd.DateOffset(years=years) + pd.Timedelta(days=1))

        # If slice collapses (e.g., very short cycle), keep at least a single day
        if slice_start > slice_end:
            slice_start = slice_end

        slice_n_days = int((slice_end - slice_start).days) + 1

        out_rows.append({
            "start_date": slice_start,
            "end_date": slice_end,
            "full_start": a,
            "full_end": b,
            "is_recession": row.get("is_recession", None),
            "label": row.get("label", None),
            "full_n_periods": row.get("n_periods", None),
            "slice_n_days": slice_n_days,
        })

    out = pd.DataFrame(out_rows).reset_index(drop=True)
    return out


""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ realtime tape                                                                                                    │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
 """



# def get_real_time_tape(
#     series_id: str,
#     start_date: str = None,
#     end_date: str = None,
#     max_vintages: int | None = None,
# ) -> pd.DataFrame:
#     """
#     Create a real-time tape showing the latest value that was available for each reference date,
#     preventing lookahead bias by ensuring only data available at each point in time is used.
    
#     This function reconstructs what an analyst would have seen in real-time, using only the 
#     information that was actually available on each date (accounting for publication lags).
    
#     Args:
#         series_id: FRED series identifier (e.g., 'GDP', 'CPIAUCSL')
#         start_date: Start date for the tape (format: 'YYYY-MM-DD'). If None, uses earliest available.
#         end_date: End date for the tape (format: 'YYYY-MM-DD'). If None, uses latest available.
#         max_vintages: If provided and > 0, only process the most recent N vintages (keeps chronological order)
    
#     Returns:
#         DataFrame with columns:
#         - date: The reference/observation date 
#         - value: The latest value available for that date (as it would have been known in real-time)
#         - vintage_date: When this value was first published/became available
#         - publication_lag: Days between observation date and publication date
        
#     Example:
#         >>> tape = get_real_time_tape('GDP', '2020-01-01', '2023-12-31')
#         >>> # Shows GDP values as they would have been known in real-time
#         >>> # preventing use of future revisions in backtesting
#     """
#     # Get all available vintages for this series
#     vintages = get_vintages(series_id)
#     if not vintages:
#         raise ValueError(f"No vintages available for series {series_id}")
#     total_vintages = len(vintages)
#     if max_vintages is not None and max_vintages > 0:
#         vintages = vintages[-max_vintages:]  # keep only most recent N, preserving chronological order
    
#     print(f"Processing {len(vintages)} of {total_vintages} vintages for {series_id}...")
    
#     # Dictionary to store the real-time tape: {observation_date: (value, vintage_date)}
#     real_time_data = {}
    
#     # Process each vintage chronologically
#     for vintage_date in vintages:
#         try:
#             # Get observations for this vintage
#             obs_df = get_observations(series_id, vintage_date)
            
#             if obs_df.empty:
#                 continue
                
#             vintage_dt = pd.to_datetime(vintage_date)
            
#             # For each observation in this vintage
#             for obs_date, row in obs_df.iterrows():
#                 obs_value = row[vintage_date]
                
#                 # Skip if value is NaN
#                 if pd.isna(obs_value):
#                     continue
                
#                 # Only record this value if:
#                 # 1. We haven't seen this observation date before, OR
#                 # 2. This vintage date is more recent (later revision)
#                 if (obs_date not in real_time_data or 
#                     vintage_dt > pd.to_datetime(real_time_data[obs_date][1])):
                    
#                     # But only if the vintage date is after the observation date
#                     # (can't know about data before it's published)
#                     if vintage_dt >= obs_date:
#                         real_time_data[obs_date] = (obs_value, vintage_date)
        
#         except Exception as e:
#             print(f"Warning: Error processing vintage {vintage_date}: {e}")
#             continue
    
#     if not real_time_data:
#         raise ValueError(f"No valid real-time data found for series {series_id}")
    
#     # Convert to DataFrame
#     tape_data = []
#     for obs_date, (value, vintage_date) in real_time_data.items():
#         publication_lag = (pd.to_datetime(vintage_date) - obs_date).days
#         tape_data.append({
#             'date': obs_date,
#             'value': value,
#             'vintage_date': vintage_date,
#             'publication_lag': publication_lag
#         })
    
#     df = pd.DataFrame(tape_data)
#     df = df.set_index('date').sort_index()
    
#     # Filter by date range if specified
#     if start_date:
#         df = df[df.index >= pd.to_datetime(start_date)]
#     if end_date:
#         df = df[df.index <= pd.to_datetime(end_date)]
    
#     # Add metadata
#     df.attrs = {
#         "series_id": series_id,
#         "description": "Real-time tape preventing lookahead bias",
#         "total_vintages_available": total_vintages,
#         "vintages_used": len(vintages),
#         "max_vintages": max_vintages,
#         "date_range": f"{df.index.min()} to {df.index.max()}" if not df.empty else "No data"
#     }
    
#     df.index.name = "date"
#     print(f"Real-time tape created: {len(df)} observations from {df.index.min()} to {df.index.max()}")
    
#     return df


# # def get_real_time_matrix(
# #     series_ids: list,
# #     start_date: str = None,
# #     end_date: str = None,
# #     max_vintages: int | None = None,
# # ) -> pd.DataFrame:
# #     """
# #     Create a real-time data matrix for multiple series, ensuring no lookahead bias.
    
# #     Args:
# #         series_ids: List of FRED series identifiers
# #         start_date: Start date (format: 'YYYY-MM-DD')
# #         end_date: End date (format: 'YYYY-MM-DD')
# #         max_vintages: If provided and > 0, only process the most recent N vintages per series
    
# #     Returns:
# #         DataFrame with date index and columns for each series containing real-time values
# #     """
# #     if not series_ids:
# #         raise ValueError("Must provide at least one series ID")
    
# #     print(f"Creating real-time matrix for {len(series_ids)} series (max_vintages={max_vintages})...")
    
# #     # Get real-time tape for each series
# #     tapes = {}
# #     for series_id in series_ids:
# #         try:
# #             tape = get_real_time_tape(series_id, start_date, end_date, max_vintages=max_vintages)
# #             tapes[series_id] = tape['value']  # Just the values
# #         except Exception as e:
# #             print(f"Warning: Could not process {series_id}: {e}")
# #             continue
    
# #     if not tapes:
# #         raise ValueError("No valid series data found")
    
# #     # Combine into matrix
# #     matrix = pd.DataFrame(tapes)
    
# #     # Add metadata
# #     matrix.attrs = {
# #         "description": "Real-time data matrix preventing lookahead bias",
# #         "series_ids": list(tapes.keys()),
# #         "max_vintages": max_vintages,
# #         "date_range": f"{matrix.index.min()} to {matrix.index.max()}" if not matrix.empty else "No data"
# #     }
    
# #     matrix.index.name = "date"
# #     print(f"Real-time matrix created: {matrix.shape[0]} dates × {matrix.shape[1]} series")
    
# #     return matrix


