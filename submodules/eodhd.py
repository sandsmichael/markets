from itertools import chain
import sys, os
import requests
import pandas as pd
import json
import talib
import numpy as np
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union, Optional
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import yfinance as yf
import warnings

import warnings
warnings.filterwarnings('ignore')

# Ensure imports from the same directory as this file
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from ta import TechnicalAnalysis
from fa import FundamentalAnalysis

# Find secrets.json in multiple locations
def find_secrets():
    """Find secrets.json in current dir, parent dir, or common locations."""
    possible_paths = [
        "secrets.json",  # Current directory
        "./secrets.json",  # Current directory (explicit)
        os.path.join(os.path.dirname(_current_dir), "secrets.json"),  # Parent of eodhd.py directory (markets/)
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
API_KEY = secrets['eodhd_api_key']


# Helper functions for Fundamentals class
def _parse_date(date_str: str):
    """Parse a date string into datetime object, returning None on failure."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except Exception:
        return None


def date_to_calendar_quarter(date_str: str) -> str:
    """
    Convert a date directly to calendar quarter format (YYYYQ#) based on the month.
    
    This is a simple conversion that maps the date's month to its calendar quarter
    without any period-middle adjustments. Use this for report dates, not period dates.
    
    Parameters
    ----------
    date_str : str
        Date string in YYYY-MM-DD format
        
    Returns
    -------
    str
        Calendar quarter in format YYYYQ1, YYYYQ2, YYYYQ3, or YYYYQ4
        
    Examples
    --------
    >>> date_to_calendar_quarter('2025-09-05')  # September
    '2025Q3'
    >>> date_to_calendar_quarter('2025-12-15')  # December
    '2025Q4'
    """
    dt = _parse_date(date_str)
    if not dt:
        return None
    
    # Calendar quarters: Q1=1-3, Q2=4-6, Q3=7-9, Q4=10-12
    quarter = (dt.month - 1) // 3 + 1
    return f"{dt.year}Q{quarter}"


def subtract_calendar_quarter(quarter_str: str, num_quarters: int = 1) -> str:
    """
    Subtract quarters from a calendar quarter string.
    
    Parameters
    ----------
    quarter_str : str
        Calendar quarter in format YYYYQ# (e.g., '2025Q2')
    num_quarters : int
        Number of quarters to subtract (default: 1)
        
    Returns
    -------
    str
        New calendar quarter after subtraction
        
    Examples
    --------
    >>> subtract_calendar_quarter('2025Q2', 1)
    '2025Q1'
    >>> subtract_calendar_quarter('2025Q1', 1)
    '2024Q4'
    """
    if not quarter_str or 'Q' not in quarter_str:
        return None
    
    try:
        year, quarter = quarter_str.split('Q')
        year = int(year)
        quarter = int(quarter)
        
        # Calculate total quarters from year 0
        total_quarters = year * 4 + (quarter - 1)
        # Subtract the specified number of quarters
        total_quarters -= num_quarters
        
        # Convert back to year and quarter
        new_year = total_quarters // 4
        new_quarter = (total_quarters % 4) + 1
        
        return f"{new_year}Q{new_quarter}"
    except (ValueError, AttributeError):
        return None


def standardize_to_calendar_quarter(date_str: str) -> str:
    """
    Convert a date to standardized calendar quarter format (YYYYQ#).
    
    This function standardizes dates to calendar quarters (Q1=Jan-Mar, Q2=Apr-Jun, 
    Q3=Jul-Sep, Q4=Oct-Dec) based on the period the financial data covers, not just
    the period end date. This handles companies with different fiscal year ends correctly.
    
    For quarterly data, we assume the period end date represents the END of a ~3-month
    period. To determine which calendar quarter the data belongs to, we look at the
    middle of that period (approximately 1 month before the end date).
    
    Parameters
    ----------
    date_str : str
        Date string in YYYY-MM-DD format (typically the fiscal period end date)
        
    Returns
    -------
    str
        Calendar quarter in format YYYYQ1, YYYYQ2, YYYYQ3, or YYYYQ4
        
    Examples
    --------
    >>> standardize_to_calendar_quarter('2024-03-31')  # Jan-Mar period
    '2024Q1'
    >>> standardize_to_calendar_quarter('2025-07-31')  # May-Jul period (AVGO fiscal Q3)
    '2025Q2'  # Because the period MIDDLE (Jun) is in Q2
    >>> standardize_to_calendar_quarter('2025-08-31')  # Jun-Aug period (ADBE fiscal Q3)
    '2025Q3'  # Because the period MIDDLE (Jul) is in Q3
    """
    dt = _parse_date(date_str)
    if not dt:
        return None
    
    # For quarterly data, look at the period MIDDLE (approx 1 month before end)
    # to determine which calendar quarter this data primarily represents
    from dateutil.relativedelta import relativedelta
    period_middle_approx = dt - relativedelta(months=1)
    
    # Calendar quarters: Q1=1-3, Q2=4-6, Q3=7-9, Q4=10-12
    quarter = (period_middle_approx.month - 1) // 3 + 1
    
    return f"{period_middle_approx.year}Q{quarter}"


def standardize_to_fiscal_quarter(date_str: str, fiscal_year_end_month: int = 12) -> str:
    """
    Convert a date to standardized fiscal quarter format (YYYYQ#).
    
    This function converts dates to fiscal quarters based on the company's fiscal
    year end. Different companies with different fiscal year ends will show different
    quarters for the same calendar date.
    
    Parameters
    ----------
    date_str : str
        Date string in YYYY-MM-DD format
    fiscal_year_end_month : int
        Month when fiscal year ends (1-12). Default is 12 (calendar year)
        
    Returns
    -------
    str
        Fiscal quarter in format YYYYQ1, YYYYQ2, YYYYQ3, or YYYYQ4
        
    Examples
    --------
    Calendar year company (fiscal year ends Dec 31):
    >>> standardize_to_fiscal_quarter('2024-03-31', 12)
    '2024Q1'
    >>> standardize_to_fiscal_quarter('2024-06-30', 12)
    '2024Q2'
    
    Non-calendar year company (fiscal year ends Sep 30):
    >>> standardize_to_fiscal_quarter('2024-03-31', 9)  # Q2 of FY2024 (Oct 2023 - Sep 2024)
    '2024Q2'
    >>> standardize_to_fiscal_quarter('2024-09-30', 9)  # Q4 of FY2024
    '2024Q4'
    """
    dt = _parse_date(date_str)
    if not dt:
        return None
    
    # Calculate months from fiscal year end
    # Fiscal quarters are counted backward from fiscal year end
    months_from_fye = (dt.month - fiscal_year_end_month) % 12
    
    # Determine fiscal year
    if dt.month > fiscal_year_end_month:
        fiscal_year = dt.year + 1
    else:
        fiscal_year = dt.year
    
    # Determine quarter (1-4)
    # Q4 ends at fiscal_year_end_month
    # Q3 ends 3 months before, Q2 ends 6 months before, Q1 ends 9 months before
    if months_from_fye <= 0:
        quarter = 4
    elif months_from_fye <= 3:
        quarter = 1
    elif months_from_fye <= 6:
        quarter = 2
    else:
        quarter = 3
    
    return f"{fiscal_year}Q{quarter}"


def _find_closest_date(target_date: str, data_dict: dict, field_name: str):
    """Find the closest date to target_date that has a non-null value for field_name."""
    target_dt = _parse_date(target_date)
    if not target_dt:
        return None, None

    min_diff = None
    best_date = None
    best_value = None

    for date_str, record in data_dict.items():
        dt = _parse_date(date_str)
        if not dt:
            continue
        
        value = record.get(field_name)
        if value is not None:
            diff = abs((dt - target_dt).days)
            if min_diff is None or diff < min_diff:
                min_diff = diff
                best_date = date_str
                best_value = value

    return best_date, best_value


def _calculate_ltm(data_dict: dict, field_name: str, use_date: str = None, num_periods: int = 4) -> Optional[float]:
    """
    Calculate Last Twelve Months (LTM) sum for a field.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary with date keys and records containing field values
    field_name : str
        Name of the field to sum
    use_date : str, optional
        Only include dates on or before this date
    num_periods : int
        Number of periods to sum (default 4 for quarterly data)
    
    Returns
    -------
    float or None
        Sum of the most recent periods, or None if insufficient data
    """
    use_dt = _parse_date(use_date) if use_date else None
    
    # Collect all valid dates with non-null values
    valid_dates = []
    for date_str in data_dict:
        dt = _parse_date(date_str)
        if not dt:
            continue
        
        value = data_dict[date_str].get(field_name) if isinstance(data_dict[date_str], dict) else data_dict[date_str]
        if value is not None and (use_dt is None or dt <= use_dt):
            valid_dates.append((dt, value))
    
    # Sort by date descending and take the most recent periods
    valid_dates = sorted(valid_dates, key=lambda x: x[0], reverse=True)
    ltm_values = [v for _, v in valid_dates[:num_periods]]
    
    # Sum the values if we have any
    if ltm_values:
        try:
            return sum(float(x) for x in ltm_values if x is not None)
        except Exception:
            return None
    return None


def _calculate_bf_eps_estimate(history_data: dict, use_date: str = None) -> Optional[float]:
    """
    Calculate blended forward EPS estimate: 1 forward estimate + 3 most recent actuals.
    
    Parameters
    ----------
    history_data : dict
        Earnings history data with epsEstimate and epsActual
    use_date : str, optional
        Reference date for calculating actuals
    
    Returns
    -------
    float or None
        Blended forward EPS estimate, or None if insufficient data
    """
    use_dt = _parse_date(use_date) if use_date else None
    
    # Find forward-looking estimate (has estimate but no actual)
    forward_estimate = None
    future_dates = []
    for date_str, record in history_data.items():
        dt = _parse_date(date_str)
        if not dt:
            continue
        
        eps_actual = record.get('epsActual')
        eps_estimate = record.get('epsEstimate')
        if eps_estimate is not None and eps_actual is None:
            future_dates.append((dt, eps_estimate))
    
    # Get earliest future estimate
    if future_dates:
        future_dates = sorted(future_dates, key=lambda x: x[0])
        forward_estimate = future_dates[0][1]
    
    # Get most recent actual values
    actual_values = []
    for date_str, record in history_data.items():
        dt = _parse_date(date_str)
        if not dt:
            continue
        
        eps_actual = record.get('epsActual')
        if eps_actual is not None and (use_dt is None or dt <= use_dt):
            actual_values.append((dt, eps_actual))
    
    # Sort by date descending
    actual_values = sorted(actual_values, key=lambda x: x[0], reverse=True)
    
    # Calculate blended estimate
    if forward_estimate is not None and len(actual_values) >= 3:
        recent_actuals = [v for _, v in actual_values[:3]]
        try:
            return float(forward_estimate) + sum(float(x) for x in recent_actuals)
        except Exception:
            return None
    return None


class EODHD:

    def __init__(self):
        pass


    def paginate(self, url, params_base, offset=0, total=None):
        """Fetch options data with pagination, handling API's 10K limit. Chunk request down into smaller sizes and iterate.
        
        System will not allow a page offset > 10k so the recursion is rendered obsolete.
        """
        all_rows = []
        
        # Get total count if not provided
        if total is None:
            params = params_base.copy()
            params["api_token"] = API_KEY
            params["page[offset]"] = offset
            params["page[limit]"] = 1000
            response = requests.get(url, params=params)
            data = response.json()
            total = data.get("meta", {}).get("total", 0)
            print(f"Total records to fetch: {total}")
        
        # Process up to 10K records starting from current offset
        max_offset = min(offset + 10000, total)
        current_offset = offset
        no_data_returned = False  # Track if we got no data
        
        while current_offset < max_offset:
            params = params_base.copy()
            params["api_token"] = API_KEY
            params["page[offset]"] = current_offset
            params["page[limit]"] = 1000
            
            time.sleep(0.3)  # Pause between requests
            response = requests.get(url, params=params)
            data = response.json()
            rows = [item['attributes'] for item in data.get("data", []) if item and item.get('attributes')]
            
            # If we get no rows back, break out of this loop
            if not rows:
                print(f"No rows returned at offset {current_offset}, breaking current batch")
                no_data_returned = True
                break
                
            all_rows.extend(rows)
            print(f"Fetched {len(rows)} rows at offset {current_offset}")
            current_offset += 1000
        
        # Only continue with recursion if we haven't reached the total AND we got data in this batch
        if current_offset < total and not no_data_returned:
            print(f"Reached limit at offset {current_offset}, continuing with new request...")
            more_rows = self.paginate(url, params_base, offset=current_offset, total=total)
            # Only extend if more_rows is not empty
            if more_rows:  # This checks if the list has any items
                all_rows.extend(more_rows)
        elif no_data_returned:
            print(f"No more data available, stopping pagination at offset {current_offset}")

        return all_rows  # Return list, not DataFrame



class Fundamentals:
    """
    Professional class for fetching and processing fundamental data from EODHD API.
    
    Can be used standalone for individual securities or via Index class for bulk constituent analysis.
    Supports financial statements, earnings data, and various fundamental metrics with intelligent
    date matching and Last Twelve Months (LTM) calculations.
    
    Parameters
    ----------
    api_key : str, optional
        EODHD API key. If not provided, uses global API_KEY.
        
    """
    
    BASE_URL = "https://eodhd.com/api"
    
    # Financial fields that should have LTM calculations (Last Twelve Months)
    _LTM_FINANCIAL_FIELDS = {
        'totalRevenue', 'operatingIncome', 'netIncome', 'grossProfit', 'ebit', 'ebitda',
        'costOfRevenue', 'totalOperatingExpenses', 'researchDevelopment',
        'netIncomeFromContinuingOps', 'netIncomeApplicableToCommonShares',
        'freeCashFlow', 'totalCashFromOperatingActivities', 'totalCashFromFinancingActivities',
        'totalCashflowsFromInvestingActivities', 'capitalExpenditures', 'changeInWorkingCapital',
        'otherCashflowsFromFinancingActivities'
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Fundamentals fetcher."""
        self.api_key = api_key or API_KEY
        self._fye_cache = {}  # Cache for fiscal year ends

    def _get_fiscal_year_end(self, ticker: str) -> int:
        """
        Get the fiscal year end month for a company.
        
        Parameters
        ----------
        ticker : str
            Stock ticker (e.g., 'AAPL.US')
            
        Returns
        -------
        int
            Month number (1-12) when the fiscal year ends. Defaults to 12 if not found.
        """
        # Check cache first
        if ticker in self._fye_cache:
            return self._fye_cache[ticker]
        
        url = f"{self.BASE_URL}/fundamentals/{ticker}?api_token={self.api_key}&fmt=json&filter=General"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            fye = data.get('FiscalYearEnd')
            if fye:
                # FiscalYearEnd is usually in format like "December" or "June"
                month_map = {
                    'January': 1, 'February': 2, 'March': 3, 'April': 4,
                    'May': 5, 'June': 6, 'July': 7, 'August': 8,
                    'September': 9, 'October': 10, 'November': 11, 'December': 12
                }
                fye_month = month_map.get(fye, 12)
                self._fye_cache[ticker] = fye_month
                return fye_month
            
            # Default to calendar year
            self._fye_cache[ticker] = 12
            return 12
        except Exception:
            # Default to calendar year on error
            self._fye_cache[ticker] = 12
            return 12

    def _parse_date(self, date_str: str):
        """Parse a date string into datetime object, returning None on failure."""
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except Exception:
            return None

    def _find_closest_date(self, target_date: str, data_dict: dict, field_name: str):
        """Find the closest date to target_date that has a non-null value for field_name."""
        target_dt = self._parse_date(target_date)
        if not target_dt:
            return None, None

        min_diff = None
        best_date = None
        best_value = None

        for date_str, record in data_dict.items():
            dt = self._parse_date(date_str)
            if not dt:
                continue
            
            value = record.get(field_name)
            if value is not None:
                diff = abs((dt - target_dt).days)
                if min_diff is None or diff < min_diff:
                    min_diff = diff
                    best_date = date_str
                    best_value = value

        return best_date, best_value

    def _calculate_ltm(self, data_dict: dict, field_name: str, use_date: str = None, num_periods: int = 4) -> Optional[float]:
        """
        Calculate Last Twelve Months (LTM) sum for a field.
        
        Parameters
        ----------
        data_dict : dict
            Dictionary with date keys and records containing field values
        field_name : str
            Name of the field to sum
        use_date : str, optional
            Only include dates on or before this date
        num_periods : int
            Number of periods to sum (default 4 for quarterly data)
        
        Returns
        -------
        float or None
            Sum of the most recent periods, or None if insufficient data
        """
        use_dt = self._parse_date(use_date) if use_date else None
        
        # Collect all valid dates with non-null values
        valid_dates = []
        for date_str in data_dict:
            dt = self._parse_date(date_str)
            if not dt:
                continue
            
            value = data_dict[date_str].get(field_name) if isinstance(data_dict[date_str], dict) else data_dict[date_str]
            if value is not None and (use_dt is None or dt <= use_dt):
                valid_dates.append((dt, value))
        
        # Sort by date descending and take the most recent periods
        valid_dates = sorted(valid_dates, key=lambda x: x[0], reverse=True)
        ltm_values = [v for _, v in valid_dates[:num_periods]]
        
        # Sum the values if we have any
        if ltm_values:
            try:
                return sum(float(x) for x in ltm_values if x is not None)
            except Exception:
                return None
        return None

    def _calculate_bf_eps_estimate(self, history_data: dict, use_date: str = None) -> Optional[float]:
        """
        Calculate blended forward EPS estimate: 1 forward estimate + 3 most recent actuals.
        
        Parameters
        ----------
        history_data : dict
            Earnings history data with epsEstimate and epsActual
        use_date : str, optional
            Reference date for calculating actuals
        
        Returns
        -------
        float or None
            Blended forward EPS estimate, or None if insufficient data
        """
        use_dt = self._parse_date(use_date) if use_date else None
        
        # Find forward-looking estimate (has estimate but no actual)
        forward_estimate = None
        future_dates = []
        for date_str, record in history_data.items():
            dt = self._parse_date(date_str)
            if not dt:
                continue
            
            eps_actual = record.get('epsActual')
            eps_estimate = record.get('epsEstimate')
            if eps_estimate is not None and eps_actual is None:
                future_dates.append((dt, eps_estimate))
        
        # Get earliest future estimate
        if future_dates:
            future_dates = sorted(future_dates, key=lambda x: x[0])
            forward_estimate = future_dates[0][1]
        
        # Get most recent actual values
        actual_values = []
        for date_str, record in history_data.items():
            dt = self._parse_date(date_str)
            if not dt:
                continue
            
            eps_actual = record.get('epsActual')
            if eps_actual is not None and (use_dt is None or dt <= use_dt):
                actual_values.append((dt, eps_actual))
        
        # Sort by date descending
        actual_values = sorted(actual_values, key=lambda x: x[0], reverse=True)
        
        # Calculate blended estimate
        if forward_estimate is not None and len(actual_values) >= 3:
            recent_actuals = [v for _, v in actual_values[:3]]
            try:
                return float(forward_estimate) + sum(float(x) for x in recent_actuals)
            except Exception:
                return None
        return None

    def _extract_financials_field(self, full_data: dict, field_name: str, statement_type: str, 
                                  period: str, date: str = None) -> tuple:
        """
        Extract a field from Financials section with LTM calculation.
        
        Returns tuple: (result_dict, field_date)
        result_dict contains: field_name and optionally ltm_{field_name}
        field_date is the date string for this field (or None)
        """
        result = {}
        
        if ('Financials' not in full_data or 
            statement_type not in full_data['Financials'] or
            period not in full_data['Financials'][statement_type]):
            result[field_name] = None
            if field_name in self._LTM_FINANCIAL_FIELDS and period == 'quarterly':
                result[f"ltm_{field_name}"] = None
            return result, None
        
        period_data = full_data['Financials'][statement_type][period]
        
        # Find the date to use: exact match, or closest date within reasonable window
        use_date = None
        if date and date in period_data:
            # Exact match found
            use_date = date
        elif period_data:
            if date:
                target_dt = self._parse_date(date)
                if target_dt:
                    # Find closest date within a 45-day window (allows for fiscal year differences)
                    # This handles cases where companies file slightly before or after quarter-end
                    max_days_diff = 45  # ~1.5 months window for quarterly data
                    min_diff = None
                    for date_str in period_data.keys():
                        dt = self._parse_date(date_str)
                        if not dt:
                            continue
                        
                        diff = abs((dt - target_dt).days)
                        # Accept dates within the window
                        if diff <= max_days_diff:
                            if min_diff is None or diff < min_diff:
                                min_diff = diff
                                use_date = date_str
            
            # If no date found within window (or no target date), use most recent
            if not use_date:
                sorted_dates = sorted(period_data.keys(), reverse=True)
                use_date = sorted_dates[0] if sorted_dates else None
        
        if not use_date:
            result[field_name] = None
            if field_name in self._LTM_FINANCIAL_FIELDS and period == 'quarterly':
                result[f"ltm_{field_name}"] = None
            return result, None
        
        result[field_name] = period_data[use_date].get(field_name)
        field_date = use_date
        
        # Calculate LTM for quarterly financial fields
        if field_name in self._LTM_FINANCIAL_FIELDS and period == 'quarterly':
            result[f"ltm_{field_name}"] = self._calculate_ltm(period_data, field_name, use_date)
        
        return result, field_date

    def _extract_earnings_field(self, full_data: dict, field_name: str, date: str = None) -> tuple:
        """
        Extract a field from Earnings::History section with LTM and bf_epsEstimate calculations.
        
        Returns tuple: (result_dict, field_date, report_date)
        result_dict contains: field_name, ltm_{field_name}, and optionally bf_epsEstimate
        field_date is the period date string for this field (or None)
        report_date is the actual earnings release date (reportDate field, or None)
        """
        result = {}
        
        if 'Earnings' not in full_data or 'History' not in full_data['Earnings']:
            result[field_name] = None
            result[f"ltm_{field_name}"] = None
            return result, None, None
        
        history_data = full_data['Earnings']['History']
        use_date = None
        value = None
        
        if date and history_data:
            # Try exact match first
            if date in history_data and history_data[date].get(field_name) is not None:
                use_date = date
                value = history_data[date].get(field_name)
            else:
                # Find closest date with non-null value
                use_date, value = self._find_closest_date(date, history_data, field_name)
        else:
            # Use most recent available date with non-null value
            for d in sorted(history_data.keys(), reverse=True):
                v = history_data[d].get(field_name)
                if v is not None:
                    use_date = d
                    value = v
                    break
        
        result[field_name] = value
        field_date = use_date
        
        # Extract reportDate (actual earnings release date) if available
        report_date = None
        if use_date and history_data and use_date in history_data:
            report_date = history_data[use_date].get('reportDate')
        
        # Calculate LTM
        if use_date and history_data:
            result[f"ltm_{field_name}"] = self._calculate_ltm(history_data, field_name, use_date)
        else:
            result[f"ltm_{field_name}"] = None
        
        # Calculate blended forward EPS estimate
        if field_name == 'epsEstimate' and history_data:
            result['bf_epsEstimate'] = self._calculate_bf_eps_estimate(history_data, use_date)
        
        return result, field_date, report_date

    def _extract_generic_field(self, data: dict, parts: List[str]) -> any:
        """
        Extract a field by traversing a nested dictionary using path parts.
        
        Parameters
        ----------
        data : dict
            The data to traverse
        parts : list of str
            Path parts (excluding the first section part)
        
        Returns
        -------
        any
            The extracted value or None if path doesn't exist
        """
        current = data
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current

    def get_fundamental_fields(self, ticker: str, field_paths: List[str], period: str = 'quarterly', date: str = None) -> Optional[dict]:
        """
        Get multiple fundamental fields from EODHD API in a single request.
        
        Automatically adds standardized date columns:
        - dt_epsActual: Date for epsActual period (if requested)
        - dt_epsReport: Actual earnings release date (if epsActual requested)
        - dt_FYE: Fiscal quarter based on company's fiscal year end
        - dt_CQ_Earnings: Calendar quarter for earnings (dt_epsReport minus 1 quarter)
        - dt_CQ_Income_Statement: Actual date retrieved from API for income statement
        - dt_CQ_Balance_Sheet: Actual date retrieved from API for balance sheet
        - dt_CQ_Cash_Flow: Actual date retrieved from API for cash flow
        
        Parameters
        ----------
        ticker : str
            Stock ticker (e.g., 'AAPL.US')
        field_paths : list of str
            List of field paths (e.g., ['Highlights::MarketCapitalization'])
        period : str
            For Financials fields: 'yearly' or 'quarterly'
        date : str, optional
            Target date for data retrieval
        
        Returns
        -------
        dict or None
            Dictionary with Ticker and all field values
        """
        needs_full_data = any(fp.startswith('Financials') for fp in field_paths)
        sections = set(fp.split('::')[0] for fp in field_paths)
        result = {"Ticker": ticker}
        
        # Track which statement types we have dates for
        statement_dates = {}
        report_dates = {}

        try:
            if needs_full_data:
                # Fetch full data when Financials fields are needed
                url = f"{self.BASE_URL}/fundamentals/{ticker}?api_token={self.api_key}&fmt=json"
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                full_data = response.json()

                for field_path in field_paths:
                    parts = field_path.split('::')
                    section = parts[0]
                    field_name = parts[-1]

                    if section == 'Financials' and len(parts) >= 3:
                        # Extract financial statement fields
                        statement_type = parts[1]
                        field_data, field_date = self._extract_financials_field(full_data, field_name, statement_type, period, date)
                        result.update(field_data)
                        
                        # Track the date for this statement type
                        if field_date is not None:
                            statement_dates[statement_type] = field_date
                    
                    elif section == 'Earnings' and len(parts) >= 3 and parts[1] == 'History':
                        # Extract earnings history fields
                        field_data, field_date, report_date = self._extract_earnings_field(full_data, field_name, date)
                        result.update(field_data)
                        
                        # Track both period date and report date for earnings
                        if field_date is not None:
                            statement_dates['Earnings'] = field_date
                            # Special case: add dt_epsActual if this is epsActual field
                            if field_name == 'epsActual':
                                result['dt_epsActual'] = field_date
                                # Also add dt_epsReport for the actual earnings release date
                                if report_date is not None:
                                    result['dt_epsReport'] = report_date
                                    report_dates['Earnings'] = report_date
                    
                    else:
                        # Extract generic nested fields
                        if section in full_data:
                            result[field_name] = self._extract_generic_field(full_data[section], parts[1:])
                        else:
                            result[field_name] = None
            else:
                # Fetch filtered data for non-Financials fields
                for section in sections:
                    url = f"{self.BASE_URL}/fundamentals/{ticker}?api_token={self.api_key}&fmt=json&filter={section}"
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
                    section_data = response.json()

                    for field_path in field_paths:
                        if field_path.startswith(section):
                            parts = field_path.split('::')
                            field_name = parts[-1]
                            # Check if section_data has the section as a key or is already the section data
                            if section in section_data:
                                result[field_name] = self._extract_generic_field(section_data[section], parts[1:])
                            else:
                                # section_data is already the filtered section
                                result[field_name] = self._extract_generic_field(section_data, parts[1:])

            # Add standardized date columns
            # Get fiscal year end month for this company
            fye_month = self._get_fiscal_year_end(ticker)
            
            # Add dt_FYE based on the most recent date found
            reference_date = None
            for stmt_date in statement_dates.values():
                if stmt_date:
                    reference_date = stmt_date
                    break
            
            if reference_date:
                result['dt_FYE'] = standardize_to_fiscal_quarter(reference_date, fiscal_year_end_month=fye_month)
            else:
                result['dt_FYE'] = None
            
            # Add dt_CQ_Earnings by subtracting 1 quarter from dt_epsReport
            if 'dt_epsReport' in result and result['dt_epsReport']:
                # First convert report date to calendar quarter (simple conversion based on month),
                # then subtract 1 quarter for earnings season lag
                report_quarter = date_to_calendar_quarter(result['dt_epsReport'])
                result['dt_CQ_Earnings'] = subtract_calendar_quarter(report_quarter, 1)
            else:
                result['dt_CQ_Earnings'] = None
                
            # For financial statements, use the actual date retrieved from the API
            # (which may differ from the request date if _find_closest_date was used)
            if 'Income_Statement' in statement_dates:
                result['dt_CQ_Income_Statement'] = statement_dates['Income_Statement']
            else:
                result['dt_CQ_Income_Statement'] = None
                
            if 'Balance_Sheet' in statement_dates:
                result['dt_CQ_Balance_Sheet'] = statement_dates['Balance_Sheet']
            else:
                result['dt_CQ_Balance_Sheet'] = None
                
            if 'Cash_Flow' in statement_dates:
                result['dt_CQ_Cash_Flow'] = statement_dates['Cash_Flow']
            else:
                result['dt_CQ_Cash_Flow'] = None
            
            # Return result if any field has a non-null value
            if any(v is not None for k, v in result.items() if k != "Ticker"):
                return result
        except Exception:
            pass
        return None


class Index:
    """
    Professional class for analyzing market indices and ETFs.
    
    Supports both index constituents (e.g., GSPC.INDX) and ETF holdings (e.g., IVV.US).
    Provides methods for constituent analysis, fundamental data fetching, and allocation metrics.
    
    Parameters
    ----------
    ticker : str
        Index or ETF ticker (e.g., 'GSPC.INDX', 'IVV.US')
    api_key : str, optional
        EODHD API key. If not provided, uses global API_KEY.
        
    Attributes
    ----------
    ticker : str
        Index/ETF ticker symbol
    is_index : bool
        True if ticker is an index (.INDX), False if ETF
    constituents : pd.DataFrame
        DataFrame of constituent holdings
        
    """
    
    BASE_URL = "https://eodhd.com/api"
    RUSSELL_ETFS = ['IWD.US', 'IWM.US', 'IWR.US']
    
    def __init__(self, ticker: str, api_key: Optional[str] = None):
        """Initialize Index analyzer."""
        self.ticker = ticker.upper()
        self.api_key = api_key or API_KEY
        self.is_index = 'INDX' in self.ticker
        self.constituents = None
        self._cache = {}
        self._fundamentals = Fundamentals(api_key=self.api_key)
    
    def get_constituents(self) -> pd.DataFrame:
        """
        Get index constituents or ETF holdings.
        
        Returns
        -------
        pd.DataFrame
            Columns: Ticker, Name, Weight, Sector, Industry, Exchange (Index only)
            Sorted by weight descending
        """
        if self.constituents is not None:
            return self.constituents
        
        url = f'{self.BASE_URL}/fundamentals/{self.ticker}?api_token={self.api_key}&fmt=json'
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        constituents = []
        
        if self.is_index:
            components = data.get("Components", {})
            for ticker, info in components.items():
                constituents.append({
                    "Ticker": info.get("Code"),
                    "Name": info.get("Name"),
                    "Weight": info.get("Weight"),
                    "Sector": info.get("Sector"),
                    "Industry": info.get("Industry"),
                    "Exchange": info.get("Exchange")
                })
        else:
            holdings = data.get("ETF_Data", {}).get("Holdings", {})
            for holding in holdings.values():
                constituents.append({
                    "Ticker": holding.get("Code"),
                    "Name": holding.get("Name"),
                    "Weight": holding.get("Assets_%"),
                    "Sector": holding.get("Sector"),
                    "Industry": holding.get("Industry"),
                    "Exchange": np.nan
                })
        
        self.constituents = pd.DataFrame(constituents)
        return self.constituents.sort_values(by="Weight", ascending=False).reset_index(drop=True)
    
    def _get_fundamental_fields(self, ticker: str, field_paths: List[str], period: str = 'quarterly', date: str = None) -> Optional[dict]:
        """
        Get multiple fundamental fields from EODHD API in a single request.
        
        Delegates to Fundamentals class for data fetching and processing.
        Automatically adds standardized date columns:
        - dt_epsActual, dt_epsReport (if epsActual is requested)
        - dt_FYE, dt_CQ_Earnings, dt_CQ_Income_Statement, dt_CQ_Balance_Sheet, dt_CQ_Cash_Flow
        
        Parameters
        ----------
        ticker : str
            Stock ticker (e.g., 'AAPL' or 'AAPL.US')
        field_paths : list of str
            List of field paths (e.g., ['Highlights::MarketCapitalization'])
        period : str
            For Financials fields: 'yearly' or 'quarterly'
        date : str, optional
            Target date for data retrieval
        
        Returns
        -------
        dict or None
            Dictionary with Ticker and all field values
        """
        # Add '.US' suffix if ticker doesn't have an exchange
        if '.' not in ticker:
            api_ticker = f"{ticker}.US"
        else:
            api_ticker = ticker
        
        result = self._fundamentals.get_fundamental_fields(api_ticker, field_paths, period, date)
        
        # Ensure the returned Ticker matches the input ticker (without exchange suffix)
        if result and result.get('Ticker') != ticker:
            result['Ticker'] = ticker
        
        return result
    
    def fetch_constituent_fundamentals(
        self, 
        field_paths: Union[str, List[str]], 
        period: str = 'quarterly',
        date: Optional[str] = None,
        max_workers: int = 10,
        top_n: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch fundamental fields for all constituents with parallel processing.
        
        Automatically adds standardized date columns:
        - dt_FYE: Fiscal quarter based on each company's fiscal year end
        - dt_CQ_Earnings: Calendar quarter for earnings data
        - dt_CQ_Income_Statement: Calendar quarter for income statement
        - dt_CQ_Balance_Sheet: Calendar quarter for balance sheet
        - dt_CQ_Cash_Flow: Calendar quarter for cash flow statement
        
        Parameters
        ----------
        field_paths : str or list of str
            Field path(s) to fetch (e.g., ['Highlights::MarketCapitalization'])
        period : str
            For Financials fields: 'yearly' or 'quarterly'
        date : str, optional
            Target date for data retrieval
        max_workers : int
            Number of parallel threads
        top_n : int, optional
            If specified, only fetch fundamentals for top N holdings by weight
        
        Returns
        -------
        pd.DataFrame
            Results with Ticker and requested field values
        """
        if self.constituents is None:
            self.get_constituents()
        
        if isinstance(field_paths, str):
            field_paths = [field_paths]
        
        field_names = [fp.split('::')[-1] for fp in field_paths]
        
        # Filter to top N holdings if specified
        if top_n is not None:
            constituent_tickers = self.constituents.nlargest(top_n, 'Weight')['Ticker']
            print(f"\nFetching {len(field_names)} fields for top {top_n} holdings (parallel with {max_workers} threads)...")
        else:
            constituent_tickers = self.constituents['Ticker']
            print(f"\nFetching {len(field_names)} fields for all {len(constituent_tickers)} constituents (parallel with {max_workers} threads)...")
        
        print(f"Fields: {', '.join(field_names)}")
        
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._get_fundamental_fields, ticker, field_paths, period, date): ticker 
                for ticker in constituent_tickers
            }
            
            for future in tqdm(as_completed(futures), total=len(constituent_tickers)):
                result = future.result()
                if result:
                    results.append(result)
        
        df = pd.DataFrame(results)
        return df.merge(self.constituents, on='Ticker', how='left') if not df.empty else df
    
    def get_world_regions(self) -> pd.DataFrame:
        """
        Get ETF world region allocation (ETFs only).
        
        Returns
        -------
        pd.DataFrame
            Columns: Region, Equity_%, Relative_to_Category
        """
        if self.is_index:
            raise ValueError("World regions only available for ETFs, not indices")
        
        url = f"{self.BASE_URL}/fundamentals/{self.ticker}?api_token={self.api_key}&fmt=json"
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        regions = data.get('ETF_Data', {}).get('World_Regions', {})
        
        rows = []
        for region, values in regions.items():
            if isinstance(values, dict):
                rows.append({
                    'Region': region,
                    'Equity_%': float(values.get('Equity_%', 0)),
                    'Relative_to_Category': float(values.get('Relative_to_Category', 0))
                })
            else:
                rows.append({
                    'Region': region, 
                    'Equity_%': float(values) if not isinstance(values, str) else float(values.rstrip('%'))
                })
        
        df = pd.DataFrame(rows)
        return df.sort_values('Equity_%', ascending=False).reset_index(drop=True) if not df.empty else df
    
    def get_sector_weights(self) -> pd.DataFrame:
        """
        Get ETF sector allocation (ETFs only).
        
        Returns
        -------
        pd.DataFrame
            Columns: Sector, Equity_%, Relative_to_Category
        """
        if self.is_index:
            raise ValueError("Sector weights only available for ETFs, not indices")
        
        url = f"{self.BASE_URL}/fundamentals/{self.ticker}?api_token={self.api_key}&fmt=json"
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        sectors = data.get('ETF_Data', {}).get('Sector_Weights', {})
        
        rows = []
        for sector, values in sectors.items():
            if isinstance(values, dict):
                rows.append({
                    'Sector': sector,
                    'Equity_%': float(values.get('Equity_%', 0)),
                    'Relative_to_Category': float(values.get('Relative_to_Category', 0))
                })
            else:
                rows.append({
                    'Sector': sector, 
                    'Equity_%': float(values) if not isinstance(values, str) else float(values.rstrip('%'))
                })
        
        df = pd.DataFrame(rows)
        return df.sort_values('Equity_%', ascending=False).reset_index(drop=True) if not df.empty else df
    
    def get_valuations(self) -> pd.DataFrame:
        """
        Get ETF valuation metrics (ETFs only).
        
        Returns
        -------
        pd.DataFrame
            Columns: Metric, Value
        """
        if self.is_index:
            raise ValueError("Valuations only available for ETFs, not indices")
        
        url = f"{self.BASE_URL}/fundamentals/{self.ticker}?api_token={self.api_key}&fmt=json"
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        valuations = data.get('ETF_Data', {}).get('Valuations_Growth', {}).get('Valuations_Rates_Portfolio', {})
        
        df = pd.DataFrame([
            {'Metric': metric, 'Value': value} 
            for metric, value in valuations.items()
            if not isinstance(value, dict)
        ])
        
        return df
    
    def get_growth_rates(self) -> pd.DataFrame:
        """
        Get ETF growth rate metrics (ETFs only).
        
        Returns
        -------
        pd.DataFrame
            Columns: Metric, Value
        """
        if self.is_index:
            raise ValueError("Growth rates only available for ETFs, not indices")
        
        url = f"{self.BASE_URL}/fundamentals/{self.ticker}?api_token={self.api_key}&fmt=json"
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        growth = data.get('ETF_Data', {}).get('Valuations_Growth', {}).get('Growth_Rates_Portfolio', {})
        
        df = pd.DataFrame([
            {'Metric': metric, 'Value': value} 
            for metric, value in growth.items()
            if not isinstance(value, dict)
        ])
        
        return df
    

  


class Security(TechnicalAnalysis, FundamentalAnalysis):

    def __init__(self, ticker:str, start=None, end=None):
        super().__init__()
        self.ticker = ticker
        self.start = start if start is not None else datetime.today() - relativedelta(months=12)
        self.end = end if end is not None else datetime.today()
        self._fundamentals = Fundamentals(api_key=API_KEY)

        if isinstance(self.ticker, str):
            self.eoddf = self.get_eoddf(ticker=self.ticker, start=self.start, end=self.end).set_index('date').sort_index().rename(columns={'adjusted_close': self.ticker})
        elif isinstance(self.ticker, list):
            self.eoddf = self.get_eoddf_bulk(tickers=self.ticker, start=self.start, end=self.end).pivot(index='date', columns='ticker', values='adjusted_close') # prices only


    def __str__(self):
        return self.ticker


    @property
    def prices(self):
        """Property to get unique expiration dates"""
        if isinstance(self.ticker, str):
            return self.eoddf[self.ticker]    
        else:
            return self.eoddf

    @property
    def last_close(self):
        """Property to get unique expiration dates"""
        return self.prices.iloc[-1]
    
    @property
    def returns(self):
        """Daily log returns: ln(P_t) - ln(P_{t-1}). Drops NA only if entire row is NA."""
        return np.log(self.prices).diff().dropna(how='all') 
 

    def _fetch_prices(self, ticker, start, end):
        url = f'https://eodhd.com/api/eod/{ticker}?from={start}&to={end}&api_token={API_KEY}&fmt=json'
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            df['ticker'] = ticker
            return df
        else:
            print(f"Failed to fetch data for {ticker}. Status code: {response.status_code}")
            return pd.DataFrame()

    def get_eoddf(self, ticker, start=None, end=None):
        start = start if start is not None else self.start
        end = end if end is not None else self.end
        return self._fetch_prices(ticker=ticker, start=start, end=end)

    def get_eoddf_bulk(self, tickers:list,  start=None, end=None):
        start = start if start is not None else self.start
        end = end if end is not None else self.end
        return pd.concat([self._fetch_prices(ticker=t, start=start, end=end) for t in tickers], ignore_index=True)
    
    def fetch_security_fundamentals(
        self,
        field_paths: Union[str, List[str]],
        period: str = 'quarterly',
        date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch fundamental fields for the security (single ticker, no parallel processing).
        
        Automatically adds standardized date columns:
        - dt_epsActual: Date for epsActual period (if requested)
        - dt_epsReport: Actual earnings release date (if epsActual requested)
        - dt_FYE: Fiscal quarter (YYYYQ#) based on company's fiscal year end
        - dt_CQ_Earnings: Calendar quarter for earnings date
        - dt_CQ_Income_Statement: Calendar quarter for income statement date
        - dt_CQ_Balance_Sheet: Calendar quarter for balance sheet date
        - dt_CQ_Cash_Flow: Calendar quarter for cash flow statement date
        
        Parameters
        ----------
        field_paths : str or list of str
            Field path(s) to fetch (e.g., ['Highlights::MarketCapitalization'])
        period : str
            For Financials fields: 'yearly' or 'quarterly'
        date : str, optional
            Target date for data retrieval (format: 'YYYY-MM-DD')
        
        Returns
        -------
        pd.DataFrame
            Single-row DataFrame with Ticker and requested field values
        """
        if isinstance(field_paths, str):
            field_paths = [field_paths]
        
        field_names = [fp.split('::')[-1] for fp in field_paths]
        print(f"\nFetching {len(field_names)} fields for {self.ticker}...")
        print(f"Fields: {', '.join(field_names)}")
        
        result = self._fundamentals.get_fundamental_fields(
            ticker=self.ticker,
            field_paths=field_paths,
            period=period,
            date=date
        )
        
        if result:
            return pd.DataFrame([result])
        else:
            return pd.DataFrame()






class Chain:
    # Aggregate analysis across all strikes and expiries on the chain.

    def __init__(self, ticker, max_expiration=None, refresh=False):
        self.underlying = Security(ticker)
        self.ticker = ticker
        self.refresh = refresh
        
        # Set default max_expiration
        self.max_expiration = max_expiration if max_expiration else (datetime.today() + relativedelta(months=12)).strftime("%Y-%m-%d")
        
        # Create filepath for cached data
        today = datetime.today().strftime("%Y-%m-%d")
        self.fp_chain = os.path.join(r"C:\Users\msands\OneDrive\Documents\data\EODHD Options Chain\\", f'{self.ticker}_Chain_{today}_{self.max_expiration}.pkl')
        
        self.chain = self._load_or_fetch_data()
        self.monthlies = self.chain[self.chain['expiration_type'] == 'monthly']
        self.weeklies = self.chain[self.chain['expiration_type'] == 'weekly']

    @property
    def calls(self):
        """Property to get calls data"""
        return self.chain[(self.chain['type'] == 'call')]

    @property  
    def puts(self):
        """Property to get puts data"""
        return self.chain[(self.chain['type'] == 'put')]

    @property
    def strikes(self):
        """Property to get unique strikes"""
        return self.chain['strike'].unique()

    @property
    def exp_dates(self):
        """Property to get unique expiration dates"""
        return self.chain['exp_date'].unique()

    @property
    def exp_date_nearest(self):
        """Property to get nearest expiration date"""
        exp_dates = self.chain['exp_date'].unique()
        return exp_dates[np.abs(pd.to_datetime(exp_dates) - datetime.today()).argmin()]

    @property
    def exp_date_1M(self):
        """Property to get expiration date closest to 1 month from today"""
        exp_dates = self.chain['exp_date'].unique()
        target_date = datetime.today() + timedelta(days=30)
        return exp_dates[np.abs(pd.to_datetime(exp_dates) - target_date).argmin()]

    @property
    def atm(self):
        """Get ATM (At The Money) contracts for each expiration date and type"""
        def find_atm_for_group(group):
            atm_idx = (group['strike'] - self.underlying.last_close).abs().argmin()
            atm_strike = group['strike'].iloc[atm_idx]
            return group[group['strike'] == atm_strike]
        atm_contracts = self.chain.groupby(['exp_date', 'type']).apply(find_atm_for_group).reset_index(drop=True).drop_duplicates(keep='first')
        return atm_contracts

    @property
    def atm_calls(self):
        """Get ATM (At The Money) call contracts for each expiration date"""
        return self.atm[self.atm['type'] == 'call']

    @property
    def atm_puts(self):
        """Get ATM (At The Money) put contracts for each expiration date"""
        return self.atm[self.atm['type'] == 'put']

    @property
    def delta_twenty_five(self):
        """Get 25 Delta contracts for each expiration date and type"""
        def find_delta_for_group(group):
            # Check if this is a call or put group
            option_type = group['type'].iloc[0]
            
            if option_type == 'call':
                target_delta = 0.25
            else:  # put
                target_delta = -0.25
                
            delta_idx = (group['delta'] - target_delta).abs().argmin()
            return group.iloc[delta_idx]

        return self.chain.groupby(['exp_date', 'type', 'tradetime']).apply(find_delta_for_group).reset_index(drop=True)

    @property
    def thirty_day_maturity(self):
        """
        As of the latest trade date, get all option contracts
        associated with the expiration closest to 30 days away.

        NOTE: consideres both monthly and or weekly for identifying the 30 day. may want to change.
        """
        df = self.chain.copy()
        df['tradetime'] = pd.to_datetime(df['tradetime'])
        df['exp_date'] = pd.to_datetime(df['exp_date'])

        latest_date = df['tradetime'].max()
        target_date = latest_date + pd.Timedelta(days=30)

        maturity_diffs = (df['exp_date'] - target_date).abs()
        closest_exp = df.loc[maturity_diffs.idxmin(), 'exp_date']

        df =  df[df['exp_date'] == closest_exp].reset_index(drop=True)
        return df[~df['tradetime'].isna()]


    def _find_existing_data(self):
        """Find existing pickle files for this ticker from today"""
        data_dir = r"C:\Users\msands\OneDrive\Documents\data\EODHD Options Chain"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
            return None, None
        
        today = datetime.today().strftime("%Y-%m-%d")
        # Pattern: {ticker}_Chain_{today}_{max_exp}.pkl
        ticker_today_pattern = f"{self.ticker}_Chain_{today}_"
        
        matching_files = []
        for filename in os.listdir(data_dir):
            if filename.startswith(ticker_today_pattern) and filename.endswith('.pkl'):
                # Extract max_expiration from filename
                try:
                    max_exp_part = filename.replace(ticker_today_pattern, '').replace('.pkl', '')
                    # Validate it's a date
                    pd.to_datetime(max_exp_part)
                    matching_files.append((filename, max_exp_part))
                except:
                    continue
        
        if not matching_files:
            return None, None
        
        # Sort by max_expiration and get the one that covers our range
        matching_files.sort(key=lambda x: pd.to_datetime(x[1]), reverse=True)
        
        # Find a file that covers our requested max_expiration
        for filename, file_max_exp in matching_files:
            if pd.to_datetime(file_max_exp) >= pd.to_datetime(self.max_expiration):
                file_path = os.path.join(data_dir, filename)
                print(f"Found existing data: {filename}")
                
                try:
                    existing_data = pd.read_pickle(file_path)
                    return existing_data, file_max_exp
                except Exception as e:
                    print(f"Error loading existing file {filename}: {e}")
                    continue
        
        return None, None

    def _load_or_fetch_data(self):
        """Smart loading: use existing data if available and current, otherwise fetch new"""
        
        # If refresh=True, always fetch new data
        if self.refresh:
            print(f"Refresh=True: Fetching new data for {self.ticker}")
            return self.fetch_chain()
        
        # Check if exact file exists
        if os.path.exists(self.fp_chain):
            print(f"Loading existing data from {self.fp_chain}")
            return pd.read_pickle(self.fp_chain)
        else:
            # Look for existing files from today that cover our expiration range
            existing_data, existing_max_exp = self._find_existing_data()
            
            if existing_data is not None:
                # Filter to our requested max_expiration range
                existing_data['exp_date'] = pd.to_datetime(existing_data['exp_date'])
                filtered_data = existing_data[existing_data['exp_date'] <= pd.to_datetime(self.max_expiration)]
                
                if not filtered_data.empty:
                    print(f"Using existing data filtered to {self.max_expiration}")
                    # Save filtered data to our specific filepath for future use
                    filtered_data.to_pickle(self.fp_chain)
                    return filtered_data
                else:
                    print(f"Existing data doesn't cover requested range, fetching new data")
                    return self.fetch_chain()
            else:
                print(f"No existing data found for today. Fetching new data for {self.ticker}")
                return self.fetch_chain()

    def fetch_chain(self):
        """Fetch chain data from API"""
        print(f'Fetching Chain for {self.ticker}...')
        
        url = "https://eodhd.com/api/mp/unicornbay/options/contracts"
        params = {
            "filter[underlying_symbol]": self.ticker,
            "filter[exp_date_from]": datetime.today().strftime("%Y-%m-%d"),
            "filter[exp_date_to]": self.max_expiration,
            "sort": "-exp_date",
        }
        df = pd.DataFrame(EODHD().paginate(url=url, params_base=params, offset=0, total=None))
        
        # Save to pickle for future use
        if not df.empty:
            os.makedirs(os.path.dirname(self.fp_chain), exist_ok=True)
            df.to_pickle(self.fp_chain)
            print(f"Chain data saved to {self.fp_chain}")
        
        return df

    def get_chain(self, max_expiration=None):
        """Deprecated: Use fetch_chain() or rely on automatic loading in __init__"""
        print("Warning: get_chain() is deprecated. Chain data is loaded automatically in __init__")
        if max_expiration and max_expiration != self.max_expiration:
            print(f"Note: Requested max_expiration {max_expiration} differs from initialized {self.max_expiration}")
        return self.chain


    def get_atm_avg(self):
        """Get average ATM volatility for calls and puts"""
        cols = ['exp_date', 'volatility']
        c = self.atm[self.atm['type']=='call'][cols]
        p = self.atm[self.atm['type']=='put'][cols]
        res = c.merge(p, on='exp_date', suffixes=('_call', '_put'))
        res['volatility_avg'] = (res['volatility_call'] + res['volatility_put']) / 2
        return res


    # def get_strike_by_expiry(self,  type='call', values='volatility'):
    #     df = self.chain
    #     if type == 'call':
    #         df = df[(df['type']==type) & (df['delta'] >= 0.25)  & (df['delta'] <= 0.75)].pivot(index='strike', columns='exp_date', values=values)
    #     elif type == 'put':
    #         df = df[(df['type']==type) & (df['delta'] <= -0.25)  & (df['delta'] >= -0.75)].pivot(index='strike', columns='exp_date', values=values)
    #     return df

    def get_implied_move(self): # NOTE: move to options submodule class as an eval method
        df = self.atm.pivot(index=['exp_date','dte'], columns='type', values='volatility').reset_index()
        df['ivol'] = (df['call'] + df['put']) /2
        df['undr_last_price'] = self.underlying.last_close
        df['expected_move_usd'] = (df['ivol']) * df['undr_last_price'] * np.sqrt(df['dte'] / 365)
        df['expected_move_pct'] = df['expected_move_usd'] / df['undr_last_price']
        return df    
    



class ChainHistory:
    """ 
    Historical trade analysis across the options chain.
    https://eodhd.com/marketplace/unicornbay/options/docs
    
    To determine the exact trade time, look for records where volume > 0 and last price is set. 
    Otherwise, the tradetime reflects the last recorded market event rather than an actual transaction.
    If no trade happened: tradetime may correspond to the last update of bid/ask prices, open interest changes, or other market events.
    In such cases, volume is zero, and the last price may be missing or zero.
    """
    def __init__(self, ticker, start=None, end=None, chunk_size=15, refresh=False):
        self.ticker = ticker
        self.underlying = Security(ticker)
        
        # Set default date ranges
        self.start = start if start else f"{datetime.now().year}-01-01"
        self.end = end if end else datetime.now().strftime("%Y-%m-%d")
        self.refresh = refresh
        self.fp_chain_history = os.path.join(r"C:\Users\msands\OneDrive\Documents\data\EODHD Options Chain History\\", f'{self.ticker}_Chain_History_{self.start}_{self.end}.pkl')
        
        self.history = self._load_or_fetch_data(chunk_size=chunk_size)
        self.trades = self.history[ (self.history['volume']>0) & (self.history['last']!=0) ]

    def __str__(self):
        return f"{self.ticker}: {self.start} to {self.end}"

    def _find_existing_partial_data(self):
        """Find existing pickle files for this ticker with same start date"""
        data_dir = r"C:\Users\msands\OneDrive\Documents\data\EODHD Options Chain History"
        if not os.path.exists(data_dir):
            return None, None
        
        # Pattern: {ticker}_Chain_History_{start}_{end}.pkl
        ticker_start_pattern = f"{self.ticker}_Chain_History_{self.start}_"
        
        matching_files = []
        for filename in os.listdir(data_dir):
            if filename.startswith(ticker_start_pattern) and filename.endswith('.pkl'):
                # Extract end date from filename
                try:
                    # Example: IVV_Chain_History_2025-08-01_2025-08-15.pkl
                    end_date_part = filename.replace(ticker_start_pattern, '').replace('.pkl', '')
                    # Validate it's a date
                    pd.to_datetime(end_date_part)
                    matching_files.append((filename, end_date_part))
                except:
                    continue  # Skip invalid filenames
        
        if not matching_files:
            return None, None
        
        # Sort by end date and get the latest one
        matching_files.sort(key=lambda x: pd.to_datetime(x[1]), reverse=True)
        latest_file, latest_end_date = matching_files[0]
        
        file_path = os.path.join(data_dir, latest_file)
        print(f"Found existing partial data: {latest_file}")
        
        try:
            existing_data = pd.read_pickle(file_path)
            return existing_data, latest_end_date
        except Exception as e:
            print(f"Error loading existing file {latest_file}: {e}")
            return None, None


    def _load_or_fetch_data(self, chunk_size):
        """Smart loading: use existing data if available, fetch missing ranges"""
        
        # If refresh=True, always fetch new data
        if self.refresh:
            print(f"Refresh=True: Fetching new data for {self.ticker}")
            return self.fetch_data(chunk_size=chunk_size)

        # Check if exact file exists
        if os.path.exists(self.fp_chain_history):
            print(f"Loading existing data from {self.fp_chain_history}")
            return pd.read_pickle(self.fp_chain_history)
        else:
            # Look for existing files with same ticker and start date
            existing_data, existing_end_date = self._find_existing_partial_data()
            
            if existing_data is not None:
                print(f"Found existing data ending at {existing_end_date}")
                
                # Check if we need additional data
                if existing_end_date >= self.end:
                    print(f"Existing data covers requested range. Using existing data.")
                    return existing_data
                else:
                    # Fetch only the missing data from where existing data ends
                    print(f"Fetching additional data from {existing_end_date} to {self.end}")
                    
                    # Temporarily adjust start date to continue from where existing data ends
                    original_start = self.start
                    self.start = (pd.to_datetime(existing_end_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                    
                    # Fetch missing data
                    new_data = self.fetch_data(chunk_size)
                    
                    # Restore original start date
                    self.start = original_start
                    
                    if not new_data.empty:
                        # Combine existing and new data
                        combined_data = pd.concat([existing_data, new_data], ignore_index=True)
                        combined_data = combined_data.drop_duplicates()
                        
                        # Save combined dataset with correct filename
                        combined_data.to_pickle(self.fp_chain_history)
                        print(f"Combined data saved to {self.fp_chain_history}")
                        return combined_data
                    else:
                        print("No new data fetched, returning existing data")
                        return existing_data
            else:
                print(f"No existing data found. Fetching new data for {self.ticker} from {self.start} to {self.end}")
                return self.fetch_data(chunk_size)


    def fetch_data(self, chunk_size=None):
        
        start_dt = pd.to_datetime(self.start)
        end_dt = pd.to_datetime(self.end)
            
        all_data = []
        current_start = start_dt
        
        while current_start < end_dt:
            chunk_end = min(current_start + relativedelta(days=chunk_size), end_dt) # Calculate chunk end date (15 days from current start or end_dt, whichever is earlier)
            
            print(f"Fetching history from {current_start.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}")
            
            url = "https://eodhd.com/api/mp/unicornbay/options/eod"
            params = {
                "filter[underlying_symbol]": f"{self.ticker}.US",
                "filter[tradetime_from]": current_start.strftime("%Y-%m-%d"),
                "filter[tradetime_to]": chunk_end.strftime("%Y-%m-%d"),                
                # "filter[exp_date_from]": current_start.strftime("%Y-%m-%d"),
                # "filter[exp_date_to]": chunk_end.strftime("%Y-%m-%d"),
                # "filter[tradetime_from]": "2000-01-01"
            }
            
            chunk_data = EODHD().paginate(url=url, params_base=params, offset=0, total=None)
            if chunk_data:  # Only add if we got data
                all_data.extend(chunk_data)
            
            current_start = chunk_end + relativedelta(days=1)
        
        df = pd.DataFrame(all_data).drop_duplicates(keep='first')
        if not df.empty:
            df.to_pickle(self.fp_chain_history)       

        print(f"Data fetch complete. Total records fetched: {len(df)} @ {self.fp_chain_history}")
        return df



    def get_timeseries_by_delta(self, delta_target = 0.5, delta_tolerance=0.25, trades=False):
        """
        Returns a time series of the at-the-money (ATM) options contract
        with delta closest to +0.50 for calls and -0.50 for puts,
        only if delta is within ±delta_tolerance (default 0.25).
        Skips dates with no valid contracts.
        """
        if trades:
            df = self.trades.copy()
        else:
            df = self.history.copy()
        df['tradetime'] = pd.to_datetime(df['tradetime'])

        delta_targets = {'call': delta_target, 'put': -delta_target} # 0.5, -0.5 calls, puts for atm

        def get_closest_delta(group):
            results = []
            for opt_type, target in delta_targets.items():
                subset = group[group['type'] == opt_type]
                if subset.empty:
                    continue
                # Compute delta diff
                subset['delta_diff'] = (subset['delta'] - target).abs()
                closest = subset.loc[subset['delta_diff'].idxmin()]
                if closest['delta_diff'] <= delta_tolerance:
                    results.append(closest.drop(labels='delta_diff'))
            if results:
                return pd.DataFrame(results)
            else:
                return pd.DataFrame()

        # Apply to each day (tradetime)
        atm_df = df.groupby('tradetime').apply(get_closest_delta).reset_index(drop=True)

        return atm_df



    def get_n_dte_timeseries(self, n_days: int = 30, trades=False):
        if trades:
            df = self.trades.copy()
        else:
            df = self.history.copy()
        df['tradetime'] = pd.to_datetime(df['tradetime'])
        df['exp_date'] = pd.to_datetime(df['exp_date'])

        results = []

        for trade_date, group in df.groupby('tradetime'):
            target_date = trade_date + pd.Timedelta(days=n_days)
            maturity_diffs = (group['exp_date'] - target_date).abs()
            closest_exp = group.loc[maturity_diffs.idxmin(), 'exp_date']
            selected = group[group['exp_date'] == closest_exp].copy()
            results.append(selected)

        out = pd.concat(results, ignore_index=True)

        return out[~out['tradetime'].isna()]


    def get_time_series_n_dte_by_delta(self, n_days: int = 30, delta_target=0.5, delta_tolerance: float = 0.25, trades=False):
        """
        Combine 30-day-to-expiry selection with ATM-by-delta selection.

        For each tradetime:
          1. Find the expiration date closest to tradetime + n_days.
          2. From contracts with that expiration, pick the ATM contract by delta:
             - calls: delta closest to +0.50
             - puts:  delta closest to -0.50
          3. Only include a contract if its delta distance to the target is <= delta_tolerance.

        Returns:
            DataFrame with selected ATM contracts (calls and puts) for the n-day-to-expiry timeseries.
        """
        if trades:
            df = self.trades.copy()
        else:
            df = self.history.copy()

        df['tradetime'] = pd.to_datetime(df['tradetime'])
        df['exp_date'] = pd.to_datetime(df['exp_date'])

        delta_targets = {'call': delta_target, 'put': -delta_target}
        results = []

        for trade_date, group in df.groupby('tradetime'):
            # determine target expiration (closest to trade_date + n_days)
            target_date = pd.to_datetime(trade_date) + pd.Timedelta(days=n_days)
            # if group has no exp_date values skip
            if group['exp_date'].isna().all():
                continue
            # compute abs difference to target expiration and pick the closest expiration date
            maturity_diffs = (group['exp_date'] - target_date).abs()
            try:
                closest_exp_idx = maturity_diffs.idxmin()
            except Exception:
                continue
            closest_exp = group.loc[closest_exp_idx, 'exp_date']
            # restrict to rows with that expiration
            selected = group[group['exp_date'] == closest_exp].copy()
            if selected.empty:
                continue

            # For each option type pick the contract with delta closest to the target
            chosen = []
            for opt_type, target in delta_targets.items():
                subset = selected[selected['type'] == opt_type].copy()
                # drop rows with missing delta
                subset = subset[subset['delta'].notna()]
                if subset.empty:
                    continue
                # compute absolute delta distance without mutating original frame
                delta_diff = (subset['delta'] - target).abs()
                try:
                    idx = delta_diff.idxmin()
                except Exception:
                    continue
                row = subset.loc[idx]
                if delta_diff.loc[idx] <= delta_tolerance:
                    chosen.append(row)

            if chosen:
                results.append(pd.DataFrame(chosen))

        if not results:
            return pd.DataFrame()

        out = pd.concat(results, ignore_index=True)
        # ensure tradetime/exp_date are datetimes and sort
        out['tradetime'] = pd.to_datetime(out['tradetime'])
        out['exp_date'] = pd.to_datetime(out['exp_date'])
        out = out.sort_values('tradetime').reset_index(drop=True)
        return out


    def get_pc_spread(self, df, value='volatility'):
        """
        Calculate put-call volatility spread by tradetime
        Returns: DataFrame with tradetime, call_vol, put_vol, and pc_spread (put - call)
        """
        # Pivot to get call and put volatilities side by side
        pc_spread = df.pivot_table(
            index='tradetime', 
            columns='type', 
            values=value,
            aggfunc='first'  # Use first in case of duplicates
        ).reset_index().ffill() # NOTE: ffill may not be a good idea for days where no trades occured.
        
        pc_spread['pc_spread'] = pc_spread['put'] - pc_spread['call']
        pc_spread['pc_ratio'] = pc_spread['put'] / pc_spread['call']

        if value == 'volatility':
            pc_spread['ivol'] = (pc_spread['put'] + pc_spread['call']) /2
        
        # Rename columns for clarity
        pc_spread = pc_spread.rename(columns={
            'call': 'call_vol',
            'put': 'put_vol'
        })
        
        return pc_spread.sort_values('tradetime').reset_index(drop=True)


    def get_pc_spread_rank(self, df):
        """
        Calculates full-period Put-Call Spread Rank, Percentile, and Z-Score.
        
        Args:
            df (pd.DataFrame): Output from get_pc_spread() method with 'tradetime', 'pc_spread' columns.
        
        Returns:
            pd.DataFrame: DataFrame with full-period percentile metrics for PC spread
        """
        pc_df = df.copy().sort_values('tradetime').reset_index(drop=True)
        
        # Calculate full-period metrics for PC spread
        spreads = pc_df['pc_spread']
        pc_df['pc_spread_rank'] = (spreads - spreads.min()) / (spreads.max() - spreads.min())
        pc_df['pc_spread_percentile'] = spreads.rank(pct=True)
        pc_df['pc_spread_zscore'] = (spreads - spreads.mean()) / spreads.std()

        return pc_df

    def get_ivol_rank(self, df):
        pc_df = df.copy().sort_values('tradetime').reset_index(drop=True)
        ivols = pc_df['ivol']
        pc_df['ivol_rank'] = (ivols - ivols.min()) / (ivols.max() - ivols.min()) # where ivol ranks sits within its history (own range of min/max)
        pc_df['ivol_percentile'] = ivols.rank(pct=True) # percent of time the value was less than it is currently
        pc_df['ivol_zscore'] = (ivols - ivols.mean()) / ivols.std()
        return pc_df

    def get_pc_spread_rolling(self, df, window: int = 20):
        """
        Calculates rolling-window Put-Call Spread Rank, Percentile, and Z-Score.
        
        Args:
            df (pd.DataFrame): Output from get_pc_spread() method with 'tradetime', 'pc_spread' columns.
            window (int): Rolling window size for calculations.
        
        Returns:
            pd.DataFrame: DataFrame with rolling metrics for PC spread
        """
        pc_df = df.copy().sort_values('tradetime').reset_index(drop=True)
        
        spreads = pc_df['pc_spread']
        
        # Calculate rolling metrics for PC spread
        pc_df['pc_spread_rank_roll'] = spreads.rolling(window).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0,
            raw=False
        )
        
        pc_df['pc_spread_percentile_roll'] = spreads.rolling(window).apply(
            lambda x: x.rank(pct=True).iloc[-1] if len(x) == window else None,
            raw=False
        )
        
        pc_df['pc_spread_zscore_roll'] = (spreads - spreads.rolling(window).mean()) / spreads.rolling(window).std()
        
        return pc_df



    def get_iv_rank(self, df):
        """
        Calculates full-period IV Rank, Percentile, and Z-Score for  options.
        Also computes volume-weighted average IV percentile as of the latest date.
        
        Args:
            df (pd.DataFrame): Must contain 'tradetime', 'volatility', 'volume', and 'type' columns.
        
        Returns:
            Tuple:
                - full_period_df: DataFrame with full-period percentile metrics
                - volume_weighted_iv_percentile_latest: float
        """
        def compute_full_metrics(group):
            ivs = group['volatility']
            group['iv_rank_full'] = (ivs - ivs.min()) / (ivs.max() - ivs.min())
            group['iv_percentile_full'] = ivs.rank(pct=True)
            group['iv_zscore_full'] = (ivs - ivs.mean()) / ivs.std()
            return group
        
        # Compute full-period metrics
        full_df = df.groupby('type', group_keys=False).apply(compute_full_metrics)
        
        # --- Volume-Weighted IV Percentile (latest date) ---
        latest_date = full_df['tradetime'].max()
        latest_df = full_df[full_df['tradetime'] == latest_date]
        
        if not latest_df.empty and 'iv_percentile_full' in latest_df.columns and 'volume' in latest_df.columns:
            total_volume = latest_df['volume'].sum()
            if total_volume > 0:
                vw_iv_percentile = (latest_df['iv_percentile_full'] * latest_df['volume']).sum() / total_volume
            else:
                vw_iv_percentile = None
        else:
            vw_iv_percentile = None
        
        return full_df, vw_iv_percentile


    def get_iv_rank_rolling(self, df, window: int = 20):
        """
        Calculates rolling-window IV Rank, Percentile, and Z-Score for  options.
        
        Args:
            df (pd.DataFrame): Must contain 'tradetime', 'volatility', 'volume', and 'type' columns.
            window (int): Rolling window size for calculations.
        
        Returns:
            pd.DataFrame: DataFrame with rolling metrics
        """
        def compute_rolling_metrics(group):
            ivs = group['volatility']
            group['iv_rank_roll'] = ivs.rolling(window).apply(
                lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0,
                raw=False
            )
            group['iv_percentile_roll'] = ivs.rolling(window).apply(
                lambda x: x.rank(pct=True).iloc[-1] if len(x) == window else None,
                raw=False
            )
            group['iv_zscore_roll'] = (ivs - ivs.rolling(window).mean()) / ivs.rolling(window).std()
            return group
        
        # Compute rolling metrics
        roll_df = df.groupby('type', group_keys=False).apply(compute_rolling_metrics)
        
        return roll_df
    

    def get_implied_move_timeseries(self):
        def group_ffill_by_exp(df, group_col='exp_date', time_col='tradetime', cols=None):
            df = df.copy()
            df[time_col] = pd.to_datetime(df[time_col])
            df[group_col] = pd.to_datetime(df[group_col])
            df = df.sort_values([group_col, time_col])
            if cols is None:
                # default: numeric columns except the group/time columns
                cols = [c for c in df.select_dtypes(include='number').columns
                        if c not in {group_col, time_col}]
            df[cols] = df.groupby(group_col)[cols].ffill()
            return df

        df = self.get_timeseries_by_delta(delta_target=0.5, delta_tolerance=0.1, trades=True).pivot(index=['tradetime','exp_date','dte'], columns='type', values='volatility').reset_index()
        df = group_ffill_by_exp(df, group_col='exp_date', time_col='tradetime', cols=['dte','call','put'])  
        df['ivol'] = (df['call'] + df['put']) /2
        df['undr_last_price'] = self.underlying.last_close
        df['expected_move_usd'] = (df['ivol']) * df['undr_last_price'] * np.sqrt(df['dte'] / 365)
        df['expected_move_pct'] = df['expected_move_usd'] / df['undr_last_price']
        return df

    def get_implied_move_latest(self):
        df = self.get_implied_move_timeseries()
        mask = df['exp_date'] > datetime.today()
        idx = df.loc[mask].groupby('exp_date')['tradetime'].idxmax()
        latest_by_exp = df.loc[idx].sort_values('exp_date').reset_index(drop=True)
        return latest_by_exp




class News:
    """
    Retrieve financial news from EODHD API.
    
    EODHD News API provides:
    - Latest financial news by ticker
    - News by date range
    - News tags and categories
    - Sentiment analysis (if available)
    """
    
    def __init__(self):
        """Initialize News client using global API_KEY."""
        self.api_key = API_KEY
        self.base_url = "https://eodhd.com/api/news"
    
    def get_ticker_news(self, ticker, limit=50, offset=0, from_date=None, to_date=None):
        """
        Get news for a specific ticker.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol (e.g., 'AAPL.US', 'SPY')
        limit : int
            Number of news items to retrieve (max 1000)
        offset : int
            Offset for pagination
        from_date : str
            Start date in YYYY-MM-DD format
        to_date : str
            End date in YYYY-MM-DD format
            
        Returns:
        --------
        pandas.DataFrame
            News articles with date, title, content, link, symbols, tags, sentiment
        """
        params = {
            'api_token': self.api_key,
            's': ticker,
            'limit': limit,
            'offset': offset,
            'fmt': 'json'
        }
        
        if from_date:
            params['from'] = from_date
        if to_date:
            params['to'] = to_date
        
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # Convert date to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date', ascending=False).reset_index(drop=True)
        
        return df
    
    def get_latest_news(self, limit=50, offset=0, from_date=None, to_date=None):
        """
        Get latest financial news (all markets).
        
        Parameters:
        -----------
        limit : int
            Number of news items to retrieve (max 1000)
        offset : int
            Offset for pagination
        from_date : str
            Start date in YYYY-MM-DD format
        to_date : str
            End date in YYYY-MM-DD format
            
        Returns:
        --------
        pandas.DataFrame
            News articles with date, title, content, link, symbols, tags
        """
        params = {
            'api_token': self.api_key,
            'limit': limit,
            'offset': offset,
            'fmt': 'json'
        }
        
        if from_date:
            params['from'] = from_date
        if to_date:
            params['to'] = to_date
        
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # Convert date to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date', ascending=False).reset_index(drop=True)
        
        return df
    
    def get_multiple_tickers_news(self, tickers, limit=50, from_date=None, to_date=None):
        """
        Get news for multiple tickers.
        
        Parameters:
        -----------
        tickers : list[str]
            List of ticker symbols
        limit : int
            Number of news items per ticker
        from_date : str
            Start date in YYYY-MM-DD format
        to_date : str
            End date in YYYY-MM-DD format
            
        Returns:
        --------
        pandas.DataFrame
            Combined news articles from all tickers
        """
        all_news = []
        
        for ticker in tickers:
            try:
                df = self.get_ticker_news(
                    ticker=ticker,
                    limit=limit,
                    from_date=from_date,
                    to_date=to_date
                )
                if not df.empty:
                    df['primary_ticker'] = ticker
                    all_news.append(df)
            except Exception as e:
                print(f"Error fetching news for {ticker}: {e}")
                continue
        
        if not all_news:
            return pd.DataFrame()
        
        combined = pd.concat(all_news, ignore_index=True)
        
        # Remove duplicates based on title
        combined = combined.drop_duplicates(subset=['title'], keep='first')
        
        # Sort by date
        if 'date' in combined.columns:
            combined = combined.sort_values('date', ascending=False).reset_index(drop=True)
        
        return combined
    
    def search_news(self, keywords, limit=50, from_date=None, to_date=None):
        """
        Search news by keywords (filters results client-side).
        
        Parameters:
        -----------
        keywords : str or list[str]
            Keyword(s) to search for in title and content
        limit : int
            Number of initial news items to fetch and filter
        from_date : str
            Start date in YYYY-MM-DD format
        to_date : str
            End date in YYYY-MM-DD format
            
        Returns:
        --------
        pandas.DataFrame
            Filtered news articles containing keywords
        """
        # Get latest news
        df = self.get_latest_news(limit=limit, from_date=from_date, to_date=to_date)
        
        if df.empty:
            return df
        
        # Convert keywords to list
        if isinstance(keywords, str):
            keywords = [keywords]
        
        # Filter by keywords (case-insensitive)
        mask = pd.Series(False, index=df.index)
        for keyword in keywords:
            if 'title' in df.columns:
                mask |= df['title'].str.contains(keyword, case=False, na=False)
            if 'content' in df.columns:
                mask |= df['content'].str.contains(keyword, case=False, na=False)
        
        return df[mask].reset_index(drop=True)



class MarketCalendar:
    """
    Retrieve financial calendar data from EODHD API.
    
    EODHD Calendar API provides:
    - Earnings announcements
    - IPO calendar
    - Stock splits
    - Economic events
    - Dividend dates
    """
    
    def __init__(self):
        """Initialize MarketCalendar client using global API_KEY."""
        self.api_key = API_KEY
        self.base_url = "https://eodhd.com/api/calendar"
    
    def get_earnings(self, from_date=None, to_date=None, symbols=None):
        """
        Get earnings announcements calendar.
        
        Parameters:
        -----------
        from_date : str
            Start date in YYYY-MM-DD format
        to_date : str
            End date in YYYY-MM-DD format
        symbols : str or list
            Ticker symbol(s) to filter (e.g., 'AAPL.US' or ['AAPL.US', 'MSFT.US'])
            
        Returns:
        --------
        pandas.DataFrame
            Earnings calendar with date, ticker, estimate, actual values
        """
        url = f"{self.base_url}/earnings"
        params = {
            'api_token': self.api_key,
            'fmt': 'json'
        }
        
        if from_date:
            params['from'] = from_date
        if to_date:
            params['to'] = to_date
        if symbols:
            if isinstance(symbols, list):
                params['symbols'] = ','.join(symbols)
            else:
                params['symbols'] = symbols
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if not data or not data.get('earnings'):
            return pd.DataFrame()
        
        df = pd.DataFrame(data['earnings'])
        
        # Convert date to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date', ascending=True).reset_index(drop=True)
        
        return df
    
    def get_ipos(self, from_date=None, to_date=None):
        """
        Get upcoming IPO calendar.
        
        Parameters:
        -----------
        from_date : str
            Start date in YYYY-MM-DD format
        to_date : str
            End date in YYYY-MM-DD format
            
        Returns:
        --------
        pandas.DataFrame
            IPO calendar with date, ticker, name, exchange, price range
        """
        url = f"{self.base_url}/ipos"
        params = {
            'api_token': self.api_key,
            'fmt': 'json'
        }
        
        if from_date:
            params['from'] = from_date
        if to_date:
            params['to'] = to_date
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if not data or not data.get('ipos'):
            return pd.DataFrame()
        
        df = pd.DataFrame(data['ipos'])
        
        # Convert date to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date', ascending=True).reset_index(drop=True)
        
        return df
    
    def get_splits(self, from_date=None, to_date=None, symbols=None):
        """
        Get stock splits calendar.
        
        Parameters:
        -----------
        from_date : str
            Start date in YYYY-MM-DD format
        to_date : str
            End date in YYYY-MM-DD format
        symbols : str or list
            Ticker symbol(s) to filter
            
        Returns:
        --------
        pandas.DataFrame
            Stock splits with date, ticker, split ratio
        """
        url = f"{self.base_url}/splits"
        params = {
            'api_token': self.api_key,
            'fmt': 'json'
        }
        
        if from_date:
            params['from'] = from_date
        if to_date:
            params['to'] = to_date
        if symbols:
            if isinstance(symbols, list):
                params['symbols'] = ','.join(symbols)
            else:
                params['symbols'] = symbols
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if not data or not data.get('splits'):
            return pd.DataFrame()
        
        df = pd.DataFrame(data['splits'])
        
        # Convert date to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date', ascending=True).reset_index(drop=True)
        
        return df
    
    def get_economic_events(self, from_date=None, to_date=None, country=None, limit=1000, offset=0):
        """
        Get economic calendar events (GDP, CPI, unemployment, etc.).
        
        Parameters:
        -----------
        from_date : str
            Start date in YYYY-MM-DD format
        to_date : str
            End date in YYYY-MM-DD format
        country : str
            Country code to filter (e.g., 'US', 'GB', 'JP')
        limit : int
            Number of events to retrieve (default 1000, max 1000)
        offset : int
            Offset for pagination (default 0)
            
        Returns:
        --------
        pandas.DataFrame
            Economic events with date, country, event, actual, forecast, previous values
        """
        url = "https://eodhd.com/api/economic-events"
        params = {
            'api_token': self.api_key,
            'fmt': 'json',
            'limit': limit,
            'offset': offset
        }
        
        if from_date:
            params['from'] = from_date
        if to_date:
            params['to'] = to_date
        if country:
            params['country'] = country
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # Convert date to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date', ascending=True).reset_index(drop=True)
        
        return df
    
    def get_trends(self, from_date=None, to_date=None, symbols=None):
        """
        Get trends (market-moving events and sentiment) and concensus estimates.
        
        Parameters:
        -----------
        from_date : str
            Start date in YYYY-MM-DD format
        to_date : str
            End date in YYYY-MM-DD format
        symbols : str or list
            Ticker symbol(s) to filter
            
        Returns:
        --------
        pandas.DataFrame
            Market trends and events, normalized from nested structure
        """
        url = f"{self.base_url}/trends"
        params = {
            'api_token': self.api_key,
            'fmt': 'json'
        }
        
        if from_date:
            params['from'] = from_date
        if to_date:
            params['to'] = to_date
        if symbols:
            symbols = [f'{s}.US' for s in symbols]
            if isinstance(symbols, list):
                params['symbols'] = ','.join(symbols)
            else:
                params['symbols'] = symbols
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if not data or not data.get('trends'):
            return pd.DataFrame()
        
        # The API returns nested dictionaries - we need to flatten them
        trends_list = []
        for trend in data['trends']:
            if isinstance(trend, dict):
                trends_list.append(trend)
            elif isinstance(trend, list):
                # If it's a list of dicts, extend
                trends_list.extend(trend)
        
        if not trends_list:
            return pd.DataFrame()
        
        df = pd.DataFrame(trends_list)
        
        # Convert date to datetime if present
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date', ascending=True).reset_index(drop=True)
        
        return df
