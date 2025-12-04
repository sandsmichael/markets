"""
Professional Calendar Utilities for Financial Analysis
Comprehensive date handling for trading days, quarters, month-ends, and business calendars
"""

from typing import Union, List, Optional, Tuple
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
import dateutil.rrule as rrule
import pandas as pd
import numpy as np
import pandas_market_calendars as mcal
import warnings


class Calendar:
    """
    Professional calendar utilities for financial and trading applications.
    
    Provides methods for:
    - Trading day calculations (NYSE calendar)
    - Quarter and month-end dates
    - Business day arithmetic
    - Date range generation
    - Fiscal period handling
    
    Parameters
    ----------
    exchange : str
        Market calendar to use (default: 'NYSE')
        Options: 'NYSE', 'NASDAQ', 'LSE', 'TSX', etc.
    
    Examples
    --------
    >>> cal = Calendar()
    >>> today = cal.today()
    >>> quarter_end = cal.previous_quarter_end()
    >>> trading_days = cal.get_trading_days('2024-01-01', '2024-12-31')
    """
    
    QUARTER_END_MONTHS = (3, 6, 9, 12)
    CALENDAR_CACHE_START = '1990-01-01'
    CALENDAR_CACHE_END = '2100-12-31'
    
    def __init__(self, exchange: str = 'NYSE'):
        self.exchange = exchange
        self._market_calendar = mcal.get_calendar(exchange)
        self._trading_days_cache = None
    
    # =============================================================================
    # CURRENT DATE UTILITIES
    # =============================================================================
    
    @staticmethod
    def today() -> datetime:
        """Get current date and time."""
        return datetime.today()
    
    @staticmethod
    def today_date() -> date:
        """Get current date (no time component)."""
        return date.today()
    
    @staticmethod
    def now() -> datetime:
        """Get current datetime (alias for today)."""
        return datetime.now()
    
    def current_year(self, dt: Optional[datetime] = None) -> int:
        """
        Get year from datetime object.
        
        Parameters
        ----------
        dt : datetime, optional
            Date object (default: today)
            
        Returns
        -------
        int
            Year
        """
        return (dt or self.today()).year
    
    def current_month(self, dt: Optional[datetime] = None) -> int:
        """Get month from datetime object."""
        return (dt or self.today()).month
    
    def current_quarter(self, dt: Optional[datetime] = None) -> int:
        """
        Get quarter (1-4) from datetime object.
        
        Returns
        -------
        int
            Quarter number (1, 2, 3, or 4)
        """
        month = (dt or self.today()).month
        return (month - 1) // 3 + 1
    
    # =============================================================================
    # MONTH-END UTILITIES
    # =============================================================================
    
    def month_end(self, dt: Optional[datetime] = None) -> date:
        """
        Get last day of month for given date.
        
        Parameters
        ----------
        dt : datetime, optional
            Date object (default: today)
            
        Returns
        -------
        date
            Last day of the month
        """
        dt = dt or self.today()
        next_month = dt.replace(day=28) + timedelta(days=4)
        return (next_month - timedelta(days=next_month.day)).date()
    
    def previous_month_end(self, offset: int = 0, dt: Optional[datetime] = None) -> date:
        """
        Get previous month-end date with optional offset.
        
        Parameters
        ----------
        offset : int
            Number of months to offset (default: 0)
            Positive = future, negative = past
        dt : datetime, optional
            Reference date (default: today)
            
        Returns
        -------
        date
            Month-end date
            
        Examples
        --------
        >>> cal = Calendar()
        >>> cal.previous_month_end()  # Last month's end
        >>> cal.previous_month_end(offset=-1)  # Two months ago
        >>> cal.previous_month_end(offset=1)  # Next month's end
        """
        dt = dt or self.today()
        first_of_month = dt.replace(day=1)
        last_month_end = (first_of_month - timedelta(days=1)).date()
        
        if offset != 0:
            last_month_end = (last_month_end + relativedelta(months=offset))
            # Ensure we're still at month end after offset
            return self.month_end(datetime.combine(last_month_end, datetime.min.time()))
        
        return last_month_end
    
    def month_end_list(self, start_date: Union[str, datetime], 
                      end_date: Union[str, datetime]) -> List[str]:
        """
        Generate list of month-end dates between start and end dates.
        
        Parameters
        ----------
        start_date : str or datetime
            Start date
        end_date : str or datetime
            End date
            
        Returns
        -------
        list
            Month-end dates as strings (YYYY-MM-DD)
        """
        dates = pd.date_range(
            pd.to_datetime(start_date), 
            pd.to_datetime(end_date), 
            freq='M'
        )
        return dates.strftime('%Y-%m-%d').tolist()
    
    # =============================================================================
    # QUARTER-END UTILITIES
    # =============================================================================
    
    def quarter_end(self, dt: Optional[datetime] = None) -> date:
        """
        Get quarter-end date for given date.
        
        Parameters
        ----------
        dt : datetime, optional
            Date object (default: today)
            
        Returns
        -------
        date
            Quarter-end date
        """
        dt = dt or self.today()
        quarter = self.current_quarter(dt)
        year = dt.year
        month = quarter * 3
        return self.month_end(datetime(year, month, 1))
    
    def previous_quarter_end(self, dt: Optional[datetime] = None) -> date:
        """
        Get previous quarter-end date.
        
        Parameters
        ----------
        dt : datetime, optional
            Reference date (default: today)
            
        Returns
        -------
        date
            Previous quarter-end date
        """
        dt = dt or self.today()
        rr = rrule.rrule(
            rrule.DAILY,
            bymonth=self.QUARTER_END_MONTHS,
            bymonthday=-1,
            dtstart=dt - timedelta(days=100)
        )
        result = rr.before(dt, inc=False)
        return result.date() if result else None
    
    def next_quarter_end(self, dt: Optional[datetime] = None) -> date:
        """Get next quarter-end date."""
        dt = dt or self.today()
        rr = rrule.rrule(
            rrule.DAILY,
            bymonth=self.QUARTER_END_MONTHS,
            bymonthday=-1,
            dtstart=dt + timedelta(days=1)
        )
        result = rr.after(dt, inc=False)
        return result.date() if result else None
    
    def prior_quarter_end(self, dt: Optional[datetime] = None) -> date:
        """
        Get quarter-end two quarters ago.
        
        Alias for getting the quarter-end before the previous quarter-end.
        """
        prev_qtr = self.previous_quarter_end(dt)
        return (prev_qtr - relativedelta(months=3))
    
    def quarter_end_list(self, start_date: Union[str, datetime], 
                        end_date: Union[str, datetime]) -> List[str]:
        """
        Generate list of quarter-end dates between start and end dates.
        
        Parameters
        ----------
        start_date : str or datetime
            Start date
        end_date : str or datetime
            End date
            
        Returns
        -------
        list
            Quarter-end dates as strings (YYYY-MM-DD)
            
        Examples
        --------
        >>> cal = Calendar()
        >>> quarters = cal.quarter_end_list('2024-01-01', '2024-12-31')
        >>> # Returns: ['2024-03-31', '2024-06-30', '2024-09-30', '2024-12-31']
        """
        dates = pd.date_range(
            pd.to_datetime(start_date),
            pd.to_datetime(end_date) + pd.offsets.QuarterBegin(1),
            freq='Q'
        )
        return dates.strftime('%Y-%m-%d').tolist()
    
    def quarters_in_year(self, year: int) -> List[date]:
        """
        Get all quarter-end dates for a specific year.
        
        Parameters
        ----------
        year : int
            Year
            
        Returns
        -------
        list
            List of four quarter-end dates
        """
        return [
            date(year, 3, 31),
            date(year, 6, 30),
            date(year, 9, 30),
            date(year, 12, 31)
        ]
    
    # =============================================================================
    # TRADING DAY UTILITIES
    # =============================================================================
    
    def _get_trading_days_cache(self) -> pd.DatetimeIndex:
        """Get or create cached trading days."""
        if self._trading_days_cache is None:
            self._trading_days_cache = self._market_calendar.valid_days(
                start_date=self.CALENDAR_CACHE_START,
                end_date=self.CALENDAR_CACHE_END
            )
        return self._trading_days_cache
    
    def is_trading_day(self, dt: Union[datetime, date, str]) -> bool:
        """
        Check if a date is a trading day.
        
        Parameters
        ----------
        dt : datetime, date, or str
            Date to check
            
        Returns
        -------
        bool
            True if trading day, False otherwise
        """
        dt = pd.to_datetime(dt)
        trading_days = self._get_trading_days_cache()
        return dt.normalize() in trading_days
    
    def closest_trading_day(self, dt: Union[datetime, date, str], 
                           direction: str = 'nearest') -> pd.Timestamp:
        """
        Find closest trading day to given date.
        
        Parameters
        ----------
        dt : datetime, date, or str
            Reference date
        direction : str
            'nearest', 'previous', or 'next'
            
        Returns
        -------
        pd.Timestamp
            Closest trading day
            
        Examples
        --------
        >>> cal = Calendar()
        >>> cal.closest_trading_day('2024-07-04')  # July 4 is a holiday
        >>> cal.closest_trading_day('2024-07-04', direction='next')
        """
        dt = pd.to_datetime(dt)
        trading_days = self._get_trading_days_cache()
        
        if direction == 'nearest':
            # Find closest by timestamp difference
            deltas = np.abs((trading_days - dt).total_seconds())
            idx = np.argmin(deltas)
            return trading_days[idx]
        
        elif direction == 'previous':
            valid_days = trading_days[trading_days <= dt]
            return valid_days[-1] if len(valid_days) > 0 else None
        
        elif direction == 'next':
            valid_days = trading_days[trading_days >= dt]
            return valid_days[0] if len(valid_days) > 0 else None
        
        else:
            raise ValueError("direction must be 'nearest', 'previous', or 'next'")
    
    def previous_trading_day(self, dt: Optional[datetime] = None, n: int = 1) -> pd.Timestamp:
        """
        Get the nth previous trading day.
        
        Parameters
        ----------
        dt : datetime, optional
            Reference date (default: today)
        n : int
            Number of trading days back (default: 1)
            
        Returns
        -------
        pd.Timestamp
            Previous trading day
        """
        dt = pd.to_datetime(dt or self.today())
        trading_days = self._get_trading_days_cache()
        valid_days = trading_days[trading_days < dt]
        
        if len(valid_days) < n:
            raise ValueError(f"Not enough trading days before {dt}")
        
        return valid_days[-n]
    
    def next_trading_day(self, dt: Optional[datetime] = None, n: int = 1) -> pd.Timestamp:
        """Get the nth next trading day."""
        dt = pd.to_datetime(dt or self.today())
        trading_days = self._get_trading_days_cache()
        valid_days = trading_days[trading_days > dt]
        
        if len(valid_days) < n:
            raise ValueError(f"Not enough trading days after {dt}")
        
        return valid_days[n - 1]
    
    def get_trading_days(self, start_date: Union[str, datetime], 
                        end_date: Union[str, datetime]) -> pd.DatetimeIndex:
        """
        Get all trading days between start and end dates (inclusive).
        
        Parameters
        ----------
        start_date : str or datetime
            Start date
        end_date : str or datetime
            End date
            
        Returns
        -------
        pd.DatetimeIndex
            Trading days
        """
        return self._market_calendar.valid_days(
            start_date=start_date,
            end_date=end_date
        )
    
    def count_trading_days(self, start_date: Union[str, datetime], 
                          end_date: Union[str, datetime]) -> int:
        """Count number of trading days between dates."""
        return len(self.get_trading_days(start_date, end_date))
    
    # =============================================================================
    # BUSINESS DAY UTILITIES
    # =============================================================================
    
    def add_business_days(self, dt: Union[datetime, date, str], days: int) -> date:
        """
        Add business days to a date.
        
        Parameters
        ----------
        dt : datetime, date, or str
            Starting date
        days : int
            Number of business days to add (negative for subtraction)
            
        Returns
        -------
        date
            Resulting date
        """
        dt = pd.to_datetime(dt)
        result = dt + pd.offsets.BDay(days)
        return result.date()
    
    def business_days_between(self, start_date: Union[str, datetime],
                             end_date: Union[str, datetime]) -> int:
        """
        Count business days between two dates.
        
        Note: Uses standard business day convention (Mon-Fri),
        not exchange-specific trading days.
        """
        return len(pd.bdate_range(start_date, end_date))
    
    # =============================================================================
    # DATE RANGE GENERATION
    # =============================================================================
    
    def date_range(self, start_date: Union[str, datetime], 
                   end_date: Union[str, datetime],
                   freq: str = 'D') -> pd.DatetimeIndex:
        """
        Generate date range with specified frequency.
        
        Parameters
        ----------
        start_date : str or datetime
            Start date
        end_date : str or datetime
            End date
        freq : str
            Frequency: 'D' (daily), 'W' (weekly), 'M' (month-end),
            'Q' (quarter-end), 'Y' (year-end), 'B' (business day)
            
        Returns
        -------
        pd.DatetimeIndex
            Date range
        """
        return pd.date_range(start_date, end_date, freq=freq)
    
    # =============================================================================
    # FISCAL PERIOD UTILITIES
    # =============================================================================
    
    def fiscal_quarter(self, dt: Optional[datetime] = None, 
                      fiscal_year_end: int = 12) -> Tuple[int, int]:
        """
        Get fiscal year and quarter for a date.
        
        Parameters
        ----------
        dt : datetime, optional
            Date (default: today)
        fiscal_year_end : int
            Fiscal year end month (1-12, default: 12 for calendar year)
            
        Returns
        -------
        tuple
            (fiscal_year, fiscal_quarter)
            
        Examples
        --------
        >>> cal = Calendar()
        >>> cal.fiscal_quarter(datetime(2024, 2, 15), fiscal_year_end=6)
        >>> # Returns (2024, 3) for June fiscal year-end
        """
        dt = dt or self.today()
        
        # Calculate months from fiscal year start
        months_from_fy_start = (dt.month - fiscal_year_end - 1) % 12
        
        fiscal_year = dt.year
        if dt.month > fiscal_year_end:
            fiscal_year += 1
        
        fiscal_quarter = (months_from_fy_start // 3) + 1
        
        return fiscal_year, fiscal_quarter
    
    # =============================================================================
    # UTILITY METHODS
    # =============================================================================
    
    def format_date(self, dt: datetime, fmt: str = '%Y-%m-%d') -> str:
        """Format datetime to string."""
        return dt.strftime(fmt)
    
    def parse_date(self, date_str: str, fmt: Optional[str] = None) -> datetime:
        """
        Parse string to datetime.
        
        Parameters
        ----------
        date_str : str
            Date string
        fmt : str, optional
            Format string (if None, uses pandas parsing)
            
        Returns
        -------
        datetime
            Parsed datetime
        """
        if fmt:
            return datetime.strptime(date_str, fmt)
        return pd.to_datetime(date_str).to_pydatetime()
    
    def days_between(self, start_date: Union[datetime, date], 
                    end_date: Union[datetime, date]) -> int:
        """Calculate number of days between two dates."""
        if isinstance(start_date, datetime):
            start_date = start_date.date()
        if isinstance(end_date, datetime):
            end_date = end_date.date()
        
        return (end_date - start_date).days
    
    def is_month_end(self, dt: Union[datetime, date]) -> bool:
        """Check if date is last day of month."""
        dt_date = dt.date() if isinstance(dt, datetime) else dt
        next_day = dt_date + timedelta(days=1)
        return next_day.day == 1
    
    def is_quarter_end(self, dt: Union[datetime, date]) -> bool:
        """Check if date is last day of quarter."""
        dt_date = dt.date() if isinstance(dt, datetime) else dt
        return (self.is_month_end(dt_date) and 
                dt_date.month in self.QUARTER_END_MONTHS)
    
    def is_year_end(self, dt: Union[datetime, date]) -> bool:
        """Check if date is last day of year."""
        dt_date = dt.date() if isinstance(dt, datetime) else dt
        return dt_date.month == 12 and dt_date.day == 31

    