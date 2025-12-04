"""
Statistical Analysis and Data Description Toolkit
Expert-level helper classes for comprehensive data analysis, statistical testing, and formatting
"""

from typing import Union, List, Dict, Optional, Tuple, Any
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import shapiro, normaltest, jarque_bera, kstest, anderson
from scipy.stats import entropy, hmean, gmean, sem, skew, kurtosis
from statsmodels.tsa.stattools import pacf, acf, adfuller, kpss
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson
import warnings


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def first_not_null_date(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Find the first non-null date (index) for each column.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime index
        
    Returns
    -------
    dict
        Mapping of column name to first valid index
        
    Example
    -------
    >>> dates = pd.date_range('2024-01-01', periods=10)
    >>> df = pd.DataFrame({'A': [np.nan, np.nan, 1, 2], 'B': [5, 6, 7, 8]}, index=dates[:4])
    >>> first_not_null_date(df)
    {'A': Timestamp('2024-01-03'), 'B': Timestamp('2024-01-01')}
    """
    return {col: df[col].first_valid_index() for col in df.columns}


def last_not_null_date(df: pd.DataFrame) -> Dict[str, Any]:
    """Find the last non-null date (index) for each column."""
    return {col: df[col].last_valid_index() for col in df.columns}


def coverage_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize data coverage for each column.
    
    Returns DataFrame with first/last valid dates, count, and coverage %.
    """
    summary = pd.DataFrame({
        'first_date': [df[col].first_valid_index() for col in df.columns],
        'last_date': [df[col].last_valid_index() for col in df.columns],
        'count': df.count(),
        'total': len(df),
        'coverage_pct': (df.count() / len(df) * 100).round(2)
    }, index=df.columns)
    return summary


# =============================================================================
# DESCRIPTIVE STATISTICS CLASS
# =============================================================================

class Descriptive:
    """
    Comprehensive descriptive statistics analyzer for DataFrames and Series.
    
    Provides extensive statistical measures including:
    - Central tendency (mean, median, mode, trimmed mean)
    - Dispersion (std, var, IQR, MAD, range)
    - Distribution shape (skewness, kurtosis)
    - Outlier detection (IQR, z-score methods)
    - Time series properties (autocorrelation, stationarity)
    - Normality tests
    
    Parameters
    ----------
    data : pd.DataFrame or pd.Series
        Input data for analysis
        
    Examples
    --------
    >>> data = pd.DataFrame({'returns': np.random.randn(1000)})
    >>> desc = Descriptive(data)
    >>> summary = desc.describe()
    >>> desc.summary_table(precision=4)
    >>> desc.distribution_analysis()
    """
    
    def __init__(self, data: Union[pd.DataFrame, pd.Series]) -> None:
        if not isinstance(data, (pd.DataFrame, pd.Series)):
            raise TypeError("Input must be pandas DataFrame or Series")
        
        self.data = data.to_frame(name=data.name or "series") if isinstance(data, pd.Series) else data
        self._numeric_data = self.data.select_dtypes(include=[np.number])
    
    def describe(self, include_quantiles: bool = True, include_outliers: bool = True,
                include_timeseries: bool = True) -> pd.DataFrame:
        """
        Generate comprehensive descriptive statistics.
        
        Parameters
        ----------
        include_quantiles : bool
            Include quantile analysis
        include_outliers : bool
            Include outlier detection statistics
        include_timeseries : bool
            Include time series statistics (ACF, PACF)
            
        Returns
        -------
        pd.DataFrame
            Transposed statistics table
        """
        stats = {}
        
        # Basic statistics
        stats['count'] = self._numeric_data.count()
        stats['mean'] = self._numeric_data.mean()
        stats['median'] = self._numeric_data.median()
        stats['std'] = self._numeric_data.std()
        stats['var'] = self._numeric_data.var()
        stats['min'] = self._numeric_data.min()
        stats['max'] = self._numeric_data.max()
        
        # Dispersion measures
        stats['range'] = self._numeric_data.max() - self._numeric_data.min()
        stats['cv'] = self._numeric_data.std() / self._numeric_data.mean().abs()  # Coefficient of variation
        stats['mad'] = self._numeric_data.apply(lambda x: np.abs(x - x.mean()).mean())  # Mean absolute deviation
        stats['sem'] = self._numeric_data.apply(lambda x: sem(x.dropna()) if len(x.dropna()) > 1 else np.nan)
        
        # Distribution shape
        stats['skew'] = self._numeric_data.skew()
        stats['kurtosis'] = self._numeric_data.kurtosis()
        
        # Missing data
        stats['missing_count'] = self._numeric_data.isnull().sum()
        stats['missing_pct'] = (self._numeric_data.isnull().mean() * 100).round(2)
        
        # Unique values
        stats['unique_count'] = self._numeric_data.nunique()
        
        # Quantiles
        if include_quantiles:
            quantiles = self._numeric_data.quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
            quantiles.index = ['p01', 'p05', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99']
            quantiles = quantiles.T
            
            # IQR and percentile ranges
            q1 = self._numeric_data.quantile(0.25)
            q3 = self._numeric_data.quantile(0.75)
            stats['iqr'] = q3 - q1
            stats['p90_p10_range'] = self._numeric_data.quantile(0.90) - self._numeric_data.quantile(0.10)
        
        # Outlier detection
        if include_outliers:
            q1 = self._numeric_data.quantile(0.25)
            q3 = self._numeric_data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            stats['outliers_iqr'] = ((self._numeric_data < lower_bound) | (self._numeric_data > upper_bound)).sum()
            
            z_scores = (self._numeric_data - self._numeric_data.mean()) / self._numeric_data.std()
            stats['outliers_3std'] = (z_scores.abs() > 3).sum()
        
        # Time series properties
        if include_timeseries:
            stats['autocorr_lag1'] = self._numeric_data.apply(lambda x: x.autocorr(lag=1) if len(x.dropna()) > 1 else np.nan)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stats['highest_acf_lag'] = self._numeric_data.apply(self._find_highest_acf_lag)
                stats['highest_pacf_lag'] = self._numeric_data.apply(self._find_highest_pacf_lag)
        
        # Compile results
        stats_df = pd.DataFrame(stats)
        
        if include_quantiles:
            stats_df = pd.concat([stats_df, quantiles], axis=1)
        
        return stats_df.T
    
    @staticmethod
    def _find_highest_acf_lag(series: pd.Series, max_lags: int = 10) -> float:
        """Find lag with highest autocorrelation."""
        try:
            clean = series.dropna()
            if len(clean) < max_lags + 2:
                return np.nan
            acf_values = acf(clean, nlags=min(max_lags, len(clean) // 2 - 1), fft=False)
            return np.argmax(np.abs(acf_values[1:])) + 1
        except:
            return np.nan
    
    @staticmethod
    def _find_highest_pacf_lag(series: pd.Series, max_lags: int = 10) -> float:
        """Find lag with highest partial autocorrelation."""
        try:
            clean = series.dropna()
            if len(clean) < max_lags + 2:
                return np.nan
            pacf_values = pacf(clean, nlags=min(max_lags, len(clean) // 2 - 1))
            return np.argmax(np.abs(pacf_values[1:])) + 1
        except:
            return np.nan
    
    def summary_table(self, precision: int = 4) -> pd.DataFrame:
        """
        Generate formatted summary table with key statistics.
        
        Returns clean, presentation-ready DataFrame.
        """
        key_stats = ['count', 'mean', 'median', 'std', 'min', 'max', 
                    'skew', 'kurtosis', 'missing_pct']
        
        full_stats = self.describe(include_quantiles=False, include_timeseries=False)
        summary = full_stats.loc[key_stats]
        
        return summary.round(precision)
    
    def distribution_analysis(self) -> pd.DataFrame:
        """
        Analyze distribution characteristics and normality.
        
        Returns
        -------
        pd.DataFrame
            Distribution statistics and test results
        """
        results = {}
        
        for col in self._numeric_data.columns:
            series = self._numeric_data[col].dropna()
            
            if len(series) < 3:
                continue
            
            col_stats = {
                'n_obs': len(series),
                'mean': series.mean(),
                'median': series.median(),
                'std': series.std(),
                'skew': series.skew(),
                'kurtosis': series.kurtosis(),
                'min': series.min(),
                'max': series.max(),
            }
            
            # Normality tests
            if len(series) >= 8:
                try:
                    shapiro_stat, shapiro_p = shapiro(series)
                    col_stats['shapiro_p'] = shapiro_p
                    col_stats['is_normal_shapiro'] = shapiro_p > 0.05
                except:
                    col_stats['shapiro_p'] = np.nan
                    col_stats['is_normal_shapiro'] = np.nan
            
            if len(series) >= 20:
                try:
                    jb_stat, jb_p = jarque_bera(series)
                    col_stats['jarque_bera_p'] = jb_p
                    col_stats['is_normal_jb'] = jb_p > 0.05
                except:
                    col_stats['jarque_bera_p'] = np.nan
                    col_stats['is_normal_jb'] = np.nan
            
            results[col] = col_stats
        
        return pd.DataFrame(results).T
    
    def correlation_matrix(self, method: str = 'pearson', min_periods: int = 30) -> pd.DataFrame:
        """
        Calculate correlation matrix.
        
        Parameters
        ----------
        method : str
            Correlation method: 'pearson', 'spearman', or 'kendall'
        min_periods : int
            Minimum observations required
            
        Returns
        -------
        pd.DataFrame
            Correlation matrix
        """
        return self._numeric_data.corr(method=method, min_periods=min_periods)
    
    def outlier_report(self, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """
        Generate detailed outlier report.
        
        Parameters
        ----------
        method : str
            Detection method: 'iqr' or 'zscore'
        threshold : float
            Threshold multiplier (1.5 for IQR, 3 for z-score)
            
        Returns
        -------
        pd.DataFrame
            Outlier statistics per column
        """
        report = {}
        
        for col in self._numeric_data.columns:
            series = self._numeric_data[col].dropna()
            
            if method == 'iqr':
                q1, q3 = series.quantile([0.25, 0.75])
                iqr = q3 - q1
                lower = q1 - threshold * iqr
                upper = q3 + threshold * iqr
                outliers = series[(series < lower) | (series > upper)]
            else:  # zscore
                z_scores = (series - series.mean()) / series.std()
                outliers = series[z_scores.abs() > threshold]
            
            report[col] = {
                'n_outliers': len(outliers),
                'pct_outliers': len(outliers) / len(series) * 100,
                'min_outlier': outliers.min() if len(outliers) > 0 else np.nan,
                'max_outlier': outliers.max() if len(outliers) > 0 else np.nan,
            }
        
        return pd.DataFrame(report).T


# =============================================================================
# TIME SERIES ANALYSIS CLASS
# =============================================================================

class TimeSeriesAnalysis:
    """
    Time series specific statistical analysis.
    
    Examples
    --------
    >>> returns = pd.Series(np.random.randn(1000))
    >>> tsa = TimeSeriesAnalysis(returns)
    >>> tsa.stationarity_test()
    >>> tsa.autocorrelation_summary(lags=20)
    """
    
    def __init__(self, data: Union[pd.DataFrame, pd.Series]):
        if isinstance(data, pd.Series):
            self.data = data
        elif isinstance(data, pd.DataFrame):
            self.data = data.iloc[:, 0] if data.shape[1] == 1 else data
        else:
            raise TypeError("Input must be pandas Series or DataFrame")
    
    def stationarity_test(self, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Test for stationarity using ADF and KPSS tests.
        
        Parameters
        ----------
        alpha : float
            Significance level
            
        Returns
        -------
        dict
            Test statistics and interpretations
        """
        if isinstance(self.data, pd.Series):
            series = self.data.dropna()
        else:
            series = self.data.iloc[:, 0].dropna()
        
        results = {}
        
        # Augmented Dickey-Fuller test
        try:
            adf_result = adfuller(series, autolag='AIC')
            results['adf'] = {
                'statistic': adf_result[0],
                'p_value': adf_result[1],
                'critical_values': adf_result[4],
                'is_stationary': adf_result[1] < alpha
            }
        except Exception as e:
            results['adf'] = {'error': str(e)}
        
        # KPSS test
        try:
            kpss_result = kpss(series, regression='c', nlags='auto')
            results['kpss'] = {
                'statistic': kpss_result[0],
                'p_value': kpss_result[1],
                'critical_values': kpss_result[3],
                'is_stationary': kpss_result[1] > alpha
            }
        except Exception as e:
            results['kpss'] = {'error': str(e)}
        
        return results
    
    def autocorrelation_summary(self, lags: int = 20) -> pd.DataFrame:
        """
        Calculate ACF and PACF values.
        
        Parameters
        ----------
        lags : int
            Number of lags to compute
            
        Returns
        -------
        pd.DataFrame
            ACF and PACF values
        """
        if isinstance(self.data, pd.Series):
            series = self.data.dropna()
        else:
            series = self.data.iloc[:, 0].dropna()
        
        max_lags = min(lags, len(series) // 2 - 1)
        
        acf_vals = acf(series, nlags=max_lags, fft=False)
        pacf_vals = pacf(series, nlags=max_lags)
        
        return pd.DataFrame({
            'lag': range(max_lags + 1),
            'acf': acf_vals,
            'pacf': pacf_vals
        })


# =============================================================================
# SIGNIFICANCE TESTING CLASS
# =============================================================================

class Significance:
    """
    Statistical significance testing and formatting utilities.
    
    Provides methods for:
    - P-value annotation with significance stars
    - Significance level indicators
    - Hypothesis test formatting
    - Multiple comparison corrections
    
    Examples
    --------
    >>> sig = Significance()
    >>> coef_df = pd.DataFrame({'A': [0.5, 0.3], 'B': [0.7, 0.1]})
    >>> pval_df = pd.DataFrame({'A': [0.001, 0.06], 'B': [0.04, 0.15]})
    >>> annotated = sig.annotate_with_stars(coef_df, pval_df)
    """
    
    SIGNIFICANCE_LEVELS = {
        0.001: '***',
        0.01: '**',
        0.05: '*',
        0.10: '†'
    }
    
    def __init__(self, levels: Optional[Dict[float, str]] = None):
        """
        Parameters
        ----------
        levels : dict, optional
            Custom significance levels mapping
        """
        self.levels = levels or self.SIGNIFICANCE_LEVELS
    
    def annotate_with_stars(self, values: pd.DataFrame, pvalues: pd.DataFrame, 
                           precision: int = 4) -> pd.DataFrame:
        """
        Annotate values with significance stars based on p-values.
        
        Parameters
        ----------
        values : pd.DataFrame
            Values to annotate (e.g., coefficients, correlations)
        pvalues : pd.DataFrame
            Corresponding p-values (same shape as values)
        precision : int
            Decimal precision for values
            
        Returns
        -------
        pd.DataFrame
            Annotated values with stars
            
        Examples
        --------
        >>> sig = Significance()
        >>> vals = pd.DataFrame({'A': [0.5, 0.3]})
        >>> pvals = pd.DataFrame({'A': [0.001, 0.06]})
        >>> sig.annotate_with_stars(vals, pvals)
                    A
        0  0.5000***
        1   0.3000†
        """
        if values.shape != pvalues.shape:
            raise ValueError("Values and p-values must have same shape")
        
        result = values.copy()
        
        for col in values.columns:
            result[col] = result.apply(
                lambda row: self._add_stars(row[col], pvalues.at[row.name, col], precision),
                axis=1
            )
        
        return result
    
    def _add_stars(self, value: float, pvalue: float, precision: int = 4) -> str:
        """Add significance stars to a single value."""
        if not isinstance(value, (int, float)) or not isinstance(pvalue, (int, float)):
            return str(value)
        
        if pd.isna(value) or pd.isna(pvalue):
            return 'NaN'
        
        stars = ''
        for threshold, symbol in sorted(self.levels.items()):
            if pvalue < threshold:
                stars = symbol
                break
        
        return f"{value:.{precision}f}{stars}"
    
    def add_significance_flags(self, df: pd.DataFrame, pvalue_col: str = 'p_value',
                              levels: List[float] = [0.01, 0.05, 0.10]) -> pd.DataFrame:
        """
        Add boolean significance columns to DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with p-value column
        pvalue_col : str
            Name of p-value column
        levels : list
            Significance levels to test
            
        Returns
        -------
        pd.DataFrame
            DataFrame with added significance columns
        """
        if pvalue_col not in df.columns:
            raise ValueError(f"Column '{pvalue_col}' not found in DataFrame")
        
        result = df.copy()
        
        for level in levels:
            col_name = f'sig_{int(level*100):02d}'
            result[col_name] = result[pvalue_col] < level
        
        return result
    
    def significance_summary(self, pvalues: Union[pd.Series, pd.DataFrame],
                            levels: List[float] = [0.01, 0.05, 0.10]) -> pd.DataFrame:
        """
        Summarize count of significant results at each level.
        
        Parameters
        ----------
        pvalues : pd.Series or pd.DataFrame
            P-values to summarize
        levels : list
            Significance levels
            
        Returns
        -------
        pd.DataFrame
            Summary of significant results
        """
        if isinstance(pvalues, pd.Series):
            pvalues = pvalues.to_frame()
        
        summary = {}
        
        for level in levels:
            summary[f'p<{level}'] = (pvalues < level).sum()
            summary[f'pct_p<{level}'] = ((pvalues < level).sum() / pvalues.count() * 100).round(2)
        
        return pd.DataFrame(summary)
    
    @staticmethod
    def bonferroni_correction(pvalues: Union[pd.Series, np.ndarray], 
                             alpha: float = 0.05) -> Tuple[np.ndarray, float]:
        """
        Apply Bonferroni correction for multiple comparisons.
        
        Parameters
        ----------
        pvalues : array-like
            Original p-values
        alpha : float
            Family-wise error rate
            
        Returns
        -------
        tuple
            (significant_flags, adjusted_alpha)
        """
        pvals = np.asarray(pvalues)
        n_tests = len(pvals)
        adjusted_alpha = alpha / n_tests
        significant = pvals < adjusted_alpha
        
        return significant, adjusted_alpha
    
    @staticmethod
    def format_pvalue(p: float, precision: int = 4) -> str:
        """
        Format p-value for presentation.
        
        Examples
        --------
        >>> Significance.format_pvalue(0.0001)
        '< 0.001'
        >>> Significance.format_pvalue(0.0456)
        '0.0456'
        """
        if pd.isna(p):
            return 'NaN'
        elif p < 0.001:
            return '< 0.001'
        elif p < 0.01:
            return f'{p:.4f}'
        else:
            return f'{p:.{precision}f}'