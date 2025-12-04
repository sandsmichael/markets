# fundamental analysis
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
from scipy import stats


class FundamentalAnalysis:
    """
    Cross-sectional fundamental analysis for peer comparison.
    
    Analyzes a dataset of companies with fundamental metrics and produces
    actionable insights including percentile rankings, peer comparisons,
    quality scores, and investment signals.
    
    """
    
    def __init__(self, df):
        """
        Initialize analyzer with fundamental data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Cross-sectional fundamental data with Ticker as index
        """
        self.df = df.copy()
        self._convert_to_numeric()
        self._validate_data()
        # self._calculate_derived_metrics()


    
    def _convert_to_numeric(self):
        """Convert all columns with convertible numeric values to proper dtypes."""
        for col in self.df.columns:
            # Skip columns that are already numeric
            if pd.api.types.is_numeric_dtype(self.df[col]):
                continue
            
            # Try to convert to numeric, keeping original if conversion fails
            converted = pd.to_numeric(self.df[col], errors='coerce')
            
            # Only apply conversion if at least some values were successfully converted
            # and not all became NaN (which would indicate the column is truly non-numeric)
            if converted.notna().any() and not (converted.isna().all() and self.df[col].notna().any()):
                self.df[col] = converted
        
    def _validate_data(self):
        """Validate input data has required fields."""
        required_fields = ['MarketCapitalization', 'totalRevenue', 'netIncome']
        missing = [f for f in required_fields if f not in self.df.columns]
        if missing:
            print(f"Warning: Missing recommended fields: {missing}")
    
    def _calculate_derived_metrics(self, ltm=False, by_sector=False):
        """
        Calculate additional derived metrics (prefixed with _ to distinguish from input data).
        
        Parameters
        ----------
        ltm : bool
            If True, uses ltm_ prefixed columns and removes the prefix before calculation
        by_sector : bool
            If True, aggregates data by sector first, then calculates sector-level metrics
            
        Returns
        -------
        pd.DataFrame or None
            If by_sector=True, returns DataFrame with sector-level metrics
            If by_sector=False, updates self.df in place and returns None
        """
        # Store original df to avoid mutation issues when called multiple times
        # Use self.df directly if _original_df doesn't exist (allows by_sector=True to work standalone)
        if not hasattr(self, '_original_df'):
            self._original_df = self.df.copy()
        
        # Always start from the original dataframe (or self.df if _original_df not yet set)
        df = self._original_df.copy() if hasattr(self, '_original_df') else self.df.copy()

        if ltm:
            # Select ltm columns plus MarketCapitalization
            ltm_cols = [c for c in df.columns if 'ltm_' in c]
            other_cols = ['MarketCapitalization', 'totalStockholderEquity', 'bf_epsEstimate', 'totalLiab', 'totalAssets', 
                         'totalCurrentAssets', 'totalCurrentLiabilities', 'inventory', 'longTermDebt', 'cash', 
                         'cashAndEquivalents', 'shortTermDebt', 'accountsPayable', 'netReceivables',
                         'commonStockSharesOutstanding', 'propertyPlantEquipment', 'intangibleAssets'] 
            if by_sector:
                other_cols = ['Sector'] + other_cols
            available_cols = [c for c in ltm_cols + other_cols if c in df.columns]
            df = df[available_cols].copy()
            
            # Rename ltm_ columns by removing the prefix
            rename_dict = {c: c.replace('ltm_', '') for c in ltm_cols}
            df = df.rename(columns=rename_dict)
        
        # If by_sector, aggregate constituent data first
        if by_sector:
            if 'Sector' not in df.columns:
                print("Sector column not found in data")
                return pd.DataFrame()
            
            # Calculate total earnings at company level before aggregation (EPS is per-share, can't be summed)
            if 'epsActual' in df.columns and 'commonStockSharesOutstanding' in df.columns:
                df['totalEarnings'] = df['epsActual'] * df['commonStockSharesOutstanding']
            if 'bf_epsEstimate' in df.columns and 'commonStockSharesOutstanding' in df.columns:
                df['totalFwdEarnings'] = df['bf_epsEstimate'] * df['commonStockSharesOutstanding']
            
            # Define which columns to sum
            sum_cols = [
                'MarketCapitalization', 'totalRevenue', 'netIncome', 'totalAssets', 'totalStockholderEquity',
                'totalLiab', 'totalCurrentAssets', 'totalCurrentLiabilities', 'longTermDebt', 'shortTermDebt',
                'cash', 'cashAndEquivalents', 'inventory', 'accountsPayable', 'netReceivables',
                'commonStockSharesOutstanding', 'propertyPlantEquipment', 'intangibleAssets',
                'freeCashFlow', 'totalCashFromOperatingActivities', 'totalCashFromFinancingActivities',
                'capitalExpenditures', 'grossProfit', 'operatingIncome', 'ebit', 'ebitda',
                'costOfRevenue', 'totalOperatingExpenses', 'researchDevelopment',
                'changeInWorkingCapital', 'totalCashflowsFromInvestingActivities',
                'totalEarnings', 'totalFwdEarnings'
            ]
            
            # Filter to only columns that exist in df
            sum_cols = [c for c in sum_cols if c in df.columns]
            
            # Aggregate by sector
            df = df.groupby('Sector')[sum_cols].sum().reset_index()

        # Valuation metrics
        if by_sector:
            df['_PriceToEarnings'] = df['MarketCapitalization'] / df['totalEarnings'].replace(0, np.nan)
            df['_PriceToFwdEarnings'] = df['MarketCapitalization'] / df['totalFwdEarnings'].replace(0, np.nan)
        else:
            df['_PriceToEarnings'] = df['MarketCapitalization'] / (df['epsActual'] * df['commonStockSharesOutstanding']).replace(0, np.nan)
            df['_PriceToFwdEarnings'] = df['MarketCapitalization'] / (df['bf_epsEstimate'] * df['commonStockSharesOutstanding']).replace(0, np.nan)

        df['_PriceToSales'] = df['MarketCapitalization'] / df['totalRevenue'].replace(0, np.nan)
        df['_PriceToBook'] = df['MarketCapitalization'] / df['totalStockholderEquity'].replace(0, np.nan)
        df['_PriceToFCF'] = df['MarketCapitalization'] / df['freeCashFlow'].replace(0, np.nan)
        df['_EV_to_EBITDA'] = (df['MarketCapitalization'] + df['longTermDebt'] - df['cash']) / df['ebitda'].replace(0, np.nan)
        df['_EV_to_Sales'] = (df['MarketCapitalization'] + df['longTermDebt'] - df['cash']) / df['totalRevenue'].replace(0, np.nan)
        # df['_PEG_Ratio'] = df['_PriceToEarnings'] / df['_EPS_Growth_Estimate']

        # Profitability metrics
        df['_GrossProfitMargin'] = (df['grossProfit'] / df['totalRevenue'].replace(0, np.nan)) * 100
        df['_OperatingProfitMargin'] = (df['operatingIncome'] / df['totalRevenue'].replace(0, np.nan)) * 100
        df['_NetProfitMargin'] = (df['netIncome'] / df['totalRevenue'].replace(0, np.nan)) * 100
        df['_EBITDA_Margin'] = (df['ebitda'] / df['totalRevenue'].replace(0, np.nan)) * 100
        df['_EBIT_Margin'] = (df['ebit'] / df['totalRevenue'].replace(0, np.nan)) * 100
        df['_ROE'] = (df['netIncome'] / df['totalStockholderEquity'].replace(0, np.nan)) * 100
        df['_ROA'] = (df['netIncome'] / df['totalAssets'].replace(0, np.nan)) * 100
        df['_ROIC'] = (df['netIncome'] / (df['totalStockholderEquity'] + df['longTermDebt']).replace(0, np.nan)) * 100
        df['_ROC'] = (df['ebit'] / (df['totalAssets'] - df['totalCurrentLiabilities']).replace(0, np.nan)) * 100

        # Leverage metrics
        df['_DebtToEquity'] = df['totalLiab'] / df['totalStockholderEquity'].replace(0, np.nan)
        df['_DebtRatio'] = df['totalLiab'] / df['totalAssets'].replace(0, np.nan)
        df['_NetDebt'] = df['longTermDebt'] + df['shortTermDebt'] - df['cash']
        df['_NetDebtToEquity'] = df['_NetDebt'] / df['totalStockholderEquity'].replace(0, np.nan)
        df['_NetDebtToEBITDA'] = df['_NetDebt'] / df['ebitda'].replace(0, np.nan)
        df['_InterestCoverage'] = df['ebit'] / (df['totalLiab'] * 0.05).replace(0, np.nan)  # Assuming 5% avg interest rate
        df['_EquityMultiplier'] = df['totalAssets'] / df['totalStockholderEquity'].replace(0, np.nan)

        # Liquidity metrics
        df['_CurrentRatio'] = df['totalCurrentAssets'] / df['totalCurrentLiabilities'].replace(0, np.nan)
        df['_QuickRatio'] = (df['totalCurrentAssets'] - df['inventory']) / df['totalCurrentLiabilities'].replace(0, np.nan)
        df['_CashRatio'] = df['cashAndEquivalents'] / df['totalCurrentLiabilities'].replace(0, np.nan)
        df['_WorkingCapital'] = df['totalCurrentAssets'] - df['totalCurrentLiabilities']
        df['_WorkingCapitalRatio'] = df['_WorkingCapital'] / df['totalAssets'].replace(0, np.nan)

        # Efficiency metrics
        df['_AssetTurnover'] = df['totalRevenue'] / df['totalAssets'].replace(0, np.nan)
        df['_InventoryTurnover'] = df['costOfRevenue'] / df['inventory'].replace(0, np.nan)
        df['_ReceivablesTurnover'] = df['totalRevenue'] / df['netReceivables'].replace(0, np.nan)
        df['_PayablesTurnover'] = df['costOfRevenue'] / df['accountsPayable'].replace(0, np.nan)
        df['_FixedAssetTurnover'] = df['totalRevenue'] / df['propertyPlantEquipment'].replace(0, np.nan)
        df['_DaysInventoryOutstanding'] = 365 / df['_InventoryTurnover'].replace(0, np.nan)
        df['_DaysSalesOutstanding'] = 365 / df['_ReceivablesTurnover'].replace(0, np.nan)
        df['_DaysPayablesOutstanding'] = 365 / df['_PayablesTurnover'].replace(0, np.nan)
        df['_CashConversionCycle'] = df['_DaysInventoryOutstanding'] + df['_DaysSalesOutstanding'] - df['_DaysPayablesOutstanding']

        # Cash flow metrics
        df['_FCF_Margin'] = (df['freeCashFlow'] / df['totalRevenue'].replace(0, np.nan)) * 100
        df['_FCF_Yield'] = (df['freeCashFlow'] / df['MarketCapitalization'].replace(0, np.nan)) * 100
        df['_OperatingCashFlow_Margin'] = (df['totalCashFromOperatingActivities'] / df['totalRevenue'].replace(0, np.nan)) * 100
        df['_CapexToRevenue'] = (df['capitalExpenditures'] / df['totalRevenue'].replace(0, np.nan)) * 100
        df['_CapexToOperatingCashFlow'] = df['capitalExpenditures'] / df['totalCashFromOperatingActivities'].replace(0, np.nan)
        df['_FCFPerShare'] = df['freeCashFlow'] / df['commonStockSharesOutstanding'].replace(0, np.nan)
        df['_OperatingCashFlowPerShare'] = df['totalCashFromOperatingActivities'] / df['commonStockSharesOutstanding'].replace(0, np.nan)

        # Quality metrics
        df['_FCF_to_NetIncome'] = df['freeCashFlow'] / df['netIncome'].replace(0, np.nan)
        df['_CashFlow_Quality'] = df['totalCashFromOperatingActivities'] / df['netIncome'].replace(0, np.nan)
        df['_Accruals'] = (df['netIncome'] - df['totalCashFromOperatingActivities']) / df['totalAssets'].replace(0, np.nan)
        df['_EarningsQualityScore'] = (df['_CashFlow_Quality'] + df['_FCF_to_NetIncome']) / 2
        df['_TangibleBookValue'] = df['totalStockholderEquity'] - df['intangibleAssets']
        df['_PriceToTangibleBook'] = df['MarketCapitalization'] / df['_TangibleBookValue'].replace(0, np.nan)

        # Growth proxy (using estimates if available)
        # df['_EPS_Growth_Estimate'] = ((df['bf_epsEstimate'] - df['epsActual']) / df['epsActual']) * 100
        df['_RevenuePerShare'] = df['totalRevenue'] / df['commonStockSharesOutstanding'].replace(0, np.nan)
        df['_BookValuePerShare'] = df['totalStockholderEquity'] / df['commonStockSharesOutstanding'].replace(0, np.nan)
        df['_TangibleBookValuePerShare'] = df['_TangibleBookValue'] / df['commonStockSharesOutstanding'].replace(0, np.nan)
        
        # DuPont Analysis components
        df['_DuPont_NetMargin'] = df['netIncome'] / df['totalRevenue'].replace(0, np.nan)
        df['_DuPont_AssetTurnover'] = df['totalRevenue'] / df['totalAssets'].replace(0, np.nan)
        df['_DuPont_EquityMultiplier'] = df['totalAssets'] / df['totalStockholderEquity'].replace(0, np.nan)
        df['_DuPont_ROE'] = df['_DuPont_NetMargin'] * df['_DuPont_AssetTurnover'] * df['_DuPont_EquityMultiplier']
        
        # Factor Scores (percentile-based composite metrics)
        # Value Factor: Low valuation multiples (invert so low = high percentile)
        value_pe = 100 - df['_PriceToEarnings'].rank(pct=True) * 100
        value_ps = 100 - df['_PriceToSales'].rank(pct=True) * 100
        value_pb = 100 - df['_PriceToBook'].rank(pct=True) * 100
        value_ev_ebitda = 100 - df['_EV_to_EBITDA'].rank(pct=True) * 100
        df['_Value_Score'] = pd.concat([value_pe, value_ps, value_pb, value_ev_ebitda], axis=1).mean(axis=1)
        
        # Deep Value Factor: Even lower valuations + high FCF yield
        deepvalue_pe = 100 - df['_PriceToEarnings'].rank(pct=True) * 100
        deepvalue_pb = 100 - df['_PriceToBook'].rank(pct=True) * 100
        deepvalue_fcf_yield = df['_FCF_Yield'].rank(pct=True) * 100
        df['_DeepValue_Score'] = pd.concat([deepvalue_pe, deepvalue_pb, deepvalue_fcf_yield], axis=1).mean(axis=1)
        
        # Quality Factor: High margins, returns, and cash flow quality
        quality_roe = df['_ROE'].rank(pct=True) * 100
        quality_roa = df['_ROA'].rank(pct=True) * 100
        quality_roic = df['_ROIC'].rank(pct=True) * 100
        quality_fcf_netincome = df['_FCF_to_NetIncome'].rank(pct=True) * 100
        quality_cf_quality = df['_CashFlow_Quality'].rank(pct=True) * 100
        quality_accruals = 100 - df['_Accruals'].rank(pct=True) * 100  # Lower accruals = higher quality
        df['_Quality_Score'] = pd.concat([quality_roe, quality_roa, quality_roic, quality_fcf_netincome, 
                                          quality_cf_quality, quality_accruals], axis=1).mean(axis=1)
        
        # Profitability Factor: High margins and returns
        prof_gross = df['_GrossProfitMargin'].rank(pct=True) * 100
        prof_op = df['_OperatingProfitMargin'].rank(pct=True) * 100
        prof_net = df['_NetProfitMargin'].rank(pct=True) * 100
        prof_roe = df['_ROE'].rank(pct=True) * 100
        prof_roic = df['_ROIC'].rank(pct=True) * 100
        df['_Profitability_Score'] = pd.concat([prof_gross, prof_op, prof_net, prof_roe, prof_roic], axis=1).mean(axis=1)
        
        # Growth Factor: Forward estimates if available
        if '_PriceToFwdEarnings' in df.columns and df['_PriceToFwdEarnings'].notna().sum() > 0:
            growth_fwdpe = 100 - df['_PriceToFwdEarnings'].rank(pct=True) * 100
            growth_fcf_margin = df['_FCF_Margin'].rank(pct=True) * 100
            growth_opcf_margin = df['_OperatingCashFlow_Margin'].rank(pct=True) * 100
            df['_Growth_Score'] = pd.concat([growth_fwdpe, growth_fcf_margin, growth_opcf_margin], axis=1).mean(axis=1)
        else:
            # Use margins as growth proxy if forward estimates not available
            growth_gross = df['_GrossProfitMargin'].rank(pct=True) * 100
            growth_fcf = df['_FCF_Margin'].rank(pct=True) * 100
            growth_opcf = df['_OperatingCashFlow_Margin'].rank(pct=True) * 100
            df['_Growth_Score'] = pd.concat([growth_gross, growth_fcf, growth_opcf], axis=1).mean(axis=1)
        
        # Return sector df or update self.df
        if by_sector:
            return df
        else:
            self.df = df
            return df
    
    def calculate_z_scores(self, metrics=None, by_sector=False):
        """
        Calculate z-scores for derived metrics (columns starting with '_').
        
        Parameters
        ----------
        metrics : list, optional
            List of metrics to calculate z-scores for. If None, uses all derived metrics (_).
        by_sector : bool
            Calculate z-scores within sectors (sector-relative)
            
        Returns
        -------
        pd.DataFrame
            DataFrame with z-score transformed values (original column names preserved)
        """
        if metrics is None:
            metrics = [c for c in self.df.columns if c.startswith('_')]
        
        df_result = pd.DataFrame(index=self.df.index)
        
        if by_sector and 'Sector' in self.df.columns:
            for metric in metrics:
                if metric in self.df.columns:
                    df_result[metric] = self.df.groupby('Sector')[metric].transform(
                        lambda x: (x - x.mean()) / x.std()
                    )
        else:
            for metric in metrics:
                if metric in self.df.columns:
                    mean = self.df[metric].mean()
                    std = self.df[metric].std()
                    df_result[metric] = (self.df[metric] - mean) / std
        
        return df_result
    
    def calculate_percentile_ranks(self, metrics=None, by_sector=False):
        """
        Calculate percentile ranks for derived metrics (columns starting with '_').
        
        Parameters
        ----------
        metrics : list, optional
            List of metrics to calculate percentile ranks for. If None, uses all derived metrics (_).
        by_sector : bool
            Calculate percentile ranks within sectors (sector-relative)
            
        Returns
        -------
        pd.DataFrame
            DataFrame with percentile rank transformed values (original column names preserved)
        """
        if metrics is None:
            metrics = [c for c in self.df.columns if c.startswith('_')]
        
        df_result = pd.DataFrame(index=self.df.index)
        
        if by_sector and 'Sector' in self.df.columns:
            for metric in metrics:
                if metric in self.df.columns:
                    df_result[metric] = self.df.groupby('Sector')[metric].transform(
                        lambda x: x.rank(pct=True) * 100
                    )
        else:
            for metric in metrics:
                if metric in self.df.columns:
                    df_result[metric] = self.df[metric].rank(pct=True) * 100
        
        return df_result

    def calculate_percentiles(self, metrics=None, by_sector=False):
        """
        Calculate percentile rankings for specified metrics.
        
        Parameters
        ----------
        metrics : list, optional
            List of metrics to rank. If None, ranks key metrics
        by_sector : bool
            Calculate percentiles within sectors
            
        Returns
        -------
        pd.DataFrame
            DataFrame with percentile columns added
        """
        if metrics is None:
            metrics = [
                'PERatio', 'PEGRatio', '_PriceToSales', '_PriceToBook',
                'ROE', 'ROA', '_NetProfitMargin', '_GrossProfitMargin',
                '_CurrentRatio', '_DebtToEquity', '_FCF_Yield',
                'MarketCapitalization', 'Beta'
            ]
            # Filter to available metrics
            metrics = [m for m in metrics if m in self.df.columns]
        
        df_result = self.df.copy()
        
        if by_sector and 'Sector' in self.df.columns:
            for metric in metrics:
                if metric in df_result.columns:
                    df_result[f'{metric}_Percentile'] = df_result.groupby('Sector')[metric].transform(
                        lambda x: x.rank(pct=True) * 100
                    )
        else:
            for metric in metrics:
                if metric in df_result.columns:
                    df_result[f'{metric}_Percentile'] = df_result[metric].rank(pct=True) * 100
        
        return df_result
    
    
    # def peer_comparison(self, ticker=None, sector=None, top_n=None):
    #     """
    #     Compare a company against its peers.
        
    #     Parameters
    #     ----------
    #     ticker : str, optional
    #         Ticker to analyze (uses index if None)
    #     sector : str, optional
    #         Sector for peer group (uses company's sector if None)
    #     top_n : int
    #         Number of top peers to include
            
    #     Returns
    #     -------
    #     pd.DataFrame
    #         Peer comparison table
    #     """
    #     if ticker is not None and ticker in self.df.index:
    #         company = self.df.loc[ticker]
    #         if sector is None and 'Sector' in self.df.columns:
    #             sector = company['Sector']
        
    #     if sector and 'Sector' in self.df.columns:
    #         peers = self.df[self.df['Sector'] == sector].copy()
    #     else:
    #         peers = self.df.copy()
        
    #     # Key comparison metrics
    #     compare_metrics = [
    #         'MarketCapitalization', 'PERatio', 'PEGRatio', 'ForwardPE',
    #         'ROE', 'ROA', '_NetProfitMargin', '_DebtToEquity',
    #         '_CurrentRatio', '_FCF_Yield', 'DividendYield'
    #     ]
    #     compare_metrics = [m for m in compare_metrics if m in peers.columns]
        
    #     # Calculate z-scores for ranking
    #     for metric in compare_metrics:
    #         peers[f'{metric}_zscore'] = stats.zscore(peers[metric].fillna(peers[metric].median()))
        
    #     # Composite ranking (simple average of z-scores)
    #     zscore_cols = [f'{m}_zscore' for m in compare_metrics]
    #     peers['Composite_Rank'] = peers[zscore_cols].mean(axis=1)
        
    #     # Sort by composite rank
    #     result = peers.sort_values('Composite_Rank', ascending=False)
        
    #     if top_n is not None:
    #         result = result.head(top_n)
        
    #     # Clean up - keep relevant columns
    #     keep_cols = ['Name', 'Sector', 'MarketCapitalization'] + compare_metrics + ['Composite_Rank']
    #     keep_cols = [c for c in keep_cols if c in result.columns]
        
    #     return result[keep_cols]
    
    
    # def sector_analysis(self, df=None):
    #     """
    #     Analyze metrics by sector.
        
    #     Parameters
    #     ----------
    #     df : pd.DataFrame, optional
    #         DataFrame with required columns. If not provided, uses self.df.
    #         Required column: Sector
        
    #     Returns
    #     -------
    #     pd.DataFrame
    #         Sector-level statistics
    #     """
    #     # Use provided df or fall back to self.df
    #     data_source = df if df is not None else self.df
        
    #     if 'Sector' not in data_source.columns:
    #         print("⚠️ Sector information not available")
    #         print(f"   Hint: Pass the full dataframe with Sector data as df= parameter")
    #         print(f"   Example: fa.sector_analysis(df=df_fun)")
    #         return pd.DataFrame()
        
    #     metrics = [
    #         'PERatio', 'PEGRatio', 'ROE', 'ROA', '_NetProfitMargin',
    #         '_DebtToEquity', '_CurrentRatio', '_FCF_Yield', 'DividendYield',
    #         'MarketCapitalization'
    #     ]
    #     metrics = [m for m in metrics if m in data_source.columns]
        
    #     if not metrics:
    #         print("⚠️ No analyzable metrics found in the data")
    #         return pd.DataFrame()
        
    #     sector_stats = data_source.groupby('Sector')[metrics].agg(['mean', 'median', 'std', 'count'])
        
    #     return sector_stats
    
    def rank_by_metric(self, metric, ascending=True, top_n=20, df=None):
        """
        Rank companies by a specific metric.
        
        Parameters
        ----------
        metric : str
            Metric to rank by
        ascending : bool
            Sort order (True for lowest first)
        top_n : int
            Number of results to return
        df : pd.DataFrame, optional
            DataFrame with required columns. If not provided, uses self.df.
            
        Returns
        -------
        pd.DataFrame
            Ranked companies
        """
        # Use provided df or fall back to self.df
        data_source = df if df is not None else self.df
        
        if metric not in data_source.columns:
            print(f"⚠️ Metric '{metric}' not found in data")
            print(f"   Available columns: {list(data_source.columns)[:20]}... (showing first 20)")
            print(f"\n   Hint: Pass the full dataframe as df= parameter")
            print(f"   Example: fa.rank_by_metric(metric='{metric}', ascending={ascending}, top_n={top_n}, df=df_fun)")
            return pd.DataFrame()
        
        ranked = data_source.sort_values(metric, ascending=ascending).head(top_n)
        
        display_cols = ['Name', 'Sector', 'MarketCapitalization', metric]
        display_cols = [c for c in display_cols if c in ranked.columns]
        
        return ranked[display_cols]
    
        

    def sector_pe_contribution(self, official_pe=None, df=None):
        """
        Calculate aggregate sector-level P/E as 1/earnings yield from aggregate sector LTM earnings.
        Uses columns: ltm_epsActual, dt_epsActual, commonStockSharesOutstanding, MarketCapitalization.
        Returns a DataFrame with sector, sector weight, aggregate sector P/E, contribution to index P/E, and total index P/E.
        
        Parameters
        ----------
        official_pe : float, optional
            If provided, rescales all sector P/Es and contributions so that the Index_PE matches this official value.
        df : pd.DataFrame, optional
            DataFrame with required columns. If not provided, uses self.df (which may not have all required columns).
            Required columns: Sector, ltm_epsActual, commonStockSharesOutstanding, MarketCapitalization, Weight
        """
        # Use provided df or fall back to self.df
        data_source = df if df is not None else self.df
        
        required_cols = ['Sector', 'ltm_epsActual', 'commonStockSharesOutstanding', 'MarketCapitalization']
        missing_cols = [col for col in required_cols if col not in data_source.columns]
        
        if missing_cols:
            print(f"⚠️ Error: Missing required columns: {missing_cols}")
            print(f"   Available columns: {list(data_source.columns)[:20]}... (showing first 20)")
            print(f"\n   Hint: Pass the full dataframe with ltm/sector data as df= parameter")
            print(f"   Example: fa.sector_pe_contribution(official_pe=None, df=df_fun)")
            return pd.DataFrame()

        # Make a copy and ensure numeric columns
        df_clean = data_source.copy()
        
        # Convert numeric columns that might be strings
        numeric_cols = ['ltm_epsActual', 'commonStockSharesOutstanding', 'MarketCapitalization']
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Filter for rows with required data
        df_clean = df_clean[
            df_clean['Sector'].notna() &
            df_clean['ltm_epsActual'].notna() &
            df_clean['commonStockSharesOutstanding'].notna() &
            df_clean['MarketCapitalization'].notna()
        ].copy()

        # Calculate aggregate LTM earnings for each row
        df_clean['Earnings'] = df_clean['ltm_epsActual'] * df_clean['commonStockSharesOutstanding']

        # Check if Weight column exists, if not calculate it from market cap
        if 'Weight' in df_clean.columns:
            df_clean['Weight'] = pd.to_numeric(df_clean['Weight'], errors='coerce')
            weight_col = 'Weight'
        else:
            # Calculate weight from market cap
            total_mcap = df_clean['MarketCapitalization'].sum()
            df_clean['Calculated_Weight'] = df_clean['MarketCapitalization'] / total_mcap
            weight_col = 'Calculated_Weight'

        # Aggregate by sector
        agg_dict = {
            'MarketCapitalization': 'sum',
            'Earnings': 'sum',
            weight_col: 'sum',
            'commonStockSharesOutstanding': 'sum',
            'ltm_epsActual': 'count'
        }
        
        sector_stats = df_clean.groupby('Sector').agg(agg_dict).rename(columns={
            'MarketCapitalization': 'Sector_MarketCap',
            'Earnings': 'Sector_Earnings',
            weight_col: 'Sector_Weight',
            'commonStockSharesOutstanding': 'Sector_SharesOutstanding',
            'ltm_epsActual': 'Count'
        }).reset_index()

        # Calculate sector earnings yield and sector PE
        sector_stats['Sector_Earnings_Yield'] = sector_stats['Sector_Earnings'] / sector_stats['Sector_MarketCap']
        sector_stats['Sector_PE'] = 1 / sector_stats['Sector_Earnings_Yield']

        # Index aggregates
        index_marketcap = sector_stats['Sector_MarketCap'].sum()
        index_earnings = sector_stats['Sector_Earnings'].sum()
        calculated_index_pe = 1 / (index_earnings / index_marketcap) if index_marketcap and index_earnings else np.nan

        # Rescale if official P/E is provided
        if official_pe is not None and not np.isnan(calculated_index_pe) and calculated_index_pe != 0:
            scale_factor = official_pe / calculated_index_pe
            sector_stats['Sector_PE'] = sector_stats['Sector_PE'] * scale_factor
            index_pe = official_pe
            print(f'Calculated index P/E: {calculated_index_pe:.2f} -> Rescaled to official P/E: {index_pe:.2f}')
        else:
            index_pe = calculated_index_pe
            print(f'The index P/E ratio is: {index_pe:.2f}')

        # Calculate PE contribution using earnings yield approach
        # PE_Contribution = (sector_earnings / index_earnings) * index_PE
        sector_stats['Earnings_Weight'] = sector_stats['Sector_Earnings'] / index_earnings
        sector_stats['PE_Contribution'] = sector_stats['Earnings_Weight'] * index_pe
        sector_stats['PE_Contribution_Pct'] = (sector_stats['PE_Contribution'] / index_pe) * 100
        
        # Calculate sector-level earnings yield and EPS
        sector_stats['Sector_Earnings_Yield'] = (sector_stats['Sector_Earnings'] / sector_stats['Sector_MarketCap']) * 100
        sector_stats['Sector_EPS'] = sector_stats['Sector_Earnings'] / sector_stats['Sector_SharesOutstanding']
        sector_stats['Sector_Earnings_Pct'] = (sector_stats['Sector_Earnings'] / index_earnings) * 100
        
        # Calculate index-level values for summary row
        index_earnings_yield = (index_earnings / index_marketcap) * 100
        index_eps = index_earnings / df_clean['commonStockSharesOutstanding'].sum()

        # Format output
        result_df = sector_stats[['Sector', 'Count', 'Sector_Weight', 'Sector_PE', 'Sector_Earnings_Yield',
                                   'Sector_EPS', 'Sector_Earnings_Pct', 'PE_Contribution', 'PE_Contribution_Pct']].copy()
        
        # Add summary row with totals for Sector_Weight and PE_Contribution
        summary_row = pd.DataFrame({
            'Sector': ['TOTAL'],
            'Count': result_df['Count'].sum(),
            'Sector_Weight': [result_df['Sector_Weight'].sum()],
            'Sector_PE': [None],
            'Sector_Earnings_Yield': [index_earnings_yield],
            'Sector_EPS': [index_eps],
            'Sector_Earnings_Pct': [result_df['Sector_Earnings_Pct'].sum()],
            'PE_Contribution': [result_df['PE_Contribution'].sum()],
            'PE_Contribution_Pct': [result_df['PE_Contribution_Pct'].sum()],
        })
        
        result_df = pd.concat([result_df, summary_row], ignore_index=True)
        
        return result_df
    







    
    # def calculate_piotroski_fscore(self, use_lagged=True):
    #     """
    #     Calculate Piotroski F-Score (0-9). Higher = better quality value stock.
    #     """
    #     df = self.df
    #     score = pd.DataFrame(index=df.index)
        
    #     df['ROA'] = df['netIncome'] / df['totalAssets'].replace(0, np.nan)
    #     score['F_ROA'] = (df['ROA'] > 0).astype(int)
    #     score['F_CFO'] = (df['totalCashFromOperatingActivities'] > 0).astype(int)
    #     score['F_ACCRUAL'] = (df['totalCashFromOperatingActivities'] > df['netIncome']).astype(int)
        
    #     if use_lagged:
    #         leverage = df['longTermDebt'] / df['totalAssets'].replace(0, np.nan)
    #         leverage_lag = df['longTermDebt_lag1'] / df['totalAssets_lag1'].replace(0, np.nan)
    #         score['F_DELTA_ROA'] = (df['ROA'] > df['ROA_lag1']).astype(int)
    #         score['F_DELTA_LEVER'] = (leverage < leverage_lag).astype(int)
    #         score['F_DELTA_LIQUID'] = (df['_CurrentRatio'] > df['_CurrentRatio_lag1']).astype(int)
    #         score['F_EQ_OFFER'] = (df['commonStockSharesOutstanding'] <= df['commonStockSharesOutstanding_lag1']).astype(int)
    #         score['F_DELTA_MARGIN'] = (df['_GrossProfitMargin'] > df['_GrossProfitMargin_lag1']).astype(int)
    #         score['F_DELTA_TURN'] = (df['_AssetTurnover'] > df['_AssetTurnover_lag1']).astype(int)
        
    #     score['Piotroski_FScore'] = score.filter(like='F_').sum(axis=1)
    #     return score
    
    
    # def calculate_beneish_mscore(self):
    #     """
    #     Calculate Beneish M-Score for earnings manipulation detection.
    #     M > -1.78 = HIGH manipulation risk, M < -2.22 = low risk
    #     """
    #     df = self.df
    #     result = pd.DataFrame(index=df.index)
        
    #     receivables_sales = df['netReceivables'] / df['totalRevenue'].replace(0, np.nan)
    #     receivables_sales_lag = df['netReceivables_lag1'] / df['totalRevenue_lag1'].replace(0, np.nan)
    #     result['DSRI'] = receivables_sales / receivables_sales_lag.replace(0, np.nan)
    #     result['GMI'] = df['_GrossProfitMargin_lag1'] / df['_GrossProfitMargin'].replace(0, np.nan)
        
    #     nca = df['totalAssets'] - df['totalCurrentAssets'] - df['propertyPlantEquipment']
    #     nca_lag = df['totalAssets_lag1'] - df['totalCurrentAssets_lag1'] - df['propertyPlantEquipment_lag1']
    #     result['AQI'] = (nca / df['totalAssets'].replace(0, np.nan)) / (nca_lag / df['totalAssets_lag1'].replace(0, np.nan))
        
    #     result['SGI'] = df['totalRevenue'] / df['totalRevenue_lag1'].replace(0, np.nan)
        
    #     depr = df['propertyPlantEquipment'] / df['totalAssets'].replace(0, np.nan)
    #     depr_lag = df['propertyPlantEquipment_lag1'] / df['totalAssets_lag1'].replace(0, np.nan)
    #     result['DEPI'] = depr_lag / depr.replace(0, np.nan)
        
    #     sga_ratio = df['totalOperatingExpenses'] / df['totalRevenue'].replace(0, np.nan)
    #     sga_ratio_lag = df['totalOperatingExpenses_lag1'] / df['totalRevenue_lag1'].replace(0, np.nan)
    #     result['SGAI'] = sga_ratio / sga_ratio_lag.replace(0, np.nan)
        
    #     leverage = df['totalLiab'] / df['totalAssets'].replace(0, np.nan)
    #     leverage_lag = df['totalLiab_lag1'] / df['totalAssets_lag1'].replace(0, np.nan)
    #     result['LVGI'] = leverage / leverage_lag.replace(0, np.nan)
        
    #     result['TATA'] = (df['netIncome'] - df['totalCashFromOperatingActivities']) / df['totalAssets'].replace(0, np.nan)
        
    #     result['Beneish_MScore'] = (
    #         -4.84 + 0.920 * result['DSRI'] + 0.528 * result['GMI'] + 0.404 * result['AQI'] +
    #         0.892 * result['SGI'] + 0.115 * result['DEPI'] - 0.172 * result['SGAI'] +
    #         4.679 * result['TATA'] - 0.327 * result['LVGI']
    #     )
    #     result['Beneish_Risk'] = result['Beneish_MScore'].apply(
    #         lambda x: 'HIGH' if x > -1.78 else ('MODERATE' if x > -2.22 else 'LOW')
    #     )
    #     return result
    
    
    # def calculate_altman_zscore(self, model='services'):
    #     """
    #     Calculate Altman Z-Score for bankruptcy prediction.
    #     model: 'manufacturing', 'private', or 'services'
    #     """
    #     df = self.df
    #     result = pd.DataFrame(index=df.index)
        
    #     result['X1'] = df['netWorkingCapital'] / df['totalAssets'].replace(0, np.nan)
    #     result['X2'] = df['retainedEarnings'] / df['totalAssets'].replace(0, np.nan)
    #     result['X3'] = df['ebit'] / df['totalAssets'].replace(0, np.nan)
    #     result['X4'] = df['MarketCapitalization'] / df['totalLiab'].replace(0, np.nan)
    #     result['X5'] = df['totalRevenue'] / df['totalAssets'].replace(0, np.nan)
        
    #     if model == 'manufacturing':
    #         result['Altman_ZScore'] = 1.2*result['X1'] + 1.4*result['X2'] + 3.3*result['X3'] + 0.6*result['X4'] + 1.0*result['X5']
    #         result['Altman_Zone'] = result['Altman_ZScore'].apply(lambda x: 'Safe' if x > 3.0 else ('Grey' if x > 2.0 else 'Distress'))
    #     elif model == 'private':
    #         result['Altman_ZScore'] = 0.717*result['X1'] + 0.847*result['X2'] + 3.107*result['X3'] + 0.420*result['X4'] + 0.998*result['X5']
    #         result['Altman_Zone'] = result['Altman_ZScore'].apply(lambda x: 'Safe' if x > 2.9 else ('Grey' if x > 1.23 else 'Distress'))
    #     else:
    #         result['Altman_ZScore'] = 6.56*result['X1'] + 3.26*result['X2'] + 6.72*result['X3'] + 1.05*result['X4']
    #         result['Altman_Zone'] = result['Altman_ZScore'].apply(lambda x: 'Safe' if x > 2.6 else ('Grey' if x > 1.1 else 'Distress'))
        
    #     return result
    
    
    # def calculate_ohlson_oscore(self):
    #     """
    #     Calculate Ohlson O-Score for bankruptcy probability.
    #     Higher O-Score = higher bankruptcy risk.
    #     """
    #     df = self.df
    #     result = pd.DataFrame(index=df.index)
        
    #     result['SIZE'] = np.log(df['totalAssets'].replace(0, np.nan))
    #     result['TLTA'] = df['totalLiab'] / df['totalAssets'].replace(0, np.nan)
    #     result['WCTA'] = df['netWorkingCapital'] / df['totalAssets'].replace(0, np.nan)
    #     result['CLCA'] = df['totalCurrentLiabilities'] / df['totalCurrentAssets'].replace(0, np.nan)
    #     result['NITA'] = df['netIncome'] / df['totalAssets'].replace(0, np.nan)
    #     result['FUTL'] = df['totalCashFromOperatingActivities'] / df['totalLiab'].replace(0, np.nan)
    #     result['OENEG'] = (df['totalLiab'] > df['totalAssets']).astype(int)
    #     result['CHIN'] = ((df['netIncome'] - df['netIncome_lag1']) / 
    #                       ((df['netIncome'].abs() + df['netIncome_lag1'].abs()) / 2).replace(0, np.nan))
        
    #     result['Ohlson_OScore'] = (
    #         -1.32 - 0.407*result['SIZE'] + 6.03*result['TLTA'] - 1.43*result['WCTA'] +
    #         0.0757*result['CLCA'] - 2.37*result['NITA'] - 1.83*result['FUTL'] +
    #         0.285*result['OENEG'] - 0.521*result['CHIN']
    #     )
    #     result['Ohlson_Probability'] = 1 / (1 + np.exp(-result['Ohlson_OScore']))
    #     result['Ohlson_Risk'] = result['Ohlson_Probability'].apply(
    #         lambda x: 'HIGH' if x > 0.5 else ('MODERATE' if x > 0.2 else 'LOW')
    #     )
    #     return result
    
    
    # def calculate_sloan_accruals(self):
    #     """
    #     Calculate Sloan Accruals Ratio. High accruals (>10%) = low earnings quality.
    #     """
    #     df = self.df
    #     result = pd.DataFrame(index=df.index)
        
    #     result['Accruals'] = (df['netIncome'] - df['totalCashFromOperatingActivities']) / df['totalAssets'].replace(0, np.nan)
    #     result['Accruals_Pct'] = result['Accruals'] * 100
    #     result['Earnings_Quality'] = result['Accruals_Pct'].apply(
    #         lambda x: 'HIGH' if abs(x) < 5 else ('MODERATE' if abs(x) < 10 else 'LOW')
    #     )
    #     return result
    
    
    # def calculate_mohanram_gscore(self):
    #     """
    #     Calculate Mohanram G-Score for growth stock quality (0-8).
    #     Higher = better quality growth stock.
    #     """
    #     df = self.df
    #     score = pd.DataFrame(index=df.index)
        
    #     roa = df['netIncome'] / df['totalAssets'].replace(0, np.nan)
    #     accruals = abs((df['netIncome'] - df['totalCashFromOperatingActivities']) / df['totalAssets'].replace(0, np.nan))
    #     sales_growth = (df['totalRevenue'] - df['totalRevenue_lag1']) / df['totalRevenue_lag1'].replace(0, np.nan)
    #     rd_intensity = df['researchDevelopment'] / df['totalRevenue'].replace(0, np.nan)
    #     opex_ratio = df['totalOperatingExpenses'] / df['totalRevenue'].replace(0, np.nan)
    #     capex_ratio = abs(df['capitalExpenditures']) / df['totalAssets'].replace(0, np.nan)
        
    #     score['G_ROA'] = (roa > roa.median()).astype(int)
    #     score['G_CFO'] = (df['totalCashFromOperatingActivities'] > df['netIncome']).astype(int)
    #     score['G_ACCRUAL'] = (accruals < accruals.median()).astype(int)
    #     score['G_MARGIN'] = (df['_NetProfitMargin'] >= df['_NetProfitMargin_lag1']).astype(int)
    #     score['G_RD'] = (rd_intensity > rd_intensity.median()).astype(int)
    #     score['G_SALES_GROWTH'] = (sales_growth > sales_growth.median()).astype(int)
    #     score['G_ADVERTISING'] = (opex_ratio > opex_ratio.median()).astype(int)
    #     score['G_CAPEX'] = (capex_ratio < capex_ratio.median()).astype(int)
        
    #     score['Mohanram_GScore'] = score.filter(like='G_').sum(axis=1)
    #     score['Growth_Quality'] = score['Mohanram_GScore'].apply(
    #         lambda x: 'Excellent' if x >= 7 else ('Good' if x >= 5 else ('Fair' if x >= 3 else 'Weak'))
    #     )
    #     return score
    
    
    # def calculate_gross_profitability(self):
    #     """
    #     Calculate Novy-Marx Gross Profitability Premium.
    #     GP = (Revenue - COGS) / Total Assets
    #     """
    #     df = self.df
    #     result = pd.DataFrame(index=df.index)
        
    #     result['Gross_Profitability'] = df['grossProfit'] / df['totalAssets'].replace(0, np.nan)
    #     result['GP_Quality'] = result['Gross_Profitability'].apply(
    #         lambda x: 'Strong' if x > 0.25 else ('Moderate' if x > 0.15 else 'Weak')
    #     )
    #     return result
    
    
    # def calculate_dividend_safety(self):
    #     """
    #     Calculate dividend safety metrics.
    #     """
    #     df = self.df
    #     result = pd.DataFrame(index=df.index)
        
    #     total_dividends = df['DividendYield'] * df['MarketCapitalization']
    #     result['FCF_Coverage'] = df['freeCashFlow'] / total_dividends.replace(0, np.nan)
    #     result['NetDebt_EBITDA'] = (df['longTermDebt'] - df['cash']) / df['ebitda'].replace(0, np.nan)
        
    #     dps = (df['DividendYield'] * df['MarketCapitalization']) / df['commonStockSharesOutstanding'].replace(0, np.nan)
    #     result['Payout_Ratio'] = (dps / df['EarningsShare'].replace(0, np.nan)) * 100
        
    #     result['Dividend_Safety'] = (
    #         (result['FCF_Coverage'] > 1.5).astype(int) +
    #         (result['NetDebt_EBITDA'] < 2.0).astype(int) +
    #         (result['Payout_Ratio'] < 50).astype(int)
    #     )
    #     result['Safety_Rating'] = result['Dividend_Safety'].apply(
    #         lambda x: 'Safe' if x == 3 else ('Moderate' if x == 2 else ('Caution' if x == 1 else 'Risky'))
    #     )
    #     return result
    
    
    # def calculate_sustainable_growth_rate(self):
    #     """
    #     Calculate Sustainable Growth Rate (SGR).
    #     SGR = ROE × (1 - Payout Ratio)
    #     """
    #     df = self.df
    #     result = pd.DataFrame(index=df.index)
        
    #     roe = df['netIncome'] / df['totalStockholderEquity'].replace(0, np.nan)
    #     dps = (df['DividendYield'] * df['MarketCapitalization']) / df['commonStockSharesOutstanding'].replace(0, np.nan)
    #     retention = 1 - (dps / df['EarningsShare'].replace(0, np.nan))
        
    #     result['SGR'] = roe * retention * 100
    #     result['Growth_Capacity'] = result['SGR'].apply(
    #         lambda x: 'Strong' if x > 10 else ('Moderate' if x > 5 else 'Weak')
    #     )
    #     return result
    
    
    # def quality_summary(self, include_lagged=False):
    #     """
    #     Generate comprehensive quality summary combining all metrics.
    #     """
    #     summary = pd.DataFrame(index=self.df.index)
        
    #     fscore = self.calculate_piotroski_fscore(use_lagged=include_lagged)
    #     summary['Piotroski_FScore'] = fscore['Piotroski_FScore']
        
    #     if include_lagged:
    #         mscore = self.calculate_beneish_mscore()
    #         summary['Beneish_MScore'] = mscore['Beneish_MScore']
    #         summary['Beneish_Risk'] = mscore['Beneish_Risk']
        
    #     zscore = self.calculate_altman_zscore()
    #     summary['Altman_ZScore'] = zscore['Altman_ZScore']
    #     summary['Altman_Zone'] = zscore['Altman_Zone']
        
    #     accruals = self.calculate_sloan_accruals()
    #     summary['Accruals_Pct'] = accruals['Accruals_Pct']
    #     summary['Earnings_Quality'] = accruals['Earnings_Quality']
        
    #     gp = self.calculate_gross_profitability()
    #     summary['Gross_Profitability'] = gp['Gross_Profitability']
        
    #     summary['Name'] = self.df['Name']
    #     summary['Sector'] = self.df['Sector']
        
    #     return summary
