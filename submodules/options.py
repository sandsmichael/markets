"""
Professional Options Analysis Toolkit
Expert-level tools for options contract analysis, strategy evaluation, and risk management
"""

from typing import Union, List, Optional, Dict, Tuple, Any
from enum import Enum
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Optional dependencies
try:
    import QuantLib as ql
    QUANTLIB_AVAILABLE = True
except ImportError:
    QUANTLIB_AVAILABLE = False


# =============================================================================
# CONSTANTS AND ENUMS
# =============================================================================

RISK_FREE_RATE = 0.05
DAYS_PER_YEAR = 252

class OptionType(Enum):
    """Option type enumeration"""
    CALL = 'call' if not QUANTLIB_AVAILABLE else ql.Option.Call
    PUT = 'put' if not QUANTLIB_AVAILABLE else ql.Option.Put

class OptionExposure(Enum):
    """Option position type"""
    LONG = 'long'
    SHORT = 'short'


# =============================================================================
# OPTIONS CHAIN ANALYZER
# =============================================================================

class OptionsAnalyzer:
    """
    Comprehensive options chain analysis and strategy selection.
    
    Provides tools for:
    - Implied move calculations
    - Strategy screening (spreads, straddles, etc.)
    - Risk/reward analysis
    - Probability calculations
    - Greeks analysis
    
    Parameters
    ----------
    chain : pd.DataFrame
        Options chain with columns: strike, expiration, dte, bid, ask, 
        impliedVolatility, delta, gamma, theta, vega
    underlying_price : float
        Current price of underlying asset
    
    Examples
    --------
    >>> analyzer = OptionsAnalyzer(chain_df, underlying_price=450.50)
    >>> implied_move = analyzer.get_implied_move()
    >>> spreads = analyzer.select_bull_put_spreads(min_credit=1.0, max_loss=5.0)
    >>> iron_condors = analyzer.select_iron_condors(target_credit=2.0)
    """
    
    def __init__(self, chain: pd.DataFrame, underlying_price: float):
        self.chain = chain.copy()
        self.underlying_price = underlying_price
        self._process_chain()
    
    def _process_chain(self):
        """Process and validate chain data"""
        # Ensure required columns exist
        required_cols = ['strike', 'bid', 'ask']
        missing = [col for col in required_cols if col not in self.chain.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Calculate mid price if not present
        if 'mid' not in self.chain.columns:
            self.chain['mid'] = (self.chain['bid'] + self.chain['ask']) / 2
        
        # Calculate moneyness
        self.chain['moneyness'] = self.chain['strike'] / self.underlying_price
        self.chain['otm_amount'] = abs(self.chain['strike'] - self.underlying_price)
    
    # =========================================================================
    # IMPLIED MOVE CALCULATIONS
    # =========================================================================
    
    def get_implied_move(self, expiration: Optional[str] = None, 
                        method: str = 'straddle') -> Dict[str, float]:
        """
        Calculate implied move based on options prices.
        
        Parameters
        ----------
        expiration : str, optional
            Specific expiration date to analyze (uses nearest if None)
        method : str
            Calculation method: 'straddle' or 'atm_volatility'
            
        Returns
        -------
        dict
            Implied move statistics with dollar and percentage values
            
        Examples
        --------
        >>> move = analyzer.get_implied_move()
        >>> print(f"Implied move: ±${move['move_dollar']:.2f} (±{move['move_pct']:.1f}%)")
        """
        if expiration:
            chain_subset = self.chain[self.chain['expiration'] == expiration]
        else:
            # Use nearest expiration
            chain_subset = self.chain[self.chain['expiration'] == self.chain['expiration'].min()]
        
        if method == 'straddle':
            # Find ATM strike
            atm_strike = chain_subset.iloc[(chain_subset['strike'] - self.underlying_price).abs().argsort()[:1]]['strike'].values[0]
            
            # Get ATM call and put prices
            atm_call = chain_subset[(chain_subset['strike'] == atm_strike) & 
                                   (chain_subset['type'] == 'call')]['mid'].values
            atm_put = chain_subset[(chain_subset['strike'] == atm_strike) & 
                                  (chain_subset['type'] == 'put')]['mid'].values
            
            if len(atm_call) > 0 and len(atm_put) > 0:
                straddle_price = atm_call[0] + atm_put[0]
                move_dollar = straddle_price
                move_pct = (move_dollar / self.underlying_price) * 100
            else:
                return {'error': 'ATM options not found'}
        
        elif method == 'atm_volatility':
            # Use ATM IV and time to expiration
            atm_iv = chain_subset.iloc[(chain_subset['strike'] - self.underlying_price).abs().argsort()[:1]]['impliedVolatility'].values
            
            if len(atm_iv) > 0 and 'dte' in chain_subset.columns:
                dte = chain_subset['dte'].iloc[0]
                move_pct = atm_iv[0] * np.sqrt(dte / 365)
                move_dollar = self.underlying_price * (move_pct / 100)
            else:
                return {'error': 'Required data not found'}
        
        return {
            'move_dollar': round(move_dollar, 2),
            'move_pct': round(move_pct, 2),
            'upper_target': round(self.underlying_price + move_dollar, 2),
            'lower_target': round(self.underlying_price - move_dollar, 2),
            'method': method
        }
    
    # =========================================================================
    # CREDIT SPREAD SELECTION
    # =========================================================================
    
    def select_bull_put_spreads(self, min_credit: float = 0.5, max_loss: float = None,
                                min_delta: float = 0.15, max_delta: float = 0.35,
                                width: int = 5, expiration: Optional[str] = None) -> pd.DataFrame:
        """
        Screen for optimal bull put credit spreads.
        
        Parameters
        ----------
        min_credit : float
            Minimum credit to receive per spread
        max_loss : float, optional
            Maximum loss per spread
        min_delta : float
            Minimum short put delta (default: 0.15)
        max_delta : float
            Maximum short put delta (default: 0.35)
        width : int
            Strike width of spread (default: 5)
        expiration : str, optional
            Target expiration date
            
        Returns
        -------
        pd.DataFrame
            Ranked spread opportunities with risk metrics
            
        Examples
        --------
        >>> spreads = analyzer.select_bull_put_spreads(
        ...     min_credit=1.0,
        ...     max_loss=4.0,
        ...     width=5
        ... )
        >>> print(spreads[['short_strike', 'credit', 'max_loss', 'pl_ratio', 'prob_profit']])
        """
        # Filter puts
        puts = self.chain[self.chain['type'] == 'put'].copy()
        
        if expiration:
            puts = puts[puts['expiration'] == expiration]
        
        # Filter by delta if available
        if 'delta' in puts.columns:
            short_puts = puts[(puts['delta'].abs() >= min_delta) & 
                            (puts['delta'].abs() <= max_delta)]
        else:
            # Filter by moneyness
            short_puts = puts[(puts['moneyness'] >= 0.95) & (puts['moneyness'] <= 1.0)]
        
        spreads = []
        
        for _, short_put in short_puts.iterrows():
            short_strike = short_put['strike']
            long_strike = short_strike - width
            
            # Find corresponding long put
            long_put = puts[puts['strike'] == long_strike]
            
            if long_put.empty:
                continue
            
            long_put = long_put.iloc[0]
            
            # Calculate spread metrics
            credit = short_put['bid'] - long_put['ask']
            max_risk = width - credit
            pl_ratio = credit / max_risk if max_risk > 0 else 0
            breakeven = short_strike - credit
            prob_profit = self._calculate_prob_profit(breakeven, short_put.get('delta', 0.3))
            
            # Apply filters
            if credit < min_credit:
                continue
            if max_loss and max_risk > max_loss:
                continue
            
            spreads.append({
                'short_strike': short_strike,
                'long_strike': long_strike,
                'width': width,
                'credit': round(credit, 2),
                'max_loss': round(max_risk, 2),
                'pl_ratio': round(pl_ratio, 3),
                'breakeven': round(breakeven, 2),
                'prob_profit': round(prob_profit * 100, 1),
                'short_delta': short_put.get('delta', np.nan),
                'dte': short_put.get('dte', np.nan),
                'expiration': short_put.get('expiration', None)
            })
        
        if not spreads:
            return pd.DataFrame()
        
        result = pd.DataFrame(spreads)
        # Sort by P/L ratio descending
        result = result.sort_values('pl_ratio', ascending=False)
        
        return result
    
    def select_bear_call_spreads(self, min_credit: float = 0.5, max_loss: float = None,
                                 min_delta: float = 0.15, max_delta: float = 0.35,
                                 width: int = 5, expiration: Optional[str] = None) -> pd.DataFrame:
        """Screen for optimal bear call credit spreads."""
        # Filter calls
        calls = self.chain[self.chain['type'] == 'call'].copy()
        
        if expiration:
            calls = calls[calls['expiration'] == expiration]
        
        # Filter by delta if available
        if 'delta' in calls.columns:
            short_calls = calls[(calls['delta'].abs() >= min_delta) & 
                              (calls['delta'].abs() <= max_delta)]
        else:
            short_calls = calls[(calls['moneyness'] >= 1.0) & (calls['moneyness'] <= 1.05)]
        
        spreads = []
        
        for _, short_call in short_calls.iterrows():
            short_strike = short_call['strike']
            long_strike = short_strike + width
            
            long_call = calls[calls['strike'] == long_strike]
            
            if long_call.empty:
                continue
            
            long_call = long_call.iloc[0]
            
            credit = short_call['bid'] - long_call['ask']
            max_risk = width - credit
            pl_ratio = credit / max_risk if max_risk > 0 else 0
            breakeven = short_strike + credit
            prob_profit = self._calculate_prob_profit(breakeven, short_call.get('delta', -0.3))
            
            if credit < min_credit:
                continue
            if max_loss and max_risk > max_loss:
                continue
            
            spreads.append({
                'short_strike': short_strike,
                'long_strike': long_strike,
                'width': width,
                'credit': round(credit, 2),
                'max_loss': round(max_risk, 2),
                'pl_ratio': round(pl_ratio, 3),
                'breakeven': round(breakeven, 2),
                'prob_profit': round(prob_profit * 100, 1),
                'short_delta': short_call.get('delta', np.nan),
                'dte': short_call.get('dte', np.nan),
                'expiration': short_call.get('expiration', None)
            })
        
        if not spreads:
            return pd.DataFrame()
        
        result = pd.DataFrame(spreads)
        result = result.sort_values('pl_ratio', ascending=False)
        
        return result
    
    # =========================================================================
    # IRON CONDOR SELECTION
    # =========================================================================
    
    def select_iron_condors(self, target_credit: float = 2.0, width: int = 5,
                           put_delta: float = 0.20, call_delta: float = 0.20,
                           expiration: Optional[str] = None) -> pd.DataFrame:
        """
        Screen for iron condor opportunities.
        
        Parameters
        ----------
        target_credit : float
            Minimum total credit target
        width : int
            Width of each spread
        put_delta : float
            Target delta for short put
        call_delta : float
            Target delta for short call
        expiration : str, optional
            Target expiration
            
        Returns
        -------
        pd.DataFrame
            Iron condor opportunities
        """
        bull_puts = self.select_bull_put_spreads(
            min_credit=target_credit/2,
            min_delta=put_delta*0.8,
            max_delta=put_delta*1.2,
            width=width,
            expiration=expiration
        )
        
        bear_calls = self.select_bear_call_spreads(
            min_credit=target_credit/2,
            min_delta=call_delta*0.8,
            max_delta=call_delta*1.2,
            width=width,
            expiration=expiration
        )
        
        if bull_puts.empty or bear_calls.empty:
            return pd.DataFrame()
        
        condors = []
        
        for _, put_spread in bull_puts.head(10).iterrows():
            for _, call_spread in bear_calls.head(10).iterrows():
                if put_spread['expiration'] != call_spread['expiration']:
                    continue
                
                total_credit = put_spread['credit'] + call_spread['credit']
                max_loss = max(put_spread['max_loss'], call_spread['max_loss'])
                
                if total_credit < target_credit:
                    continue
                
                condors.append({
                    'put_short_strike': put_spread['short_strike'],
                    'put_long_strike': put_spread['long_strike'],
                    'call_short_strike': call_spread['short_strike'],
                    'call_long_strike': call_spread['long_strike'],
                    'total_credit': round(total_credit, 2),
                    'max_loss': round(max_loss, 2),
                    'pl_ratio': round(total_credit / max_loss, 3),
                    'expiration': put_spread['expiration'],
                    'dte': put_spread['dte']
                })
        
        if not condors:
            return pd.DataFrame()
        
        result = pd.DataFrame(condors)
        result = result.sort_values('pl_ratio', ascending=False)
        
        return result
    
    # =========================================================================
    # STRADDLE/STRANGLE SELECTION
    # =========================================================================
    
    def select_strangles(self, min_credit: float = 2.0, put_delta: float = 0.25,
                        call_delta: float = 0.25, expiration: Optional[str] = None) -> pd.DataFrame:
        """
        Screen for short strangle opportunities.
        
        Parameters
        ----------
        min_credit : float
            Minimum total credit
        put_delta : float
            Target put delta
        call_delta : float
            Target call delta
        expiration : str, optional
            Target expiration
            
        Returns
        -------
        pd.DataFrame
            Strangle opportunities
        """
        chain_subset = self.chain.copy()
        if expiration:
            chain_subset = chain_subset[chain_subset['expiration'] == expiration]
        
        puts = chain_subset[chain_subset['type'] == 'put']
        calls = chain_subset[chain_subset['type'] == 'call']
        
        if 'delta' not in chain_subset.columns:
            return pd.DataFrame()
        
        # Filter by delta
        target_puts = puts[(puts['delta'].abs() >= put_delta*0.8) & 
                          (puts['delta'].abs() <= put_delta*1.2)]
        target_calls = calls[(calls['delta'].abs() >= call_delta*0.8) & 
                            (calls['delta'].abs() <= call_delta*1.2)]
        
        strangles = []
        
        for _, put in target_puts.iterrows():
            for _, call in target_calls.iterrows():
                if put['expiration'] != call['expiration']:
                    continue
                
                credit = put['bid'] + call['bid']
                
                if credit < min_credit:
                    continue
                
                strangles.append({
                    'put_strike': put['strike'],
                    'call_strike': call['strike'],
                    'credit': round(credit, 2),
                    'breakeven_lower': round(put['strike'] - credit, 2),
                    'breakeven_upper': round(call['strike'] + credit, 2),
                    'range_width': round(call['strike'] - put['strike'], 2),
                    'put_delta': put['delta'],
                    'call_delta': call['delta'],
                    'expiration': put['expiration'],
                    'dte': put.get('dte', np.nan)
                })
        
        if not strangles:
            return pd.DataFrame()
        
        return pd.DataFrame(strangles).sort_values('credit', ascending=False)
    
    # =========================================================================
    # PROBABILITY CALCULATIONS
    # =========================================================================
    
    def _calculate_prob_profit(self, breakeven: float, delta: float) -> float:
        """
        Estimate probability of profit using delta as proxy.
        
        Delta approximates probability of being ITM at expiration.
        Probability of profit = 1 - abs(delta) for credit spreads.
        """
        if pd.isna(delta):
            return 0.5  # Default to 50% if delta unavailable
        
        return 1 - abs(delta)
    
    def calculate_pop(self, strikes: List[float], option_type: str = 'put',
                     position: str = 'short') -> float:
        """
        Calculate probability of profit for a position.
        
        Parameters
        ----------
        strikes : list
            Strike prices in position
        option_type : str
            'put' or 'call'
        position : str
            'long' or 'short'
            
        Returns
        -------
        float
            Probability of profit (0-1)
        """
        # Simplified calculation using deltas
        chain_subset = self.chain[self.chain['type'] == option_type]
        
        total_delta = 0
        for strike in strikes:
            option = chain_subset[chain_subset['strike'] == strike]
            if not option.empty and 'delta' in option.columns:
                total_delta += option.iloc[0]['delta']
        
        avg_delta = total_delta / len(strikes) if strikes else 0
        
        if position == 'short':
            return 1 - abs(avg_delta)
        else:
            return abs(avg_delta)
    
    # =========================================================================
    # VOLATILITY ANALYSIS
    # =========================================================================
    
    def get_volatility_skew(self, expiration: Optional[str] = None,
                           plot: bool = False) -> pd.DataFrame:
        """
        Analyze implied volatility skew across strikes.
        
        Parameters
        ----------
        expiration : str, optional
            Specific expiration to analyze
        plot : bool
            Whether to plot the skew
            
        Returns
        -------
        pd.DataFrame
            Volatility data by strike
        """
        chain_subset = self.chain.copy()
        
        if expiration:
            chain_subset = chain_subset[chain_subset['expiration'] == expiration]
        
        if 'impliedVolatility' not in chain_subset.columns:
            print("Implied volatility data not available")
            return pd.DataFrame()
        
        skew_data = chain_subset[['strike', 'type', 'impliedVolatility', 'moneyness']].copy()
        skew_data = skew_data.dropna(subset=['impliedVolatility'])
        
        if plot:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for opt_type in ['call', 'put']:
                data = skew_data[skew_data['type'] == opt_type]
                ax.plot(data['strike'], data['impliedVolatility'], 
                       marker='o', label=f'{opt_type.title()}s', linewidth=2)
            
            ax.axvline(self.underlying_price, color='black', linestyle='--', 
                      alpha=0.5, label='Current Price')
            ax.set_xlabel('Strike Price', fontsize=12)
            ax.set_ylabel('Implied Volatility', fontsize=12)
            ax.set_title('Volatility Skew', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        
        return skew_data
    
    # =========================================================================
    # SUMMARY REPORTS
    # =========================================================================
    
    def generate_strategy_report(self, strategy_type: str = 'bull_put',
                                 **kwargs) -> Dict:
        """
        Generate comprehensive strategy analysis report.
        
        Parameters
        ----------
        strategy_type : str
            Type of strategy: 'bull_put', 'bear_call', 'iron_condor', 'strangle'
        **kwargs
            Additional parameters for strategy selection
            
        Returns
        -------
        dict
            Comprehensive strategy report
        """
        report = {
            'underlying_price': self.underlying_price,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'strategy_type': strategy_type
        }
        
        # Get implied move
        implied_move = self.get_implied_move()
        report['implied_move'] = implied_move
        
        # Select strategies
        if strategy_type == 'bull_put':
            strategies = self.select_bull_put_spreads(**kwargs)
        elif strategy_type == 'bear_call':
            strategies = self.select_bear_call_spreads(**kwargs)
        elif strategy_type == 'iron_condor':
            strategies = self.select_iron_condors(**kwargs)
        elif strategy_type == 'strangle':
            strategies = self.select_strangles(**kwargs)
        else:
            strategies = pd.DataFrame()
        
        report['top_strategies'] = strategies.head(10) if not strategies.empty else pd.DataFrame()
        report['num_opportunities'] = len(strategies)
        
        return report


import pandas as pd
import numpy as np 
import QuantLib as ql
import seaborn as sns
import matplotlib.pyplot as plt
from pyfi.core.underlying import Underlying

from datetime import datetime
today = datetime.today()
from enum import Enum

class OptionType(Enum):
    CALL = ql.Option.Call
    PUT = ql.Option.Put

class OptionExposure(Enum):
    LONG = 'Ask'
    SHORT = 'Bid'

RISK_FREE_RATE = 0.05

""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ OptionContract                                                                                                   │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""
class Contract(Underlying):

    def __init__(
        self, 
        ticker:str = None,
        option_type:OptionType=None, 
        option_exposure:OptionExposure=None, 
        valuation:ql.Date=None,
        expiration:ql.Date=None,
        premium=None,
        spot=None,
        K=None,
        ivol=None,
        contract_id=None,
        **kwargs
    ):
        """ 
        BSM Options pricing model. Assumes American exercise styles.

        premium: option contract price (credit/debit premium)
        K: strike price
        ivol: market implied volatility
        """
        period = (datetime(expiration.year(), expiration.month(), expiration.dayOfMonth()) - datetime.today()).days

        super().__init__(ticker=ticker, period=period) # spot; hvol; hvol_two_sigma

        if spot is not None:
            self.spot = spot # NOTE: Override inherited self.spot from Underlying();  self._spot is @set in Underlying
        else:
            self._spot = self.spot # self.spot is most recent price available from Underlying() with no override
     
        self.intrinsic_value = self._spot - K
        self.time_value = premium - self.intrinsic_value

        self.option_type = option_type
        self.option_exposure = option_exposure
        self.valuation = valuation
        self.expiration = expiration
        self.premium = premium
        self.K = K
        self.ivol = ivol 
        self.contract_id = contract_id
        self.rfr = RISK_FREE_RATE

        print(self)


    def __str__(self):
        return str({k: v for k, v in vars(self).items() if not isinstance(v, pd.Series)
                    and not isinstance(v, pd.DataFrame)})


    def solve_for_iv(
        self, option_type, underlying_price, strike_price, expiration_date, valuation_date, option_price, risk_free_rate
    ):
        """ 
        https://stackoverflow.com/questions/4891490/calculating-europeanoptionimpliedvolatility-in-quantlib-python
        """
        day_count = ql.Actual365Fixed()
        calendar = ql.UnitedStates(ql.UnitedStates.NYSE)

        option = ql.VanillaOption(
            ql.PlainVanillaPayoff(option_type, strike_price),
            ql.AmericanExercise(valuation_date, expiration_date)
        )
        
        process = ql.BlackScholesProcess(
            ql.QuoteHandle(ql.SimpleQuote(underlying_price)),
            ql.YieldTermStructureHandle(ql.FlatForward(valuation_date, risk_free_rate, day_count)),
            ql.BlackVolTermStructureHandle(ql.BlackConstantVol(valuation_date, calendar, 0, day_count))  # Initial volatility guess (0.2)
        )

        binomial_engine = ql.BinomialVanillaEngine(process, "crr", 300)
        option.setPricingEngine(binomial_engine)

        try:
            implied_volatility = option.impliedVolatility(option_price, process)
        except RuntimeError:
            implied_volatility = np.nan

        return implied_volatility


    def solve_for_npv(
        self, option_type, underlying_price, strike_price, expiration_date, valuation_date, implied_volatility, risk_free_rate
    ):
        day_count = ql.Actual365Fixed()
        calendar = ql.UnitedStates(ql.UnitedStates.NYSE)

        option = ql.VanillaOption(
            ql.PlainVanillaPayoff(option_type, strike_price), 
            ql.AmericanExercise(valuation_date, expiration_date)
        )
        
        process = ql.BlackScholesProcess(
            ql.QuoteHandle(ql.SimpleQuote(underlying_price)),
            ql.YieldTermStructureHandle(ql.FlatForward(valuation_date, risk_free_rate, day_count)),
            ql.BlackVolTermStructureHandle(ql.BlackConstantVol(valuation_date, calendar, implied_volatility, day_count))
        )  

        binomial_engine = ql.BinomialVanillaEngine(process, "crr", 100)
        option.setPricingEngine(binomial_engine)

        try:
            npv = option.NPV() # RuntimeError: negative probability
        except RuntimeError:
            npv = np.nan

        return npv


    def calculate_option_greeks(
        self, option_type, underlying_price, strike_price, expiration_date, valuation_date, volatility, risk_free_rate
    ):
        day_count = ql.Actual365Fixed()
        calendar = ql.UnitedStates(ql.UnitedStates.NYSE)

        option = ql.VanillaOption(
            ql.PlainVanillaPayoff(option_type, strike_price),
            ql.AmericanExercise(valuation_date, expiration_date)
        )
        
        underlying = ql.SimpleQuote(underlying_price)
        underlying_handle = ql.QuoteHandle(underlying)
        flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(valuation_date, risk_free_rate, day_count))
        flat_vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(valuation_date, calendar, volatility, day_count))
        process = ql.BlackScholesProcess(underlying_handle, flat_ts, flat_vol_ts)

        binomial_engine = ql.BinomialVanillaEngine(process, "crr", 100)
        option.setPricingEngine(binomial_engine)

        try:
            delta = option.delta()
            gamma = option.gamma()
            theta = option.theta()
        except:
            delta, gamma, theta = np.nan, np.nan, np.nan

        return {'delta': delta, 'gamma': gamma, 'theta': theta}



    def analyze(self):
        npv_market_iv = self.solve_for_npv(
            self.option_type, 
            self._spot, 
            self.K, 
            self.expiration, 
            self.valuation, 
            self.ivol, 
            self.rfr
        )
        
        npv_historical_iv = self.solve_for_npv(
            self.option_type, 
            self._spot, 
            self.K, 
            self.expiration, 
            self.valuation, 
            self.hvol, 
            self.rfr
        )

        npv_2std_historical_iv = self.solve_for_npv(
            self.option_type, 
            self._spot, 
            self.K, 
            self.expiration, 
            self.valuation, 
            self.hvol_two_sigma, 
            self.rfr
        )

        npv_garch_iv = self.solve_for_npv(
            self.option_type, 
            self._spot, 
            self.K, 
            self.expiration, 
            self.valuation, 
            self.hvol_garch, 
            self.rfr
        )

        iv = self.solve_for_iv(
            self.option_type, 
            self._spot, 
            self.K, 
            self.expiration, 
            self.valuation, 
            self.premium, 
            self.rfr
        )

        greeks = self.calculate_option_greeks(
            self.option_type, 
            self._spot, 
            self.K, 
            self.expiration, 
            self.valuation, 
            self.ivol, 
            self.rfr
        )
            
        instance_vars =  {key: value for key, value in vars(self).items() if not isinstance(value, pd.DataFrame)}

        res_dict = {
            **instance_vars,
            'npv_market_iv': npv_market_iv, 
            'stdev': self.stdev,
            'hvol':self.hvol,
            'npv_hvol': npv_historical_iv, 
            'hvol_two_sigma':self.hvol_two_sigma,
            'npv_hvol_two_sigma': npv_2std_historical_iv,
            'hvol_garch':self.hvol_garch,
            'npv_garch':npv_garch_iv, 
            'ivol':self.ivol,
            'ivol_calculated': iv,
            **greeks
        }
            
        return pd.DataFrame(res_dict, index=[0])



""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ OptionChain                                                                                                      │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""
class Chain:

    def __init__(self, 
                 chain:pd.DataFrame=None, 
                 ticker=None,
                 option_type = OptionType.CALL, 
                 option_exposure = OptionExposure.LONG, 
                 **kwargs ):
        """ Iterates through an option chain, represented as a pandas dataframe, and instantiates 
        Contract objects to represent each row (contract) on the chain.

        Chain representation is constructed in retreivers.options
        
        Returns pd.DataFrame with Contract detail and analytic results
        """
        
        self.chain = chain

        self.option_type = option_type.value
        
        self.option_exposure = option_exposure

        self.ticker=ticker

        for key, value in kwargs.items():
            setattr(self, key, value)


    def get_params(self, contract:pd.DataFrame):
        
        self.contract_id = contract['Contract Name']
        
        self.params = contract.to_frame().T.set_index(['Contract Name']).to_dict(orient = 'index').get(self.contract_id)
        
        return self.params


    def set_instance_vars(self, params):

        if not hasattr(self, 'spot'):
            self.spot = None

        if not hasattr(self, 'valuation_date'):
            self.valuation_date = ql.Date(today.day, today.month,  today.year) 

        if self.option_exposure == OptionExposure.LONG:
            self.premium = params['Last Price'] #params['Ask']

        elif self.option_exposure == OptionExposure.SHORT:
            self.premium = params['Last Price'] #params['Bid']


    def instantiate_contract(self, data):
            
        params = self.get_params(data)
        # print(params)

        self.set_instance_vars(params)

        option_contract = Contract(
            ticker=self.ticker,
            option_type=self.option_type,
            option_exposure=self.option_exposure,
            valuation=ql.Date(today.day, today.month, today.year),
            expiration=ql.Date(params['Expiration_dt'].day, params['Expiration_dt'].month, params['Expiration_dt'].year),
            premium=self.premium,
            spot=self.spot,
            K=params['Strike'],
            ivol=params['Market_IV'],
            contract_id=self.contract_id
        )

        return option_contract
    

    def process_chain(self):

        frames = []

        for row in self.chain.iterrows():

            ix, data = row
    
            res = self.instantiate_contract(data).analyze()

            self.full_res = pd.concat([data, res.T], axis=0)

            self.full_res.columns = self.full_res.loc['Contract Name']

            self.full_res = self.full_res.iloc[1:]

            self.full_res.columns.name = None

            frames.append(self.full_res)
        
        return  pd.concat(frames, axis=1).drop(['Change', 'contract_id', 'Expiration_dt', 'Market_IV', 'ticker', 'price_ts'], axis=0)



    def get_volatility_skew(self, plot=False):
        res = self.chain[['Strike', 'Expiration Date', 'Implied Volatility']]
        
        res['Implied Volatility'] = pd.to_numeric(res['Implied Volatility'].str.replace('%', ''))

        if plot:
            sns.lineplot(data=res, x='Strike', y='Implied Volatility', hue = 'Expiration Date')
            plt.show()

        return res


""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Option Strategies                                                                                                │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""

class Strategy:

    def __init__(self) -> None:
        """ A class to be inherited by each option strategy.

        sT: End Price; Series of underlying stock price at time T.
        K: Strike price
        p: Premium (Credit/Debit from purchase)
        """
        pass

    @staticmethod
    def put_payoff(sT, K, p):
        return np.where(sT < K, K - sT, 0) - p

    @staticmethod
    def call_payoff(sT, K, p):
        return np.where(sT > K, sT - K, 0) - p

    @staticmethod
    def get_sT(K):
        return np.arange(K*0.75,K*1.25,.1)

    @staticmethod
    def get_pl_ratio(p, l):
        return round(p / l, 2)



from pyfi.core.options.options import Strategy, Contract
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class BearCallCreditSpread(Strategy):

    def __init__(
        self,
        n_contracts:int=1,
        clsLong:Contract = None,
        clsShort:Contract = None
    ) -> None:
        
        super().__init__()

        self.n_contracts = n_contracts * 100

        self.sT = self.get_sT(clsShort.K)

        self.short_call_payoff = self.call_payoff(sT=self.sT, K=clsShort.K, p=clsShort.premium) * -1 * self.n_contracts

        self.long_call_payoff = self.call_payoff(sT=self.sT, K=clsLong.K, p=clsLong.premium) * self.n_contracts

        spot_prices_df = pd.DataFrame({'spot': pd.Series(self.sT)})
        
        self.spread_payoff = spot_prices_df.apply(lambda row: self.bear_call_spread_payoff(
            row['spot'], clsLong.K, clsShort.K, clsLong.premium, clsShort.premium), axis=1) * self.n_contracts
        
        self.clsLong = clsLong
        self.clsShort = clsShort

    @property
    def max_profit(self):
        return (self.clsShort.premium - self.clsLong.premium) * self.n_contracts

    @property
    def max_loss(self):
        return ((self.clsShort.K - self.clsLong.K) + (self.clsShort.premium - self.clsLong.premium)) * self.n_contracts

    @property
    def pl_ratio(self):
        return self.get_pl_ratio(self.max_profit, self.max_loss)

    @property
    def breakeven(self):
        short_call_strike = self.clsShort.K
        net_credit = self.clsShort.premium - self.clsLong.premium
        breakeven_point = short_call_strike + net_credit
        return breakeven_point

    def bear_call_spread_payoff(self, spot_price, call_strike_long, call_strike_short, long_call_cost, short_call_credit):
        long_call_payoff = max(spot_price - call_strike_long, 0) - long_call_cost

        short_call_payoff = min(call_strike_short - spot_price, short_call_credit)

        bear_call_spread_payoff = long_call_payoff + short_call_payoff

        return bear_call_spread_payoff

    def plot(self):
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(self.sT, self.short_call_payoff, '--', color='r', label='Short Call Payoff', alpha=0.3)
        ax.plot(self.sT, self.long_call_payoff, '--', color='g', label='Long Call Payoff', alpha=0.3)
        ax.plot(self.sT, self.spread_payoff, color='black', label='Spread Payoff')

        ax.scatter(self.sT.min(), self.max_profit, color='lightgreen', s=20, marker='o')
        ax.text(self.sT.min(), self.max_profit, f'Max Profit\n{self.max_profit}', ha='left', va='center', fontsize=8)

        ax.scatter(self.sT.max(), self.max_loss, color='red', s=20, marker='o')
        ax.text(self.sT.max(), self.max_loss, f'Max Loss\n{self.max_loss}', ha='right', va='center', fontsize=8)

        ax.axvline(self.breakeven, color='black', linestyle='--', alpha=0.5)
        ax.text(self.breakeven, self.long_call_payoff.max(), f'Breakeven\n{self.breakeven}', ha='left', va='center', fontsize=8)

        ax.axvline(self.clsShort.spot, color='grey', linestyle='--', alpha=0.5)
        ax.text(self.clsShort.spot, self.short_call_payoff.min(), f'Spot\n{round(self.clsShort.spot, 2)}', ha='left', va='center', fontsize=8)

        ax.grid(axis='x', which='major', visible=False)
        ax.grid(axis='y', which='major', visible=False)

        plt.legend()
        plt.xlabel('Stock Price')
        plt.ylabel('Profit & Loss')
        plt.show()



from pyfi.core.options.options import Strategy, Contract
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class BullPutCreditSpread(Strategy):

    def __init__(
        self,
        n_contracts:int=1,
        clsLong:Contract = None,
        clsShort:Contract = None
    ) -> None:
        
        super().__init__()

        self.n_contracts = n_contracts * 100

        self.sT = self.get_sT(clsLong.K)

        self.long_put_payoff = self.put_payoff(sT = self.sT, K = clsLong.K, p = clsLong.premium)   * self.n_contracts

        self.short_put_payoff = self.put_payoff(sT = self.sT, K = clsShort.K, p = clsShort.premium) * -1  * self.n_contracts

        spot_prices_df = pd.DataFrame({'spot': pd.Series(self.sT)})
        
        self.spread_payoff = spot_prices_df.apply(lambda row: self.bull_put_spread_payoff(
            row['spot'], clsLong.K, clsShort.K, clsLong.premium, clsShort.premium), axis=1)  * self.n_contracts
        
        self.clsLong = clsLong
        self.clsShort = clsShort

    @property
    def max_profit(self):
        return (self.clsShort.premium - self.clsLong.premium) * self.n_contracts

    @property
    def max_loss(self):
        return ((self.clsShort.K - self.clsLong.K) - (self.clsShort.premium - self.clsLong.premium)) *-1 * self.n_contracts

    @property
    def pl_ratio(self):
        return self.get_pl_ratio(self.max_profit, self.max_loss)

    @property
    def pl_odds(self):
        return f'{round(self.pl_ratio/ (1-self.pl_ratio), 2)}:1'

    @property
    def breakeven(self):
        short_put_strike = self.clsShort.K
        net_credit = self.clsShort.premium - self.clsLong.premium
        breakeven_point = short_put_strike - net_credit
        return breakeven_point


    def bull_put_spread_payoff(self, spot_price, put_strike_long, put_strike_short, long_put_cost, short_put_credit):
        long_put_payoff = (max(put_strike_long - spot_price, 0) - long_put_cost ) 

        short_put_payoff = min(spot_price - put_strike_short, short_put_credit) 

        bull_put_spread_payoff = long_put_payoff + short_put_payoff

        return bull_put_spread_payoff


    
    def plot(self):
        fig, ax = plt.subplots(figsize=(10,5))

        ax.plot(self.sT, self.short_put_payoff,'--', color ='g', label = 'Short Put Payoff', alpha = 0.3)
        ax.plot(self.sT, self.long_put_payoff,'--', color ='r', label ='Long Put Payoff', alpha = 0.3)
        ax.plot(self.sT, self.spread_payoff, color ='black', label ='Spread Payoff')

        ax.scatter(self.sT.max(), self.max_profit, color='lightgreen', s=20, marker='o')
        ax.text(self.sT.max(), self.max_profit, f'Max Profit: {self.max_profit}', ha='center', va='top',  fontsize=8)

        ax.scatter(self.sT.min(), self.max_loss, color='red', s=20, marker='o')
        ax.text(self.sT.min(), self.max_loss, f'Max Loss: {self.max_loss}', ha='center', va='top', fontsize=8)

        ax.axvline(self.breakeven, color ='black', linestyle='--', alpha = 0.5)
        ax.text(self.breakeven, self.long_put_payoff.max(), f'Breakeven: {self.breakeven}', ha='center', va='top', fontsize=8)

        ax.axvline(self.clsLong.spot,  color ='grey', linestyle='--', alpha = 0.5)
        ax.text(self.clsLong.spot, self.short_put_payoff.min(), f'Spot:\n{round(self.clsLong.spot,2)}', ha='center', va='top', fontsize=8)

        ax.text(self.sT.max()*.98, self.short_put_payoff.min(), f'P/L Ratio:\n{self.pl_ratio}', ha='center', va='top', fontsize=8)

        ax.text(self.sT.max()*.93, self.short_put_payoff.min(), f'P/L Odds:\n{self.pl_odds}', ha='center', va='top', fontsize=8)

        ax.grid(axis='x', which='major', visible=False)
        ax.grid(axis='y', which='major', visible=False)

        plt.legend()
        plt.xlabel('Stock Price')
        plt.ylabel('Profit & Loss')
        plt.show()


# =============================================================================
# POSITION RISK CALCULATOR
# =============================================================================

class PositionRiskCalculator:
    """
    Calculate risk metrics for options positions.
    
    Examples
    --------
    >>> calc = PositionRiskCalculator()
    >>> risk = calc.credit_spread_risk(short_strike=100, long_strike=95, 
    ...                                 credit=1.50, quantity=10)
    >>> print(f"Max loss: ${risk['max_loss']}, P/L ratio: {risk['pl_ratio']}")
    """
    
    @staticmethod
    def credit_spread_risk(short_strike: float, long_strike: float,
                          credit: float, quantity: int = 1) -> Dict[str, float]:
        """Calculate risk metrics for credit spreads."""
        width = abs(short_strike - long_strike)
        max_profit = credit * 100 * quantity
        max_loss = (width - credit) * 100 * quantity
        pl_ratio = credit / (width - credit) if width > credit else 0
        
        if short_strike > long_strike:
            breakeven = short_strike - credit
            spread_type = 'bull_put'
        else:
            breakeven = short_strike + credit
            spread_type = 'bear_call'
        
        return {
            'spread_type': spread_type,
            'max_profit': round(max_profit, 2),
            'max_loss': round(max_loss, 2),
            'pl_ratio': round(pl_ratio, 3),
            'breakeven': round(breakeven, 2),
            'width': width,
            'credit': credit,
            'quantity': quantity
        }
    
    @staticmethod
    def iron_condor_risk(put_short_strike: float, put_long_strike: float,
                        call_short_strike: float, call_long_strike: float,
                        total_credit: float, quantity: int = 1) -> Dict[str, float]:
        """Calculate risk metrics for iron condors."""
        put_width = put_short_strike - put_long_strike
        call_width = call_long_strike - call_short_strike
        max_width = max(put_width, call_width)
        
        max_profit = total_credit * 100 * quantity
        max_loss = (max_width - total_credit) * 100 * quantity
        pl_ratio = total_credit / (max_width - total_credit) if max_width > total_credit else 0
        
        breakeven_lower = put_short_strike - total_credit
        breakeven_upper = call_short_strike + total_credit
        
        return {
            'max_profit': round(max_profit, 2),
            'max_loss': round(max_loss, 2),
            'pl_ratio': round(pl_ratio, 3),
            'breakeven_lower': round(breakeven_lower, 2),
            'breakeven_upper': round(breakeven_upper, 2),
            'profit_range': round(breakeven_upper - breakeven_lower, 2),
            'quantity': quantity
        }


# =============================================================================
# PROBABILITY CALCULATOR  
# =============================================================================

class ProbabilityCalculator:
    """Calculate probabilities for options strategies."""
    
    @staticmethod
    def expected_move(current_price: float, volatility: float,
                     days: int, confidence: float = 0.68) -> Dict[str, float]:
        """
        Calculate expected price movement.
        
        Parameters
        ----------
        current_price : float
            Current price
        volatility : float
            Implied volatility (annualized, as decimal)
        days : int
            Days to expiration
        confidence : float
            Confidence level (0.68 = 1σ, 0.95 = 2σ)
            
        Returns
        -------
        dict
            Expected move statistics
        """
        # Calculate standard deviation for the period
        std_dev = volatility * np.sqrt(days / 365)
        
        # Approximate z-score
        if confidence >= 0.95:
            z = 2.0
        elif confidence >= 0.68:
            z = 1.0
        else:
            z = 0.5
        
        move_pct = std_dev * z
        move_dollar = current_price * move_pct
        
        return {
            'confidence': confidence,
            'move_pct': round(move_pct * 100, 2),
            'move_dollar': round(move_dollar, 2),
            'upper_bound': round(current_price + move_dollar, 2),
            'lower_bound': round(current_price - move_dollar, 2),
            'days': days
        }


# =============================================================================
# OPTIONS UTILITIES
# =============================================================================

class OptionsUtils:
    """Utility functions for options analysis."""
    
    @staticmethod
    def calculate_dte(expiration_date: Union[str, datetime]) -> int:
        """Calculate days to expiration."""
        if isinstance(expiration_date, str):
            exp_date = pd.to_datetime(expiration_date)
        else:
            exp_date = expiration_date
        
        today = datetime.now()
        dte = (exp_date - today).days
        
        return max(dte, 0)
    
    @staticmethod
    def annualize_return(return_pct: float, days: int) -> float:
        """Annualize a return based on holding period."""
        if days <= 0:
            return 0
        
        periods_per_year = 365 / days
        annualized = return_pct * periods_per_year
        
        return round(annualized, 2)
    
    @staticmethod
    def format_options_chain(raw_chain: pd.DataFrame) -> pd.DataFrame:
        """Format and clean options chain data."""
        chain = raw_chain.copy()
        
        # Standardize column names
        column_mapping = {
            'Strike': 'strike',
            'Bid': 'bid',
            'Ask': 'ask',
            'Last': 'last',
            'Volume': 'volume',
            'Open Interest': 'open_interest',
            'Implied Volatility': 'impliedVolatility'
        }
        
        chain.rename(columns=column_mapping, inplace=True)
        
        # Calculate mid price
        if 'bid' in chain.columns and 'ask' in chain.columns:
            chain['mid'] = (chain['bid'] + chain['ask']) / 2
        
        # Convert IV to decimal if percentage
        if 'impliedVolatility' in chain.columns:
            if chain['impliedVolatility'].dtype == 'object':
                chain['impliedVolatility'] = pd.to_numeric(
                    chain['impliedVolatility'].str.replace('%', ''),
                    errors='coerce'
                ) / 100
        
        return chain
    
    @staticmethod
    def filter_liquid_options(chain: pd.DataFrame, min_volume: int = 10,
                             min_oi: int = 50) -> pd.DataFrame:
        """Filter for liquid options only."""
        filtered = chain.copy()
        
        if 'volume' in filtered.columns:
            filtered = filtered[filtered['volume'] >= min_volume]
        
        if 'open_interest' in filtered.columns:
            filtered = filtered[filtered['open_interest'] >= min_oi]
        
        return filtered


# =============================================================================
# GREEKS AGGREGATOR
# =============================================================================

class GreeksAggregator:
    """
    Aggregate and analyze Greeks for multi-leg positions.
    
    Examples
    --------
    >>> agg = GreeksAggregator()
    >>> positions = [
    ...     {'delta': 0.30, 'gamma': 0.05, 'theta': -0.10, 'vega': 0.15, 'quantity': 10},
    ...     {'delta': -0.30, 'gamma': 0.05, 'theta': -0.10, 'vega': 0.15, 'quantity': 10}
    ... ]
    >>> portfolio_greeks = agg.aggregate_greeks(positions)
    """
    
    @staticmethod
    def aggregate_greeks(positions: List[Dict]) -> Dict[str, float]:
        """
        Aggregate Greeks across multiple positions.
        
        Parameters
        ----------
        positions : list of dict
            Each dict contains: delta, gamma, theta, vega, quantity
            
        Returns
        -------
        dict
            Aggregated Greeks
        """
        total_delta = sum(p['delta'] * p['quantity'] for p in positions)
        total_gamma = sum(p['gamma'] * p['quantity'] for p in positions)
        total_theta = sum(p['theta'] * p['quantity'] for p in positions)
        total_vega = sum(p['vega'] * p['quantity'] for p in positions)
        
        return {
            'delta': round(total_delta, 3),
            'gamma': round(total_gamma, 3),
            'theta': round(total_theta, 3),
            'vega': round(total_vega, 3),
            'positions': len(positions)
        }
    
    @staticmethod
    def delta_neutral_adjustment(current_delta: float, shares_per_contract: int = 100) -> Dict[str, float]:
        """
        Calculate shares needed to delta neutralize a position.
        
        Parameters
        ----------
        current_delta : float
            Portfolio delta exposure
        shares_per_contract : int
            Shares per options contract
            
        Returns
        -------
        dict
            Adjustment details
        """
        shares_needed = -current_delta * shares_per_contract
        
        return {
            'current_delta': round(current_delta, 3),
            'shares_to_trade': round(shares_needed, 0),
            'direction': 'BUY' if shares_needed > 0 else 'SELL',
            'abs_shares': abs(round(shares_needed, 0))
        }


# =============================================================================
# STRATEGY COMPARATOR
# =============================================================================

class StrategyComparator:
    """
    Compare multiple options strategies side-by-side.
    
    Examples
    --------
    >>> comp = StrategyComparator()
    >>> strategies = [
    ...     {'name': 'Bull Put', 'max_profit': 150, 'max_loss': 350, 'prob_profit': 0.70},
    ...     {'name': 'Iron Condor', 'max_profit': 200, 'max_loss': 300, 'prob_profit': 0.65}
    ... ]
    >>> comp.compare(strategies)
    """
    
    @staticmethod
    def calculate_expected_value(max_profit: float, max_loss: float,
                                 prob_profit: float) -> float:
        """Calculate expected value of strategy."""
        prob_loss = 1 - prob_profit
        ev = (max_profit * prob_profit) - (max_loss * prob_loss)
        return round(ev, 2)
    
    @staticmethod
    def compare(strategies: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare strategies and rank by metrics.
        
        Parameters
        ----------
        strategies : list of dict
            Each dict should contain: name, max_profit, max_loss, 
            and optionally prob_profit, days
            
        Returns
        -------
        pd.DataFrame
            Comparison table sorted by expected value
        """
        comparisons = []
        
        for s in strategies:
            name = s.get('name', 'Unknown')
            max_profit = s.get('max_profit', 0)
            max_loss = s.get('max_loss', 0)
            prob_profit = s.get('prob_profit', 0.5)
            days = s.get('days', 30)
            
            pl_ratio = max_profit / max_loss if max_loss > 0 else 0
            ev = StrategyComparator.calculate_expected_value(
                max_profit, max_loss, prob_profit
            )
            
            # Annualized return on risk
            if days > 0 and max_loss > 0:
                daily_return = ev / max_loss
                annual_return = daily_return * 365 / days
            else:
                annual_return = 0
            
            comparisons.append({
                'Strategy': name,
                'Max Profit': max_profit,
                'Max Loss': max_loss,
                'P/L Ratio': round(pl_ratio, 3),
                'Prob Profit': round(prob_profit, 3),
                'Expected Value': ev,
                'Annual ROI%': round(annual_return * 100, 1),
                'Days': days
            })
        
        df = pd.DataFrame(comparisons)
        df = df.sort_values('Expected Value', ascending=False)
        df = df.reset_index(drop=True)
        
        return df
    
    @staticmethod
    def plot_comparison(strategies: List[Dict[str, Any]], figsize: Tuple[int, int] = (12, 6)):
        """Plot visual comparison of strategies."""
        df = StrategyComparator.compare(strategies)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # P/L comparison
        ax1 = axes[0]
        x = np.arange(len(df))
        width = 0.35
        
        ax1.bar(x - width/2, df['Max Profit'], width, label='Max Profit', color='green', alpha=0.7)
        ax1.bar(x + width/2, df['Max Loss'], width, label='Max Loss', color='red', alpha=0.7)
        ax1.set_xlabel('Strategy')
        ax1.set_ylabel('Dollar Amount')
        ax1.set_title('Max Profit vs Max Loss')
        ax1.set_xticks(x)
        ax1.set_xticklabels(df['Strategy'], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Expected value comparison
        ax2 = axes[1]
        colors = ['green' if ev > 0 else 'red' for ev in df['Expected Value']]
        ax2.barh(df['Strategy'], df['Expected Value'], color=colors, alpha=0.7)
        ax2.set_xlabel('Expected Value ($)')
        ax2.set_title('Expected Value Comparison')
        ax2.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.show()






# # =============================================================================
# # EXAMPLE USAGE
# # =============================================================================

# if __name__ == "__main__":
#     """
#     Example implementations demonstrating how to use the options analysis toolkit.
#     """
    
#     # =========================================================================
#     # Example 1: Analyze Options Strategies with OptionsAnalyzer
#     # =========================================================================
#     print("=" * 80)
#     print("EXAMPLE 1: OptionsAnalyzer - Strategy Screening")
#     print("=" * 80)
    
#     # Create sample options chain data
#     strikes = np.arange(95, 106, 1)
#     sample_chain = pd.DataFrame({
#         'strike': strikes,
#         'bid': np.random.uniform(0.5, 3.0, len(strikes)),
#         'ask': np.random.uniform(3.5, 6.0, len(strikes)),
#         'impliedVolatility': np.random.uniform(0.20, 0.40, len(strikes)),
#         'volume': np.random.randint(50, 500, len(strikes)),
#         'open_interest': np.random.randint(100, 2000, len(strikes))
#     })
#     sample_chain['mid'] = (sample_chain['bid'] + sample_chain['ask']) / 2
    
#     # Initialize analyzer
#     analyzer = OptionsAnalyzer(current_price=100.0, days_to_expiration=30)
    
#     # Calculate implied move
#     implied_move = analyzer.get_implied_move(avg_iv=0.30)
#     print(f"\nImplied Move (1σ, 30 days):")
#     print(f"  Range: ${implied_move['lower_bound']:.2f} - ${implied_move['upper_bound']:.2f}")
#     print(f"  Move: ±${implied_move['move_dollar']:.2f} ({implied_move['move_pct']:.1f}%)")
    
#     # Screen for bull put spreads
#     print("\nBull Put Spreads (30-40 delta, min $0.30 credit):")
#     bull_puts = analyzer.select_bull_put_spreads(
#         chain=sample_chain,
#         target_delta_range=(0.30, 0.40),
#         min_credit=0.30,
#         width=5
#     )
#     if not bull_puts.empty:
#         print(bull_puts.head(3).to_string(index=False))
    
#     # Screen for iron condors
#     print("\nIron Condor Candidates (16 delta, $5 wide):")
#     condors = analyzer.select_iron_condors(
#         put_chain=sample_chain,
#         call_chain=sample_chain,
#         put_delta=0.16,
#         call_delta=0.16,
#         width=5,
#         min_credit=0.50
#     )
#     if not condors.empty:
#         print(condors.head(2).to_string(index=False))
    
#     # =========================================================================
#     # Example 2: Calculate Risk Metrics with PositionRiskCalculator
#     # =========================================================================
#     print("\n" + "=" * 80)
#     print("EXAMPLE 2: PositionRiskCalculator - Risk Analysis")
#     print("=" * 80)
    
#     # Bull put spread risk
#     bull_put_risk = PositionRiskCalculator.credit_spread_risk(
#         short_strike=95,
#         long_strike=90,
#         credit=1.25,
#         quantity=5
#     )
#     print("\nBull Put Spread (95/90, $1.25 credit, 5 contracts):")
#     print(f"  Max Profit: ${bull_put_risk['max_profit']:,.2f}")
#     print(f"  Max Loss: ${bull_put_risk['max_loss']:,.2f}")
#     print(f"  P/L Ratio: {bull_put_risk['pl_ratio']:.2f}")
#     print(f"  Breakeven: ${bull_put_risk['breakeven']:.2f}")
    
#     # Iron condor risk
#     ic_risk = PositionRiskCalculator.iron_condor_risk(
#         put_short_strike=95,
#         put_long_strike=90,
#         call_short_strike=105,
#         call_long_strike=110,
#         total_credit=2.00,
#         quantity=3
#     )
#     print("\nIron Condor (95/90 put, 105/110 call, $2.00 credit, 3 contracts):")
#     print(f"  Max Profit: ${ic_risk['max_profit']:,.2f}")
#     print(f"  Max Loss: ${ic_risk['max_loss']:,.2f}")
#     print(f"  P/L Ratio: {ic_risk['pl_ratio']:.2f}")
#     print(f"  Profit Range: ${ic_risk['breakeven_lower']:.2f} - ${ic_risk['breakeven_upper']:.2f}")
#     print(f"  Range Width: ${ic_risk['profit_range']:.2f}")
    
#     # =========================================================================
#     # Example 3: Probability Analysis with ProbabilityCalculator
#     # =========================================================================
#     print("\n" + "=" * 80)
#     print("EXAMPLE 3: ProbabilityCalculator - Expected Moves")
#     print("=" * 80)
    
#     # 1 standard deviation move
#     move_1std = ProbabilityCalculator.expected_move(
#         current_price=100,
#         volatility=0.30,
#         days=30,
#         confidence=0.68
#     )
#     print(f"\n1σ Expected Move (68% confidence, 30 days):")
#     print(f"  Range: ${move_1std['lower_bound']:.2f} - ${move_1std['upper_bound']:.2f}")
#     print(f"  Move: ±{move_1std['move_pct']:.1f}%")
    
#     # 2 standard deviation move
#     move_2std = ProbabilityCalculator.expected_move(
#         current_price=100,
#         volatility=0.30,
#         days=30,
#         confidence=0.95
#     )
#     print(f"\n2σ Expected Move (95% confidence, 30 days):")
#     print(f"  Range: ${move_2std['lower_bound']:.2f} - ${move_2std['upper_bound']:.2f}")
#     print(f"  Move: ±{move_2std['move_pct']:.1f}%")
    
#     # =========================================================================
#     # Example 4: Portfolio Greeks with GreeksAggregator
#     # =========================================================================
#     print("\n" + "=" * 80)
#     print("EXAMPLE 4: GreeksAggregator - Portfolio Greeks")
#     print("=" * 80)
    
#     # Multi-leg position
#     positions = [
#         {'delta': 0.30, 'gamma': 0.05, 'theta': -0.10, 'vega': 0.15, 'quantity': 10},
#         {'delta': -0.30, 'gamma': 0.05, 'theta': -0.10, 'vega': 0.15, 'quantity': 10},
#         {'delta': 0.15, 'gamma': 0.03, 'theta': -0.05, 'vega': 0.08, 'quantity': 5},
#         {'delta': -0.15, 'gamma': 0.03, 'theta': -0.05, 'vega': 0.08, 'quantity': 5},
#     ]
    
#     portfolio_greeks = GreeksAggregator.aggregate_greeks(positions)
#     print(f"\nPortfolio Greeks (4 legs, 30 total contracts):")
#     print(f"  Delta: {portfolio_greeks['delta']:+.3f}")
#     print(f"  Gamma: {portfolio_greeks['gamma']:+.3f}")
#     print(f"  Theta: {portfolio_greeks['theta']:+.3f}")
#     print(f"  Vega: {portfolio_greeks['vega']:+.3f}")
    
#     # Delta hedge calculation
#     hedge = GreeksAggregator.delta_neutral_adjustment(
#         current_delta=0.75,
#         shares_per_contract=100
#     )
#     print(f"\nDelta Hedge (current delta: {hedge['current_delta']}):")
#     print(f"  {hedge['direction']} {hedge['abs_shares']:.0f} shares to neutralize")
    
#     # =========================================================================
#     # Example 5: Strategy Comparison with StrategyComparator
#     # =========================================================================
#     print("\n" + "=" * 80)
#     print("EXAMPLE 5: StrategyComparator - Compare Strategies")
#     print("=" * 80)
    
#     # Define strategies to compare
#     strategies = [
#         {
#             'name': 'Bull Put 95/90',
#             'max_profit': 625,
#             'max_loss': 1875,
#             'prob_profit': 0.70,
#             'days': 30
#         },
#         {
#             'name': 'Iron Condor 95/90-105/110',
#             'max_profit': 600,
#             'max_loss': 900,
#             'prob_profit': 0.65,
#             'days': 30
#         },
#         {
#             'name': 'Bear Call 105/110',
#             'max_profit': 575,
#             'max_loss': 1925,
#             'prob_profit': 0.68,
#             'days': 30
#         },
#         {
#             'name': 'Short Strangle 90/110',
#             'max_profit': 800,
#             'max_loss': 999999,  # Undefined risk
#             'prob_profit': 0.75,
#             'days': 30
#         }
#     ]
    
#     # Compare strategies
#     comparison = StrategyComparator.compare(strategies)
#     print("\nStrategy Comparison (sorted by Expected Value):")
#     print(comparison.to_string(index=False))
    
#     # Calculate expected value manually for one strategy
#     ev = StrategyComparator.calculate_expected_value(
#         max_profit=625,
#         max_loss=1875,
#         prob_profit=0.70
#     )
#     print(f"\nBull Put Expected Value: ${ev:.2f}")
    
#     # =========================================================================
#     # Example 6: Utility Functions with OptionsUtils
#     # =========================================================================
#     print("\n" + "=" * 80)
#     print("EXAMPLE 6: OptionsUtils - Helper Functions")
#     print("=" * 80)
    
#     # Calculate DTE
#     expiration = datetime.now() + timedelta(days=45)
#     dte = OptionsUtils.calculate_dte(expiration)
#     print(f"\nDays to Expiration: {dte} days")
    
#     # Annualize return
#     return_pct = 2.5  # 2.5% return
#     days_held = 30
#     annualized = OptionsUtils.annualize_return(return_pct, days_held)
#     print(f"\nReturn Annualization:")
#     print(f"  30-day return: {return_pct:.1f}%")
#     print(f"  Annualized: {annualized:.1f}%")
    
#     # Format options chain
#     raw_chain = pd.DataFrame({
#         'Strike': [95, 100, 105],
#         'Bid': [2.0, 1.5, 1.0],
#         'Ask': [2.2, 1.7, 1.2],
#         'Volume': [100, 200, 150],
#         'Open Interest': [500, 800, 600],
#         'Implied Volatility': ['25%', '30%', '28%']
#     })
    
#     formatted = OptionsUtils.format_options_chain(raw_chain)
#     print("\nFormatted Options Chain:")
#     print(formatted[['strike', 'bid', 'ask', 'mid', 'impliedVolatility']].to_string(index=False))
    
#     # Filter for liquid options
#     liquid_only = OptionsUtils.filter_liquid_options(
#         formatted,
#         min_volume=150,
#         min_oi=600
#     )
#     print(f"\nLiquid Options (volume >= 150, OI >= 600): {len(liquid_only)} contracts")
    
#     # =========================================================================
#     # Example 7: Complete Workflow - Analyze and Select Strategy
#     # =========================================================================
#     print("\n" + "=" * 80)
#     print("EXAMPLE 7: Complete Workflow - Strategy Selection")
#     print("=" * 80)
    
#     # Step 1: Set up analysis
#     stock_price = 150.0
#     dte = 45
#     iv = 0.35
    
#     print(f"\nAnalyzing Options for:")
#     print(f"  Stock Price: ${stock_price:.2f}")
#     print(f"  DTE: {dte} days")
#     print(f"  Implied Volatility: {iv*100:.0f}%")
    
#     # Step 2: Calculate expected move
#     analyzer = OptionsAnalyzer(current_price=stock_price, days_to_expiration=dte)
#     move = analyzer.get_implied_move(avg_iv=iv)
#     print(f"\nExpected Move (1σ): ${move['lower_bound']:.2f} - ${move['upper_bound']:.2f}")
    
#     # Step 3: Evaluate potential strategies
#     strategies_to_evaluate = [
#         {
#             'name': 'Conservative IC',
#             'max_profit': 400,
#             'max_loss': 600,
#             'prob_profit': 0.75,
#             'days': dte
#         },
#         {
#             'name': 'Aggressive IC',
#             'max_profit': 800,
#             'max_loss': 1200,
#             'prob_profit': 0.60,
#             'days': dte
#         }
#     ]
    
#     comparison = StrategyComparator.compare(strategies_to_evaluate)
#     print("\nStrategy Comparison:")
#     print(comparison[['Strategy', 'Max Profit', 'Max Loss', 'Expected Value', 'Annual ROI%']].to_string(index=False))
    
#     # Step 4: Calculate risk for selected strategy
#     selected_risk = PositionRiskCalculator.iron_condor_risk(
#         put_short_strike=145,
#         put_long_strike=140,
#         call_short_strike=155,
#         call_long_strike=160,
#         total_credit=4.00,
#         quantity=2
#     )
    
#     print(f"\nSelected Strategy Risk Profile:")
#     print(f"  Max Profit: ${selected_risk['max_profit']:,.2f}")
#     print(f"  Max Loss: ${selected_risk['max_loss']:,.2f}")
#     print(f"  Breakeven Range: ${selected_risk['breakeven_lower']:.2f} - ${selected_risk['breakeven_upper']:.2f}")
#     print(f"  P/L Ratio: {selected_risk['pl_ratio']:.2f}")
    
#     print("\n" + "=" * 80)
#     print("Examples Complete!")
#     print("=" * 80)

