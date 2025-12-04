"""
Fixed Income Analytics Library
Comprehensive toolkit for yield curve analysis, bond pricing, and fixed income modeling
"""

from typing import Union, List, Tuple, Optional, Dict
from dataclasses import dataclass
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, CubicSpline
import scipy.optimize as sco
import requests
from bs4 import BeautifulSoup

np.seterr(divide='ignore', invalid='ignore')


# =============================================================================
# CORE UTILITY FUNCTIONS
# =============================================================================

def discrete_df(rate: Union[float, np.ndarray], time: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Discrete compounding discount factor: DF = 1 / (1 + r/100)^t"""
    return 1 / ((1 + rate / 100) ** time)

def continuous_df(rate: Union[float, np.ndarray], time: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Continuous compounding discount factor: DF = e^(-r*t/100)"""
    return np.exp(-rate / 100 * time)

def discrete_rate(df: Union[float, np.ndarray], time: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Extract discrete rate from discount factor"""
    return ((1 / df) ** (1 / time) - 1) * 100

def continuous_rate(df: Union[float, np.ndarray], time: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Extract continuous rate from discount factor"""
    return (np.log(1 / df) / time) * 100


# =============================================================================
# CURVE CLASS
# =============================================================================

class Curve:
    """Base yield curve class with maturities and rates"""
    
    def __init__(self, maturities: Union[np.ndarray, list], rates: Union[np.ndarray, list]) -> None:
        self._t = np.array(maturities) if isinstance(maturities, list) else maturities
        self._rt = np.array(rates) if isinstance(rates, list) else rates
    
    @property
    def get_time(self) -> np.ndarray:
        return self._t
    
    @property
    def get_rate(self) -> np.ndarray:
        return self._rt
    
    def set_time(self, t: Union[np.ndarray, list]) -> None:
        self._t = np.array(t) if isinstance(t, list) else t
    
    def set_rate(self, rt: Union[np.ndarray, list]) -> None:
        self._rt = np.array(rt) if isinstance(rt, list) else rt
    
    def plot_curve(self, title: str = "Yield Curve") -> plt.Figure:
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.suptitle(title, fontsize=14, fontweight='bold')
        xi = list(range(len(self.get_time)))
        ax.plot(xi, self.get_rate, marker='o', label="Yield", linewidth=2, color="navy")
        ax.set_xticks(xi)
        ax.set_xticklabels([f"{t:.2f}" for t in self.get_time])
        ax.set_xlabel('Maturity (Years)')
        ax.set_ylabel('Yield (%)')
        ax.grid(True, alpha=0.3)
        ax.fill_between(xi, self.get_rate, alpha=0.2, color="lightskyblue")
        ax.legend()
        plt.tight_layout()
        plt.show()
        return fig


# =============================================================================
# BOOTSTRAP YIELD CURVE
# =============================================================================

@dataclass
class BondInstrument:
    """Bond instrument for bootstrapping"""
    par: float
    maturity: float
    coupon_rate: float
    price: float
    compounding_freq: int = 2

class BootstrapYieldCurve:
    """
    Bootstrap zero rates from bonds.
    
    Example:
        bootstrap = BootstrapYieldCurve()
        bootstrap.add_instrument(100, 0.25, 0, 97.5)  # 3M zero
        bootstrap.add_instrument(100, 0.5, 0, 94.9)   # 6M zero
        bootstrap.add_instrument(100, 1.5, 8, 96.0, 2)  # 1.5Y coupon bond
        zero_rates = bootstrap.get_zero_rates()
    """
    
    def __init__(self):
        self.zero_rates: Dict[float, float] = {}
        self.instruments: Dict[float, BondInstrument] = {}
    
    def add_instrument(self, par: float, maturity: float, coupon_rate: float, 
                      price: float, compounding_freq: int = 2) -> None:
        """Add bond to bootstrap set"""
        self.instruments[maturity] = BondInstrument(par, maturity, coupon_rate, price, compounding_freq)
    
    def get_maturities(self) -> List[float]:
        return sorted(self.instruments.keys())
    
    def get_zero_rates(self) -> List[float]:
        """Calculate and return zero rates (in %)"""
        self._bootstrap_zero_coupons()
        self._bootstrap_coupon_bonds()
        return [self.zero_rates[t] * 100 for t in self.get_maturities()]
    
    def _bootstrap_zero_coupons(self) -> None:
        for maturity, inst in self.instruments.items():
            if inst.coupon_rate == 0:
                self.zero_rates[maturity] = math.log(inst.par / inst.price) / inst.maturity
    
    def _bootstrap_coupon_bonds(self) -> None:
        for maturity in self.get_maturities():
            inst = self.instruments[maturity]
            if inst.coupon_rate != 0:
                self.zero_rates[maturity] = self._calculate_bond_spot_rate(maturity, inst)
    
    def _calculate_bond_spot_rate(self, maturity: float, inst: BondInstrument) -> float:
        periods = int(maturity * inst.compounding_freq)
        coupon_payment = inst.coupon_rate / inst.compounding_freq
        remaining_value = inst.price
        
        for i in range(periods - 1):
            t = (i + 1) / inst.compounding_freq
            spot_rate = self.zero_rates[t]
            remaining_value -= coupon_payment * math.exp(-spot_rate * t)
        
        final_cf = inst.par + coupon_payment
        return -math.log(remaining_value / final_cf) / maturity
    
    def get_curve(self) -> Curve:
        """Return bootstrapped curve"""
        return Curve(self.get_maturities(), self.get_zero_rates())


# =============================================================================
# LINEAR INTERPOLATED CURVE
# =============================================================================

class LinearCurve:
    """
    Linear interpolation for yield curves.
    
    Example:
        curve = Curve([0.25, 0.5, 1, 2, 5], [2.5, 2.7, 3.0, 3.5, 4.0])
        linear = LinearCurve(curve)
        rate_3y = linear.d_rate(3.0)
        fwd_2y5y = linear.forward(2, 5)
    """
    
    def __init__(self, curve: Curve):
        self._curve = curve
        self._func_rate = interp1d(curve.get_time, curve.get_rate, 
                                   kind='linear', fill_value='extrapolate')
    
    def d_rate(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Get interpolated rate at maturity t"""
        return self._func_rate(t)
    
    def df_t(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Discount factor at maturity t"""
        return discrete_df(self.d_rate(t), t)
    
    def forward(self, t1: Union[float, np.ndarray], t2: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Forward rate from t1 to t2"""
        return ((self.d_rate(t2) * t2) - (self.d_rate(t1) * t1)) / (t2 - t1)
    
    def create_curve(self, t_array: Union[np.ndarray, list]) -> Curve:
        """Create new curve with interpolated rates"""
        return Curve(t_array, self.d_rate(t_array))


# =============================================================================
# NELSON-SIEGEL MODEL
# =============================================================================

class NelsonSiegel:
    """
    Nelson-Siegel yield curve model: r(t) = β0 + β1*f1(t) + β2*f2(t)
    
    Parameters:
        β0 (beta0): Level - long-term rate
        β1 (beta1): Slope - short vs long differential
        β2 (beta2): Curvature - medium-term hump
        τ  (tau):   Decay factor
    
    Example:
        ns = NelsonSiegel(5.0, -2.0, 3.0, 2.0)
        rate_5y = ns.d_rate(5.0)
        
        # Calibrate to data
        curve = Curve([0.5, 1, 2, 5, 10], [2.5, 3.0, 3.5, 4.0, 4.2])
        ns.calibrate(curve)
    """
    
    def __init__(self, beta0: float, beta1: float, beta2: float, tau: float) -> None:
        if beta0 < 0 or tau <= 0:
            raise ValueError("beta0 must be positive and tau must be > 0")
        self.beta0 = beta0
        self.beta1 = beta1
        self.beta2 = beta2
        self.tau = tau
        self.attr_list = ["beta0", "beta1", "beta2", "tau"]
    
    def _time_decay(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        t_arr = np.array(t, dtype=np.longdouble)
        return self.beta1 * ((1 - np.exp(-t_arr / self.tau)) / (t_arr / self.tau))
    
    def _hump(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        t_arr = np.array(t, dtype=np.longdouble)
        factor1 = (1 - np.exp(-t_arr / self.tau)) / (t_arr / self.tau)
        return self.beta2 * (factor1 - np.exp(-t_arr / self.tau))
    
    def d_rate(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Calculate yield at maturity t"""
        return self.beta0 + self._time_decay(t) + self._hump(t)
    
    def df_t(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return discrete_df(self.d_rate(t), t)
    
    def cdf_t(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return continuous_df(self.d_rate(t), t)
    
    def forward_rate(self, t1: Union[float, np.ndarray], t2: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return ((self.d_rate(t2) * t2) - (self.d_rate(t1) * t1)) / (t2 - t1)
    
    def calibrate(self, curve: Curve, verbose: bool = True) -> sco.OptimizeResult:
        """Calibrate model to observed curve"""
        def objective(x):
            try:
                ns = NelsonSiegel(x[0], x[1], x[2], x[3])
                predicted = np.array([ns.d_rate(t) for t in curve.get_time])
                return np.sum((predicted - curve.get_rate) ** 2)
            except:
                return 1e10
        
        x0 = np.array([np.mean(curve.get_rate), 0, 0, 2.0])
        bounds = ((1e-6, np.inf), (-30, 30), (-30, 30), (1e-6, 30))
        result = sco.minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
        
        for i, attr in enumerate(self.attr_list):
            setattr(self, attr, result.x[i])
        
        if verbose:
            print(f"Nelson-Siegel Calibration: SSE={result.fun:.4f}, Status={result.message}")
        return result
    
    def plot_model(self, max_t: float = 30) -> plt.Figure:
        """Plot full model and components"""
        t = np.linspace(0.1, max_t, 500)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Nelson-Siegel Model', fontsize=16, fontweight='bold')
        
        axes[0, 0].plot(t, self.d_rate(t), 'navy', linewidth=2)
        axes[0, 0].set_title('Full Yield Curve')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].axhline(self.beta0, color='green', linewidth=2, label=f'β0={self.beta0:.3f}')
        axes[0, 1].set_title('Level (β0)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(t, self._time_decay(t), 'red', linewidth=2, label=f'β1={self.beta1:.3f}')
        axes[1, 0].set_title('Slope Component')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(t, self._hump(t), 'purple', linewidth=2, label=f'β2={self.beta2:.3f}')
        axes[1, 1].set_title('Curvature Component')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        return fig
    
    def plot_calibrated(self, curve: Curve) -> plt.Figure:
        """Plot fitted vs observed"""
        fig, ax = plt.subplots(figsize=(12, 6))
        t_fine = np.linspace(curve.get_time.min(), curve.get_time.max(), 200)
        ax.plot(t_fine, self.d_rate(t_fine), 'seagreen', linewidth=2, label='Fitted')
        ax.scatter(curve.get_time, curve.get_rate, s=100, color='chocolate', 
                  label='Observed', zorder=5, edgecolors='black')
        ax.set_xlabel('Maturity (Years)')
        ax.set_ylabel('Yield (%)')
        ax.set_title('Nelson-Siegel Fit')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.show()
        return fig


# =============================================================================
# TREASURY PAR CURVE DATA
# =============================================================================

class TreasuryParCurve:
    """
    Fetch US Treasury par curve data from Treasury.gov
    
    Example:
        treasury = TreasuryParCurve(years=[2024])
        df = treasury.df  # Full dataset
        latest = treasury.get_latest_curve()
        curve_june = treasury.get_curve_on_date('2024-06-30')
    """
    
    MATURITIES = ['1 Month', '2 Month', '3 Month', '6 Month', '1 Year', 
                  '2 Year', '3 Year', '5 Year', '7 Year', '10 Year', '20 Year', '30 Year']
    
    MATURITY_MAP = {'1 Month': 1/12, '2 Month': 2/12, '3 Month': 0.25, '6 Month': 0.5,
                    '1 Year': 1, '2 Year': 2, '3 Year': 3, '5 Year': 5, '7 Year': 7,
                    '10 Year': 10, '20 Year': 20, '30 Year': 30}
    
    def __init__(self, years: List[int] = [2024]):
        self.years = years
        self.df = self._fetch_data()
    
    def _fetch_data(self) -> pd.DataFrame:
        """Scrape Treasury data"""
        xml_tags = {'Date': 'd:NEW_DATE', '1 Month': 'd:BC_1MONTH', '2 Month': 'd:BC_2MONTH',
                    '3 Month': 'd:BC_3MONTH', '6 Month': 'd:BC_6MONTH', '1 Year': 'd:BC_1YEAR',
                    '2 Year': 'd:BC_2YEAR', '3 Year': 'd:BC_3YEAR', '5 Year': 'd:BC_5YEAR',
                    '7 Year': 'd:BC_7YEAR', '10 Year': 'd:BC_10YEAR', '20 Year': 'd:BC_20YEAR',
                    '30 Year': 'd:BC_30YEAR'}
        
        all_data = []
        for year in self.years:
            url = f'https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml?data=daily_treasury_yield_curve&field_tdr_date_value={year}'
            try:
                response = requests.get(url, timeout=30)
                soup = BeautifulSoup(response.content, "lxml-xml")
                for entry in soup.find_all('m:properties'):
                    row = {col: entry.find(tag).text if entry.find(tag) else None 
                           for col, tag in xml_tags.items()}
                    if row.get('Date'):
                        all_data.append(row)
            except Exception as e:
                print(f"Error fetching {year}: {e}")
        
        df = pd.DataFrame(all_data)
        df['Date'] = pd.to_datetime(df['Date'])
        for col in self.MATURITIES:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        return df
    
    def get_latest_curve(self) -> Curve:
        """Get most recent yield curve"""
        latest = self.df.iloc[-1]
        mats, rates = [], []
        for col in self.MATURITIES:
            if pd.notna(latest[col]):
                mats.append(self.MATURITY_MAP[col])
                rates.append(latest[col])
        return Curve(mats, rates)
    
    def get_curve_on_date(self, date: str) -> Curve:
        """Get curve for specific date (YYYY-MM-DD)"""
        dt = pd.to_datetime(date)
        if dt not in self.df.index:
            idx = self.df.index.get_indexer([dt], method='nearest')[0]
            dt = self.df.index[idx]
        row = self.df.loc[dt]
        mats, rates = [], []
        for col in self.MATURITIES:
            if pd.notna(row[col]):
                mats.append(self.MATURITY_MAP[col])
                rates.append(row[col])
        return Curve(mats, rates)


    
# Bootstrap example
# bootstrap = BootstrapYieldCurve()
# bootstrap.add_instrument(100, 0.25, 0, 97.5)
# curve = bootstrap.get_curve()
# curve.plot_curve()

# # Nelson-Siegel calibration
# ns = NelsonSiegel(5.0, -2.0, 3.0, 2.0)
# ns.calibrate(observed_curve)
# ns.plot_model()

# # Treasury data
# treasury = TreasuryParCurve(years=[2024])
# latest = treasury.get_latest_curve()