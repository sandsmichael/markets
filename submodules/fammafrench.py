import pandas as pd
import pandas_datareader.data as web

def get_fama_french(factors=3, frequency='monthly'):
    """
    Retrieve Fama-French factors from Kenneth French's website.
    """
    if factors == 3:
        if frequency == 'daily':
            code = 'F-F_Research_Data_Factors_daily'
        elif frequency == 'monthly':
            code = 'F-F_Research_Data_Factors'
        else:
            raise ValueError("frequency must be 'daily' or 'monthly'")
    elif factors == 5:
        if frequency == 'monthly':
            code = 'F-F_Research_Data_5_Factors_2x3'
        else:
            raise ValueError("5-factor data only available for 'monthly' frequency")
    else:
        raise ValueError("factors must be 3 or 5")
    ff = web.DataReader(code, 'famafrench')
    return ff[0]

def get_momentum(frequency='monthly'):
    """
    Retrieve the Fama-French momentum factor (UMD).
    """
    if frequency == 'monthly':
        code = 'F-F_Momentum_Factor'
    elif frequency == 'daily':
        code = 'F-F_Momentum_Factor_daily'
    else:
        raise ValueError("frequency must be 'daily' or 'monthly'")
    mom = web.DataReader(code, 'famafrench')[0]
    return mom[['Mom']].rename(columns={'Mom': 'MOM'})

def get_quality(frequency='monthly'):
    """
    Retrieve the Fama-French quality factor (RMW).
    Only available in 5-factor monthly data.
    """
    if frequency != 'monthly':
        raise ValueError("Quality factor only available for 'monthly' frequency")
    code = 'F-F_Research_Data_5_Factors_2x3'
    ff = web.DataReader(code, 'famafrench')[0]
    return ff[['RMW']].rename(columns={'RMW': 'Quality'})

def get_investment(frequency='monthly'):
    """
    Retrieve the Fama-French investment factor (CMA).
    Only available in 5-factor monthly data.
    """
    if frequency != 'monthly':
        raise ValueError("Investment factor only available for 'monthly' frequency")
    code = 'F-F_Research_Data_5_Factors_2x3'
    ff = web.DataReader(code, 'famafrench')[0]
    return ff[['CMA']].rename(columns={'CMA': 'Investment'})

def get_liquidity():
    """
    Retrieve the Pastor-Stambaugh liquidity factor (monthly).
    """
    code = 'Liquidity_Factor'
    try:
        liq = web.DataReader(code, 'famafrench')[0]
        return liq.rename(columns={'LIQ': 'Liquidity'})
    except Exception:
        raise NotImplementedError("Liquidity factor not available via pandas_datareader for Kenneth French's library.")

def get_bab(frequency='monthly'):
    """
    Retrieve the Betting Against Beta (BAB) factor (monthly).
    """
    code = 'Betting_Against_Beta_Factor'
    try:
        bab = web.DataReader(code, 'famafrench')[0]
        return bab.rename(columns={'BAB': 'BettingAgainstBeta'})
    except Exception:
        raise NotImplementedError("BAB factor not available via pandas_datareader for Kenneth French's library.")

def get_low_volatility():
    """
    Retrieve the low volatility factor (monthly).
    """
    code = 'Low_Volatility_Factor'
    try:
        lowvol = web.DataReader(code, 'famafrench')[0]
        return lowvol.rename(columns={'LowVol': 'LowVolatility'})
    except Exception:
        raise NotImplementedError("Low volatility factor not available via pandas_datareader for Kenneth French's library.")

def get_profitability(frequency='monthly'):
    """
    Retrieve the profitability factor (Gross Profitability).
    Only available as monthly data.
    """
    code = 'Gross_Profitability_Factor'
    try:
        prof = web.DataReader(code, 'famafrench')[0]
        return prof.rename(columns={'Profit': 'Profitability'})
    except Exception:
        raise NotImplementedError("Profitability factor not available via pandas_datareader for Kenneth French's library.")

def get_size_microcap(frequency='monthly'):
    """
    Retrieve the microcap size factor (SMB Micro).
    Only available as monthly data.
    """
    code = 'SMB_Micro_Factor'
    try:
        smb_micro = web.DataReader(code, 'famafrench')[0]
        return smb_micro.rename(columns={'SMBmicro': 'SMB_Microcap'})
    except Exception:
        raise NotImplementedError("SMB Microcap factor not available via pandas_datareader for Kenneth French's library.")

def get_all_factors(factors=5, frequency='monthly'):
    """
    Retrieve Fama-French factors plus popular extensions (momentum, quality, investment, liquidity, BAB, low volatility, profitability, SMB microcap).

    Returns:
        pd.DataFrame: DataFrame with all available factors merged on date index
    """
    dfs = []
    ff = get_fama_french(factors=factors, frequency=frequency)
    dfs.append(ff)

    # Momentum
    try:
        mom = get_momentum(frequency=frequency)
        dfs.append(mom)
    except Exception:
        pass

    # # Quality (RMW)
    # try:
    #     qual = get_quality(frequency=frequency)
    #     dfs.append(qual)
    # except Exception:
    #     pass

    # # Investment (CMA)
    # try:
    #     invest = get_investment(frequency=frequency)
    #     dfs.append(invest)
    # except Exception:
    #     pass

    # Liquidity
    try:
        liq = get_liquidity()
        dfs.append(liq)
    except Exception:
        pass

    # Betting Against Beta (BAB)
    try:
        bab = get_bab(frequency=frequency)
        dfs.append(bab)
    except Exception:
        pass

    # Low Volatility
    try:
        lowvol = get_low_volatility()
        dfs.append(lowvol)
    except Exception:
        pass

    # Profitability
    try:
        prof = get_profitability(frequency=frequency)
        dfs.append(prof)
    except Exception:
        pass

    # SMB Microcap
    try:
        smb_micro = get_size_microcap(frequency=frequency)
        dfs.append(smb_micro)
    except Exception:
        pass

    # Merge all on index (date)
    out = dfs[0]
    for df in dfs[1:]:
        out = out.join(df, how='outer')

    return out