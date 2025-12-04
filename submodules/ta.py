import talib
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd

class TechnicalAnalysis:

    def __init__(self):
        # NOTE: pass the data at a method level instead of instance variable; for flexability.
        pass

    def get_rsi(self, prices, period=14):
        if isinstance(prices, pd.DataFrame):
            return prices.apply(lambda col: pd.Series(talib.RSI(col.values, timeperiod=period), index=col.index))
        else:
            return pd.Series(talib.RSI(prices.values, timeperiod=period), index=prices.index)

    def get_moving_average(self, prices, period=20, ma_type="SMA"):
        """
        Return moving average (SMA or EMA).
        - If `prices` is a Series -> returns a Series.
        - If `prices` is a DataFrame -> returns a DataFrame with same columns.
        ma_type: "SMA" or "EMA" (case-insensitive).
        """
        ma_type = (ma_type or "SMA").upper()
        if ma_type == "EMA":
            func = lambda arr: talib.EMA(arr, timeperiod=period)
        else:
            func = lambda arr: talib.SMA(arr, timeperiod=period)

        if isinstance(prices, pd.DataFrame):
            return prices.apply(lambda col: pd.Series(func(col.values), index=col.index))
        else:
            return pd.Series(func(prices.values), index=prices.index, name=f"{ma_type.lower()}_{period}")

    def get_bbands(self, prices, period=20, nbdev=2.0, matype=0):
        """
        Return Bollinger Bands.
        - If `prices` is a Series -> returns a DataFrame with columns ['upper','middle','lower'].
        - If `prices` is a DataFrame -> returns a DataFrame with columns named '<col>_upper','<col>_middle','<col>_lower' for each input column.
        matype follows TA-Lib matype parameter (0 default).
        """
        def _bb_for_array(arr):
            upper, middle, lower = talib.BBANDS(arr, timeperiod=period, nbdevup=nbdev, nbdevdn=nbdev, matype=matype)
            return upper, middle, lower

        if isinstance(prices, pd.DataFrame):
            upp = prices.apply(lambda col: pd.Series(_bb_for_array(col.values)[0], index=col.index))
            mid = prices.apply(lambda col: pd.Series(_bb_for_array(col.values)[1], index=col.index))
            low = prices.apply(lambda col: pd.Series(_bb_for_array(col.values)[2], index=col.index))

            upp.columns = [f"{c}_upper" for c in upp.columns]
            mid.columns = [f"{c}_middle" for c in mid.columns]
            low.columns = [f"{c}_lower" for c in low.columns]

            return pd.concat([upp, mid, low], axis=1)
        else:
            upper, middle, lower = _bb_for_array(prices.values)
            return pd.DataFrame({
                'upper': pd.Series(upper, index=prices.index),
                'middle': pd.Series(middle, index=prices.index),
                'lower': pd.Series(lower, index=prices.index)
            })