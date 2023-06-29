from utils import *

__author__ = 'kq'

class Trading:

    @classmethod
    def zscore_threshold(cls, zscores: pd.DataFrame, threshold: float = 2) -> pd.DataFrame:
        z_sell = (zscores >= threshold) * -1
        z_buy = (zscores <= -threshold) * 1
        return (z_buy + z_sell).shift()

    @classmethod
    def lag_signal(cls, signal: pd.DataFrame) -> pd.DataFrame:
        return signal.shift()

    @classmethod
    def fetch_turnover(cls, data: pd.DataFrame) -> pd.DataFrame:
        return data.diff().abs().sum(axis=1)
