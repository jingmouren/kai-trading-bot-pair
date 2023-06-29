import os
import numpy as np
import warnings
import statsmodels.api as sm
import pandas as pd
import datetime
from scipy.stats import zscore
from sklearn import linear_model
from itertools import combinations, permutations
from dateutil.relativedelta import relativedelta

warnings.filterwarnings("ignore")

HOME = '/Users/kq/pairs/'

prices = pd.read_csv(HOME + '/price_history.csv', index_col=0)
start = prices.index[0]
end = prices.index[-1]
bussiness_days_rng =pd.date_range(start, end, freq='BM')


prices = pd.read_csv(HOME + '/price_history.csv', index_col=0)
prices.index = pd.DatetimeIndex(prices.index).date
returns = prices.pct_change().iloc[1:]
nan_filter = (returns.isnull().sum() / len(returns))
universe = list(set(nan_filter[nan_filter < 0.05].index))
log_prices = np.log(prices[universe])
pairs = list(permutations(universe, 2))

adf = pd.read_csv(HOME + '/adf/20220630.csv')
adf = adf[adf.P_VALUE <= 0.05]
adf['NAME'] = adf.ASSET_1 + '-' + adf.ASSET_2
feasible_names = list(adf.NAME)

def fetch_beta(x, y):
    regr = linear_model.LinearRegression()
    x_constant = pd.concat([x, pd.Series([1] * len(x), index=x.index)], axis=1)
    regr.fit(x_constant, y)
    return regr.coef_[0]


def cache_betas(data_list):
    log_prices, adf = data_list
    betas = {}
    for pair in adf[['ASSET_1', 'ASSET_2']].values:
        pair_prices = log_prices[pair].loc[:pd.to_datetime(j)].iloc[-504:]
        try:
            beta = fetch_beta(pair_prices[pair[0]], pair_prices[pair[1]])
            betas['-'.join(pair)] = beta
        except:
            pass
    betas = pd.Series(betas).reset_index().rename(columns={'index': 'NAME', 0: 'BETA'})
    betas.to_csv(HOME + '/betas/{}.csv'.format(log_prices.index[-1].strftime('%Y%m%d')))

def create_spread(x, y):
    regr = linear_model.LinearRegression()
    x_constant = pd.concat([x, pd.Series([1] * len(x), index=x.index)], axis=1)
    regr.fit(x_constant, y)
    beta = regr.coef_[0]
    alpha = regr.intercept_
    return y - (x * beta) - alpha

def cache_spread(log_prices)-> None:
    end_date = log_prices.index[-1].strftime('%Y%m%d')
    pairs = list(permutations(log_prices.columns, 2))
    print('Processing {} pairs with reference date: {}'.format(len(pairs), end_date))
    records, resids = {}, {}
    for pair in pairs:
        try:
            spread = create_spread(x=log_prices[pair[0]], y=log_prices[pair[1]])
            test_result = sm.tsa.stattools.adfuller(spread)
            t_stat, p_value, _, obs, thresholds, _ = test_result
            records[pair] = t_stat, p_value, thresholds
            resids[pair] = spread
        except:
            pass

    info = pd.DataFrame(records).T.reset_index()
    info.columns = ['ASSET_1', 'ASSET_2', 'T_STAT', 'P_VALUE', 'CRITICAL_VALUES']
    info = info.drop('CRITICAL_VALUES', axis=1).sort_values('P_VALUE').reset_index(drop=True)
    info.to_csv(HOME + '/adf/{}.csv'.format(end_date), index=False)

    spreads = pd.DataFrame(resids)
    spreads.columns = spreads.columns.map('-'.join)
    spreads.to_csv(HOME + '/spread/{}.csv'.format(end_date))  
    
from joblib import Parallel, delayed
from multiprocessing import cpu_count


data_list = []
for j in bussiness_days_rng[24:][::-1]:
    adf = pd.read_csv(HOME + '/adf/{}.csv'.format(j.strftime('%Y%m%d')))
    data_list.append([log_prices.loc[:pd.to_datetime(j)].iloc[-504:], adf[adf.P_VALUE <= 0.05]])

start = prices.index[0]
end = prices.index[-1]
bussiness_days_rng =pd.date_range(start, end, freq='BM')
data_list = []
for j in bussiness_days_rng[24:][::-1]:
    data_list.append(log_prices.loc[:pd.to_datetime(j)].iloc[-504:])
    
#res = Parallel(n_jobs=cpu_count() - 1)(delayed(cache_betas)(x) for x in data_list)    
res = Parallel(n_jobs=cpu_count() - 1)(delayed(cache_spread)(x) for x in data_list)
