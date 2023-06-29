from utils import *

__author__ = 'kq'


def fetch_beta(x, y):
    """

    :param x:
    :param y:
    :return:
    """
    regr = linear_model.LinearRegression()
    x_constant = pd.concat([x, pd.Series([1] * len(x), index=x.index)], axis=1)
    regr.fit(x_constant, y)
    return regr.coef_[0]


def cache_betas(data_list):
    """

    :param data_list:
    :return:
    """
    log_prices, adf = data_list
    betas = {}
    for pair in adf[['ASSET_1', 'ASSET_2']].values:
        pair_prices = log_prices[pair].loc[:pd.to_datetime(j)].iloc[-504:]
        beta = fetch_beta(pair_prices[pair[0]], pair_prices[pair[1]])
        betas['-'.join(pair)] = beta
    betas = pd.Series(betas).reset_index().rename(columns={'index': 'NAME', 0: 'BETA'})
    betas.to_csv(HOME + '/betas/{}.csv'.format(j.strftime('%Y%m%d')), index=False)


def create_spread(x, y):
    """

    :param x:
    :param y:
    :return:
    """
    regr = linear_model.LinearRegression()
    x_constant = pd.concat([x, pd.Series([1] * len(x), index=x.index)], axis=1)
    regr.fit(x_constant, y)
    beta = regr.coef_[0]
    alpha = regr.intercept_
    return y - (x * beta) - alpha


def cache_spread(log_prices) -> None:
    """

    :param log_prices:
    :return:
    """
    end_date = log_prices.index[-1].strftime('%Y%m%d')
    pairs = list(combinations(log_prices.columns, 2))
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





