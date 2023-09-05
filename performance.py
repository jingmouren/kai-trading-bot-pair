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


class Performance:
    
    def __init__(self):
        
        self.horizon = 252
        self.styles = ['BASELINE', 'MAX_WEIGHT', 'MAXIMAL', 'APPROX_MAXIMAL']
        
    @classmethod
    def fetch_cumulative_returns(cls, returns: pd.Series) -> pd.Series:
        return ((1 + returns.fillna(0)).cumprod() - 1)

    @classmethod
    def fetch_sum_returns(cls, returns: pd.Series) -> pd.Series:
        return returns.fillna(0).cumsum()

    @classmethod    
    def fetch_sharpe(cls, returns: pd.Series, horizon: float) -> float:
        return np.sqrt(horizon) * returns.mean() / returns.std()

    @classmethod    
    def fetch_sortino(cls, returns: pd.Series, horizon: float) -> float:
        return np.sqrt(horizon) * returns.mean() / returns[returns < 0].std()        

    @classmethod
    def trade_report(cls) -> None:
        perf = Performance()
        sharpe, sortino, descriptions, corr, cumrets, simplerets = {}, {},{}, {},{}, {}
        for style in perf.styles:
            
            # Read cached files
            pnl = pd.read_csv(HOME +'/aggregate/pnl/{}.csv'.format(style.lower()), index_col=0).sum(1)
            capital = pd.read_csv(HOME +'/aggregate/capital/{}.csv'.format(style.lower()), index_col=0).sum(1)
            
            # Daily profit/loss divided by capital usage
            returns = pnl.div(capital, axis=0).sort_index()
            returns.index = pd.DatetimeIndex(returns.index)
            
            # Hash stats
            corr[style] = returns
            sharpe[style] = perf.fetch_sharpe(returns=returns, horizon=perf.horizon)
            sortino[style] = perf.fetch_sortino(returns=returns, horizon=perf.horizon)
            descriptions[style] = returns.describe()
            cumulative_returns = perf.fetch_cumulative_returns(returns=returns)
            simple_returns = perf.fetch_sum_returns(returns=returns)
            simplerets[style] = simple_returns.iloc[-1]
            cumrets[style] = cumulative_returns.iloc[-1]
            
            # Basic plotting, improve later
            sns.lineplot(cumulative_returns.index, 
                         100 * cumulative_returns.values, label=style)
            plt.xlabel('DATE')
            plt.ylabel('% RETURN')
            plt.xticks(rotation='45')

        # Combine and display
        returns_summary = pd.concat(descriptions, axis=1)
        returns_summary  = pd.concat([returns_summary, pd.Series(cumrets).to_frame().rename(columns={0: 'GEOMETRIC'}).T], axis=0)
        returns_summary  = pd.concat([returns_summary, pd.Series(simplerets).to_frame().rename(columns={0: 'ARITHMETIC'}).T], axis=0)
        returns_summary.iloc[1:] *= 100
        returns_summary  = pd.concat([returns_summary, pd.Series(sharpe).to_frame().rename(columns={0: 'SHARPE'}).T], axis=0)
        returns_summary  = pd.concat([returns_summary, pd.Series(sortino).to_frame().rename(columns={0: 'SORTINO'}).T], axis=0)
        returns_summary = returns_summary.round(2)
        display(returns_summary.iloc[1:].style.background_gradient(axis=1).format(precision=2))    
        
    @classmethod        
    def fetch_performance(cls, style: str = 'MAX_WEIGHT', threshold: float = 2) -> List[pd.DataFrame]:

        # Get available dates
        dates = [pd.to_datetime(j.split('/spread/')[1].split('.')[0]).strftime('%Y-%m-%d') 
                 for j in sorted(glob.glob(HOME + '/spread/202*.csv'))]

        pnl, pos, cap, conc = {}, {}, {}, {}
        for j in range(len(dates) - 1):
            start, end = dates[j], dates[j+1]

            # Fetch pairs for given start date and style
            optimal_pairs = fetch_optimal_pairs(date=start, style=style)
            for pair in optimal_pairs:
                x, y = pair.split('-')
                data = pd.read_csv(HOME + '/spread/pairs/{}_{}.csv'.format(x,y), index_col=0)

                # Z-score the spread and sell +ksigma, buy -ksigma
                data['ZSCORE'] = zscore(data.SPREAD, nan_policy='omit')
                data['SIGNAL'] = np.where(data.ZSCORE >= threshold, -1, 
                                          np.where(data.ZSCORE <= -threshold, 1, 0))

                # Lag to prevent lookahead bias
                data['SPREAD_POSITION'] = data.SIGNAL.shift()

                # Capital usage based on spread
                data['CAPITAL_USAGE'] = data.SPREAD_POSITION.abs() * (np.exp(data[y]) + (data.BETA * np.exp(data[x])).abs()).fillna(0)

                # PNL calculation based on spread position (-1, 0, 1)
                # The spread is defined as s = y - bx (intercept excluded in calc)
                data['PNL'] = data.SPREAD_POSITION * (np.exp(data[y]).diff() - (data.BETA * np.exp(data[x]).diff())).fillna(0)

                # Isolate month-to-month
                data = data.loc[start: end]
                
                # Hash
                pnl[pair] = data.PNL
                pos[pair] = data.SPREAD_POSITION
                cap[pair] = data.CAPITAL_USAGE

        # Profit over time
        pnl = pd.concat(pnl, axis=1)
        pnl.index = pd.DatetimeIndex(pnl.index).date

        # Capital usage over time
        cap = pd.concat(cap, axis=1)
        cap.index = pd.DatetimeIndex(cap.index).date

        # Turnover is defined as the daily cross-sectional sum of absolute differences
        turnover = pd.concat(pos, axis=1).fillna(0).diff().abs().sum(1)
        turnover.index = pd.DatetimeIndex(turnover.index).date

        # Concentration over time
        stack = pd.concat(pos, axis=1).stack().reset_index()
        stack.columns = ['DATE', 'PAIR', 'POSITION']
        stack = stack[stack.POSITION != 0].fillna(0)
        stack = pd.concat([stack, pd.DataFrame(stack.PAIR.str.split('-', n=1, expand=True), index=stack.index)], axis=1).rename(columns={0: 'ASSET_1', 1: 'ASSET_2'})
        concentration = stack.groupby(['DATE', 'ASSET_1']).POSITION.sum() + stack.groupby(['DATE', 'ASSET_2']).POSITION.sum()

        return pnl, cap, turnover, concentration        
