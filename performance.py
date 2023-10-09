import multiprocess as mp
from multiprocessing import cpu_count
from tools import *
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
        self.quantile_window = 20
        self.styles = ['BASELINE', 'MAX_WEIGHT', 'MAXIMAL', 'APPROX_MAXIMAL']
        self.tau = 0.25
        
    @classmethod
    def fetch_cumulative_returns(cls, returns: pd.Series) -> pd.Series:
        return ((1 + returns.fillna(0)).cumprod() - 1)

    @classmethod
    def fetch_sum_returns(cls, returns: pd.Series) -> pd.Series:
        return returns.fillna(0).cumsum()

    @classmethod    
    def fetch_sharpe(cls, returns: pd.Series, horizon: int = 252) -> float:
        return np.sqrt(horizon) * returns.mean() / returns.std()

    @classmethod    
    def fetch_sortino(cls, returns: pd.Series, horizon: int = 252) -> float:
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
    def run_trade(cls, data: Tuple[str, str, str], bound_window: int = 20) -> pd.DataFrame:

        perf = Performance()
        start, end, style = data
        pnl={}
        cap={}
        optimal_pairs = fetch_optimal_pairs(date=start, style=style)
        for pair in optimal_pairs:
            x, y = pair.split('-')
            data = pd.read_csv(HOME + '/spread/pairs/{}_{}.csv'.format(x,y), index_col=0)

            # Tauhid's method, not used
            data['LB'] = data.SPREAD.rolling(bound_window).quantile(perf.tau, interpolation='midpoint')
            data['UB'] = data.SPREAD.rolling(bound_window).quantile(1-perf.tau, interpolation='midpoint')


            # Subtract median, normalize by IQR
            data['QSCORE'] = (((data.SPREAD - data.rolling(bound_window).SPREAD.median()) / (data.rolling(bound_window).SPREAD.quantile(perf.tau) - data.rolling(bound_window).SPREAD.quantile(1 - perf.tau))))

            # Rolling z-score, not used
            data['ZSCORE'] = (data.SPREAD - data.SPREAD.rolling(bound_window).mean()) / data.SPREAD.rolling(bound_window).std()

            # Scale by sign function and magnitude (second order)
            data['SIGNAL'] = np.sign(data.QSCORE) * np.round(data.QSCORE.abs())  

            # Lag to prevent lookahead bias
            data['SPREAD_POSITION'] = data.SIGNAL.shift()

            # Capital usage based on spread
            data['CAPITAL_USAGE'] = data.SPREAD_POSITION.abs() * (np.exp(data[y]) + (data.BETA * np.exp(data[x])).abs()).fillna(0)

            # PNL calculation based on spread position 
            # The spread is defined as s = y - bx (intercept excluded in calc)
            prices = np.exp(data[[x, y, 'BETA']])
            data['PNL'] = data.SPREAD_POSITION * (prices[y].diff() + (-np.log(prices.BETA) * prices[x].diff())).fillna(0)


            # Isolate month-to-month
            data = data.loc[start: end]

            data.index = pd.DatetimeIndex(data.index)
            pnl[pair] = data.PNL    
            cap[pair] = data.CAPITAL_USAGE

        # Calcualte returns based on dollars generated per capital usage
        return pd.DataFrame(pnl).sum(axis=1) / pd.DataFrame(cap).sum(axis=1)

    @classmethod
    def run_history(cls, style: str = 'MAX_WEIGHT') -> None:

        data_list = []

        # Get available dates
        dates = [pd.to_datetime(j.split('/spread/')[1].split('.')[0]).strftime('%Y-%m-%d') 
                 for j in sorted(glob.glob(HOME + '/spread/202*.csv'))]

        for j in range(len(dates) - 1):
            start, end = dates[j], dates[j+1]
            data_list.append([start, end, style])

        with mp.Pool(cpu_count()) as p:
            output = list(p.map(run_trade,data_list))
            p.close()
            p.terminate()    
        returns = pd.concat(output, axis=0).sort_index()
        perf.simple_report(returns)

    @classmethod
    def simple_report(cls, returns: pd.Series) -> pd.DataFrame:
        perf = Performance()
        sharpe = perf.fetch_sharpe(returns=returns, horizon=perf.horizon)
        sortino = perf.fetch_sortino(returns=returns, horizon=perf.horizon)
        cumret = perf.fetch_cumulative_returns(returns=returns).iloc[-1]
        simpret = perf.fetch_sum_returns(returns=returns).iloc[-1]
        return pd.DataFrame([sharpe, sortino, cumret, simpret], 
                     index=['SHARPE', 'SORTINO', 'CUMULATIVE_RETURN', 'SIMPLE_RETURN']).rename(columns={0: 'PERFORMANCE'})
