import pandas as pd
import pandas_ta as ta
import numpy as np
import statsmodels as sm
from statsmodels.regression.rolling import RollingOLS



# Functions
def compute_lag_returns(dframe):
    lags = [1, 3, 6, 12]
    outlier_lim = 0.005
    for lag in lags:
        dframe[f'return_{lag}m'] = (dframe.groupby('Name')['close']
                                    .pct_change(lag)
                                    .pipe(lambda x: x.clip(upper=x.quantile(1 - outlier_lim), lower=x.quantile(outlier_lim)))
                                    .add(1)
                                    .pow(1/lag)
                                    .sub(1))

    return dframe


def compute_atr(stock_data):
    atr = ta.atr(high=stock_data['high'],
                 low=stock_data['low'],
                 close=stock_data['close'],
                 length=14)
    return atr.sub(atr.mean()).div(atr.std())


def compute_macd(close_series):
    macd = ta.macd(close=close_series, length=20).iloc[:, 0]
    return macd.sub(macd.mean()).div(macd.std())


# Setup Dataframe
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
csv_path = 'data/all_stocks_5yr.csv'
df = pd.read_csv(csv_path)
df['date'] = pd.to_datetime(df.date)
df = df.set_index(['date', 'Name'])

# Market % Change
c_series = df.unstack('Name')['close'].resample('ME').mean().stack('Name').to_frame()
mdfm = c_series.groupby('date')[0].mean().to_frame()
mdfm['pct_change'] = (mdfm[0].transform(lambda x: (x - x.shift(1))/x.shift(1))
                      .pipe(lambda y: y.clip(lower=y.quantile(0.005), upper=y.quantile(0.955))))

# Per Day Statistics
df['daily_vol'] = df.groupby(level=1).close.transform(lambda x: x.sub(x.shift(1)) / x.shift(1))
df['rsi'] = df.groupby(level=1).close.transform(lambda x: ta.rsi(x, length=20))
df['bb_low'] = df.groupby(level=1).close.transform(lambda x: ta.bbands(close=x, length=20).iloc[:, 0])
df['bb_mid'] = df.groupby(level=1).close.transform(lambda x: ta.bbands(close=x, length=20).iloc[:, 1])
df['bb_high'] = df.groupby(level=1).close.transform(lambda x: ta.bbands(close=x, length=20).iloc[:, 2])
df['atr'] = df.groupby(level=1, group_keys=False).apply(compute_atr)
df['macd'] = df.groupby(level=1).close.transform(lambda x: compute_macd(x))
df['dollar_volume'] = df.volume.mul(df.close).div(1e6)

# Per Month Statistics
last_cols = [c for c in df.columns if c not in ('open', 'close', 'high', 'low', 'dollar_volume', 'volume')]
dfm = pd.concat([df.unstack('Name')[['dollar_volume', 'close']].resample('ME').mean().stack(future_stack=True),
                 df.unstack('Name')[last_cols].resample('ME').last().stack(future_stack=True)], axis=1).dropna()
dfm['dollar_volume'] = dfm.loc[:, 'dollar_volume'].unstack('Name').rolling(4*12).mean().stack('Name')
dfm['dv_rank'] = dfm.groupby(level=0).dollar_volume.rank(ascending=False)
dfm = dfm.query('dv_rank < 150').drop(['dv_rank', 'dollar_volume'], axis=1)
dfm = dfm.groupby(level='Name', group_keys=False).apply(compute_lag_returns)

# Betas
print(dfm)
# dfm = dfm.merge(mdfm['pct_change'], on='date', how='inner')
# dfm = dfm.rename(columns={'pct_change': 'market_return_1m'})
#-0.003808
#2018-01-31  32.434286          0.095142
#cdf['stock_pct_change'] = cdf.groupby('Name')['close'].transform(lambda x: (x - x.shift(1)) / x.shift(1)).pipe(lambda y: y.clip(lower=y.quantile(0.005), upper=y.quantile(0.995)))
