import pandas as pd
import numpy as np
from backtesting3 import run_backtest


def crossover(data, col1, col2, i): 
    if data[col1].iloc[i] > data[col2].iloc[i] and data[col1].iloc[i-1] < data[col2].iloc[i-1]:
        return True
    else:
        return False

def sma(data, period, column='Close', new_column_name=None):
    """
    Calculate Simple Moving Average
    
    Parameters:
    data (pandas.DataFrame): DataFrame with price data
    period (int): Period for SMA calculation
    column (str): Column name to use for calculation
    new_column_name (str): Name for the new column, defaults to f'SMA_{period}'
    """
    if new_column_name is None:
        new_column_name = f'SMA_{period}'
    
    data[new_column_name] = data[column].rolling(window=period).mean()

def ema(data, period, column='Close', new_column_name=None):
    """
    Calculate Exponential Moving Average
    
    Parameters:
    data (pandas.DataFrame): DataFrame with price data
    period (int): Period for EMA calculation
    column (str): Column name to use for calculation
    new_column_name (str): Name for the new column, defaults to f'EMA_{period}'
    """
    if new_column_name is None:
        new_column_name = f'EMA_{period}'
    
    data[new_column_name] = data[column].ewm(span=period, adjust=False).mean()

def atr(data, period=14, use_ha=False, new_column_name=None):
    """
    Calculate Average True Range (ATR)
    
    Parameters:
    data (pandas.DataFrame): DataFrame with OHLC data
    period (int): Period for ATR calculation
    use_ha (bool): Whether to use Heikin-Ashi candles
    new_column_name (str): Name for the new column, defaults to f'ATR_{period}'
    """
    if new_column_name is None:
        prefix = 'HA_' if use_ha else ''
        new_column_name = f'{prefix}ATR_{period}'
    
    # Select appropriate columns based on whether to use Heikin-Ashi
    high_col = 'ha_high' if use_ha else 'High'
    low_col = 'ha_low' if use_ha else 'Low'
    close_col = 'ha_close' if use_ha else 'Close'
    
    high = data[high_col]
    low = data[low_col]
    close = data[close_col]
    
    # Calculate True Range
    data['TR'] = np.maximum(
        np.maximum(
            high - low,
            np.abs(high - close.shift(1))
        ),
        np.abs(low - close.shift(1))
    )
    
    # Calculate ATR
    data[new_column_name] = data['TR'].rolling(window=period).mean()
    
    # Clean up temporary columns
    data.drop(['TR'], axis=1, inplace=True)



def adx(data, period=14, use_ha=False, new_column_prefix=None):
    """
    Calculate Average Directional Index (ADX)
    
    Parameters:
    data (pandas.DataFrame): DataFrame with OHLC data
    period (int): Period for ADX calculation
    use_ha (bool): Whether to use Heikin-Ashi candles
    new_column_prefix (str): Prefix for new columns
    """
    if new_column_prefix is None:
        prefix = 'HA_' if use_ha else ''
        new_column_prefix = f'{prefix}ADX'
    
    # Select appropriate columns based on whether to use Heikin-Ashi
    high_col = 'ha_high' if use_ha else 'High'
    low_col = 'ha_low' if use_ha else 'Low'
    close_col = 'ha_close' if use_ha else 'Close'
    
    high = data[high_col]
    low = data[low_col]
    close = data[close_col]
    
    # Calculate True Range
    data['TR'] = np.maximum(
        np.maximum(
            high - low,
            np.abs(high - close.shift(1))
        ),
        np.abs(low - close.shift(1))
    )
    
    # Calculate Directional Movement
    data['Plus_DM'] = np.where(
        (high - high.shift(1)) > (low.shift(1) - low),
        np.maximum(high - high.shift(1), 0),
        0
    )
    data['Minus_DM'] = np.where(
        (low.shift(1) - low) > (high - high.shift(1)),
        np.maximum(low.shift(1) - low, 0),
        0
    )
    
    # Calculate smoothed TR, +DM, -DM using Wilder's smoothing technique
    atr_column = f'{new_column_prefix}_ATR_{period}'
    data[atr_column] = data['TR'].rolling(window=period).mean()
    data['Smooth_Plus_DM'] = data['Plus_DM'].rolling(window=period).mean()
    data['Smooth_Minus_DM'] = data['Minus_DM'].rolling(window=period).mean()
    
    # Calculate +DI and -DI
    data[f'{new_column_prefix}_Plus_DI'] = 100 * data['Smooth_Plus_DM'] / data[atr_column]
    data[f'{new_column_prefix}_Minus_DI'] = 100 * data['Smooth_Minus_DM'] / data[atr_column]
    
    # Calculate DX
    data['DX'] = 100 * np.abs(data[f'{new_column_prefix}_Plus_DI'] - data[f'{new_column_prefix}_Minus_DI']) / \
                (data[f'{new_column_prefix}_Plus_DI'] + data[f'{new_column_prefix}_Minus_DI'])
    
    # Calculate ADX
    data[f'{new_column_prefix}_{period}'] = data['DX'].rolling(window=period).mean()
    
    # Clean up temporary columns
    data.drop(['TR', 'Plus_DM', 'Minus_DM', 'Smooth_Plus_DM', 'Smooth_Minus_DM', 'DX', atr_column], axis=1, inplace=True)


def heikin_ashi(data):
    """
    Add Heikin-Ashi candle columns to the dataframe
    
    Parameters:
    data (pandas.DataFrame): DataFrame with OHLC data
    """
    # Calculate Heikin-Ashi Close
    data['ha_close'] = (data['Open'] + data['High'] + data['Low'] + data['Close']) / 4
    
    # Calculate Heikin-Ashi Open - first value is same as original open
    data['ha_open'] = data['Open'].copy()
    for i in range(1, len(data)):
        data.at[data.index[i], 'ha_open'] = (data['ha_open'].iloc[i-1] + data['ha_close'].iloc[i-1]) / 2
    
    # Calculate Heikin-Ashi High & Low
    data['ha_high'] = data.apply(lambda x: max(x['ha_open'], x['ha_close'], x['High']), axis=1)
    data['ha_low'] = data.apply(lambda x: min(x['ha_open'], x['ha_close'], x['Low']), axis=1)



def process_data(data):
    # Drop rows with zero volume
    data.drop(data[data['Volume'] == 0].index, inplace=True)
    heikin_ashi(data)
    adx(data, period=14, use_ha=True)

    atr(data)

    data.to_csv('binus.csv', index=False)
    print(data.head())

def strat(data):
    #data.dropna(inplace=True)
    signal = [0] * len(data)
    stoploss_hits = [0] * len(data)
    sum = 0
    stoploss = 0
    ## logic here

    for i in range(50, len(data)):
        if sum == 0:

            if data['HA_ADX_Plus_DI'].iloc[i] > data['HA_ADX_Minus_DI'].iloc[i]:
                if data['HA_ADX_14'].iloc[i] > 30:
                    # if data['RSI_14'].iloc[i] < 80: # not overbought
                    signal[i] = 1
                    sum += 1
                    stoploss = data['Close'].iloc[i] - (data['ATR_14'].iloc[i] * 0.50)
                    stoploss2 = data['Close'].iloc[i] * 0.995
                    stoploss = max(stoploss, stoploss2)

        elif sum == 1:
            if data['Low'].iloc[i] < stoploss:
                stoploss_hits[i] = stoploss
                signal[i] = -1
                sum -= 1
                stoploss = 0
            elif data['HA_ADX_Plus_DI'].iloc[i] < data['HA_ADX_Minus_DI'].iloc[i]:
                #if data['HA_ADX_14'] > 25:
                signal[i] = -1
                sum -= 1
            else: ## adding trailing stoploss
                stoploss1 = data['Close'].iloc[i] - (data['ATR_14'].iloc[i] * 0.50)
                stoploss2 = data['Close'].iloc[i] * 0.995
                stoploss = max(stoploss, stoploss1, stoploss2)



    data['signals'] = signal
    data['stoploss'] = stoploss_hits

    data = data[['Open', 'High', 'Low', 'Close', 'Volume', 'signals', 'stoploss']].copy()
    data.columns = data.columns.str.lower()
    data.reset_index(inplace=True)
    return data
    # try only trading when fast > slow

def main():
    data = pd.read_csv('data/AAPL.csv', parse_dates=['Price'], index_col='Price')
    
    process_data(data)
    
    strategy_signals = strat(data)
    print(strategy_signals)
    strategy_signals.to_csv("results.csv", index=False)
    
    for i in range(len(strategy_signals)):
        if sum(strategy_signals[:i]['signals']) > 1:
            print(i, 'wtf')
        
    run_backtest("results.csv")

if __name__ == "__main__":
    main()