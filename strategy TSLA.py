import pandas as pd
import numpy as np
from backtesting3 import run_backtest


def hma(series, period):
    # Calculate weights
    weights = np.arange(1, period + 1)
    
    # Calculate WMA with half period
    half = period // 2
    wmaf = pd.Series(series).rolling(window=half, center=False).apply(
        lambda x: np.sum(weights[:half] * x) / weights[:half].sum(), raw=True)
    
    # Calculate WMA with full period
    wma = pd.Series(series).rolling(window=period, center=False).apply(
        lambda x: np.sum(weights * x) / weights.sum(), raw=True)
    
    # Calculate raw HMA
    raw_hma = 2 * wmaf - wma
    
    # Calculate final HMA using sqrt(period)
    sqrt_period = int(np.sqrt(period))
    sqrt_weights = np.arange(1, sqrt_period + 1)
    hma = pd.Series(raw_hma).rolling(window=sqrt_period, center=False).apply(
        lambda x: np.sum(sqrt_weights * x) / sqrt_weights.sum(), raw=True)
    
    return hma
    
def kalman_filter(data, column='Close', use_ha=False, new_column_name=None, 
                  transition_covariance=0.01, observation_covariance=1.0, initial_value_guess=None):
    """
    Apply Kalman filter to smooth price series data
    
    Parameters:
    data (pandas.DataFrame): DataFrame with price data
    column (str): Column name to use for calculation
    use_ha (bool): Whether to use Heikin-Ashi candles
    new_column_name (str): Name for the new column, defaults to f'Kalman_{column}'
    transition_covariance (float): Process noise, higher values = more responsive filter
    observation_covariance (float): Measurement noise, higher values = smoother filter
    initial_value_guess (float): Initial state estimate, defaults to first value in series
    
    Returns:
    None: Adds result column to the input DataFrame
    """
    # Select appropriate column based on whether to use Heikin-Ashi
    price_col = f'ha_{column.lower()}' if use_ha else column
    
    if new_column_name is None:
        prefix = 'HA_' if use_ha else ''
        new_column_name = f'{prefix}Kalman_{column}'
    
    # Get the price series
    price_series = data[price_col].copy()
    
    # Remove NaN values for calculation
    valid_data = price_series.dropna()
    if len(valid_data) == 0:
        data[new_column_name] = np.nan
        return
    
    # Get initial value guess
    if initial_value_guess is None:
        initial_value_guess = valid_data.iloc[0]
    
    # Initialize Kalman filter parameters
    filtered_state_means = np.zeros(len(price_series))
    filtered_state_covariances = np.zeros(len(price_series))
    
    # Initialize state
    state_mean = initial_value_guess
    state_covariance = 1.0
    
    # Apply the Kalman filter
    for i in range(len(price_series)):
        if pd.isna(price_series.iloc[i]):
            filtered_state_means[i] = state_mean
            filtered_state_covariances[i] = state_covariance
            continue
        
        # Prediction step
        predicted_state_mean = state_mean
        predicted_state_covariance = state_covariance + transition_covariance
        
        # Update step
        kalman_gain = predicted_state_covariance / (predicted_state_covariance + observation_covariance)
        state_mean = predicted_state_mean + kalman_gain * (price_series.iloc[i] - predicted_state_mean)
        state_covariance = (1 - kalman_gain) * predicted_state_covariance
        
        # Store results
        filtered_state_means[i] = state_mean
        filtered_state_covariances[i] = state_covariance
    
    # Add to DataFrame
    data[new_column_name] = filtered_state_means


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

def obv(data, use_ha=False, new_column_name=None):
    """
    Calculate On-Balance Volume (OBV)
    
    Parameters:
    data (pandas.DataFrame): DataFrame with OHLCV data
    use_ha (bool): Whether to use Heikin-Ashi candles
    new_column_name (str): Name for the new column
    """
    if new_column_name is None:
        prefix = 'HA_' if use_ha else ''
        new_column_name = f'{prefix}OBV'
    
    # Select appropriate column based on whether to use Heikin-Ashi
    close_col = 'ha_close' if use_ha else 'Close'
    close = data[close_col]
    volume = data['Volume']
    
    # Initialize the OBV with first day's volume
    obv_values = [0]
    
    # Calculate OBV
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:  # Close price increased
            obv_values.append(obv_values[-1] + volume.iloc[i])
        elif close.iloc[i] < close.iloc[i-1]:  # Close price decreased
            obv_values.append(obv_values[-1] - volume.iloc[i])
        else:  # Close price unchanged
            obv_values.append(obv_values[-1])
    
    data[new_column_name] = obv_values

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

def supertrend(data, period=14, multiplier=3, use_ha=False, new_column_prefix=None):
    """
    Calculate Supertrend indicator
    
    Parameters:
    data (pandas.DataFrame): DataFrame with OHLC data
    period (int): Period for ATR calculation
    multiplier (int): Multiplier for ATR
    use_ha (bool): Whether to use Heikin-Ashi candles
    new_column_prefix (str): Prefix for new columns
    """
    if new_column_prefix is None:
        prefix = 'HA_' if use_ha else ''
        new_column_prefix = f'{prefix}Supertrend'
    
    # Select appropriate columns based on whether to use Heikin-Ashi
    high_col = 'ha_high' if use_ha else 'High'
    low_col = 'ha_low' if use_ha else 'Low'
    close_col = 'ha_close' if use_ha else 'Close'
    
    high = data[high_col]
    low = data[low_col]
    close = data[close_col]
    
    # Calculate ATR
    atr(data, period, use_ha=use_ha)
    atr_column = f'{"HA_" if use_ha else ""}ATR_{period}'
    
    # Calculate basic upper and lower bands
    data['Basic_Upper_Band'] = (high + low) / 2 + (multiplier * data[atr_column])
    data['Basic_Lower_Band'] = (high + low) / 2 - (multiplier * data[atr_column])
    
    # Initialize Supertrend columns
    st_column = f'{new_column_prefix}_{period}_{multiplier}'
    trend_column = f'{st_column}_Trend'
    
    # Initialize columns with NaN for better handling
    data[st_column] = np.nan
    data[trend_column] = np.nan
    data['Final_Upper_Band'] = np.nan
    data['Final_Lower_Band'] = np.nan
    
    # Find the first valid index position (not the actual index value)
    first_valid_pos = data[atr_column].first_valid_index()
    if first_valid_pos is None:
        # No valid data, return early
        data.drop(['Basic_Upper_Band', 'Basic_Lower_Band', 'Final_Upper_Band', 'Final_Lower_Band'], axis=1, inplace=True)
        return
    
    # Get position in integer index
    first_valid_pos = data.index.get_loc(first_valid_pos)
    
    # Set initial values
    data.iloc[first_valid_pos, data.columns.get_loc('Final_Upper_Band')] = data.iloc[first_valid_pos, data.columns.get_loc('Basic_Upper_Band')]
    data.iloc[first_valid_pos, data.columns.get_loc('Final_Lower_Band')] = data.iloc[first_valid_pos, data.columns.get_loc('Basic_Lower_Band')]
    
    # Initial trend and Supertrend value
    if close.iloc[first_valid_pos] <= data.iloc[first_valid_pos, data.columns.get_loc('Final_Upper_Band')]:
        data.iloc[first_valid_pos, data.columns.get_loc(trend_column)] = -1  # Downtrend
        data.iloc[first_valid_pos, data.columns.get_loc(st_column)] = data.iloc[first_valid_pos, data.columns.get_loc('Final_Upper_Band')]
    else:
        data.iloc[first_valid_pos, data.columns.get_loc(trend_column)] = 1  # Uptrend
        data.iloc[first_valid_pos, data.columns.get_loc(st_column)] = data.iloc[first_valid_pos, data.columns.get_loc('Final_Lower_Band')]
    
    # Calculate Supertrend iteratively for the rest of the data
    for i in range(first_valid_pos + 1, len(data)):
        # Calculate Final Upper Band
        if (data.iloc[i, data.columns.get_loc('Basic_Upper_Band')] < data.iloc[i-1, data.columns.get_loc('Final_Upper_Band')] or 
            close.iloc[i-1] > data.iloc[i-1, data.columns.get_loc('Final_Upper_Band')]):
            data.iloc[i, data.columns.get_loc('Final_Upper_Band')] = data.iloc[i, data.columns.get_loc('Basic_Upper_Band')]
        else:
            data.iloc[i, data.columns.get_loc('Final_Upper_Band')] = data.iloc[i-1, data.columns.get_loc('Final_Upper_Band')]
        
        # Calculate Final Lower Band
        if (data.iloc[i, data.columns.get_loc('Basic_Lower_Band')] > data.iloc[i-1, data.columns.get_loc('Final_Lower_Band')] or 
            close.iloc[i-1] < data.iloc[i-1, data.columns.get_loc('Final_Lower_Band')]):
            data.iloc[i, data.columns.get_loc('Final_Lower_Band')] = data.iloc[i, data.columns.get_loc('Basic_Lower_Band')]
        else:
            data.iloc[i, data.columns.get_loc('Final_Lower_Band')] = data.iloc[i-1, data.columns.get_loc('Final_Lower_Band')]
        
        # Determine trend and Supertrend value
        if (data.iloc[i-1, data.columns.get_loc(trend_column)] == 1 and 
            close.iloc[i] < data.iloc[i, data.columns.get_loc('Final_Lower_Band')]):
            # Switching to downtrend
            data.iloc[i, data.columns.get_loc(trend_column)] = -1
            data.iloc[i, data.columns.get_loc(st_column)] = data.iloc[i, data.columns.get_loc('Final_Upper_Band')]
        elif (data.iloc[i-1, data.columns.get_loc(trend_column)] == -1 and 
              close.iloc[i] > data.iloc[i, data.columns.get_loc('Final_Upper_Band')]):
            # Switching to uptrend
            data.iloc[i, data.columns.get_loc(trend_column)] = 1
            data.iloc[i, data.columns.get_loc(st_column)] = data.iloc[i, data.columns.get_loc('Final_Lower_Band')]
        else:
            # Continue the trend
            data.iloc[i, data.columns.get_loc(trend_column)] = data.iloc[i-1, data.columns.get_loc(trend_column)]
            if data.iloc[i, data.columns.get_loc(trend_column)] == 1:
                data.iloc[i, data.columns.get_loc(st_column)] = data.iloc[i, data.columns.get_loc('Final_Lower_Band')]
            else:
                data.iloc[i, data.columns.get_loc(st_column)] = data.iloc[i, data.columns.get_loc('Final_Upper_Band')]
    
    # Clean up temporary columns
    data.drop(['Basic_Upper_Band', 'Basic_Lower_Band', 'Final_Upper_Band', 'Final_Lower_Band'], axis=1, inplace=True)

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

def rsi(data, period=14, column='Close', use_ha=False, new_column_name=None):
    """
    Calculate Relative Strength Index (RSI)
    
    Parameters:
    data (pandas.DataFrame): DataFrame with price data
    period (int): Period for RSI calculation
    column (str): Column name to use for calculation
    use_ha (bool): Whether to use Heikin-Ashi candles
    new_column_name (str): Name for the new column, defaults to f'RSI_{period}'
    """
    if new_column_name is None:
        prefix = 'HA_' if use_ha else ''
        new_column_name = f'{prefix}RSI_{period}'
    
    # Select appropriate column based on whether to use Heikin-Ashi
    price_col = f'ha_{column.lower()}' if use_ha else column
    
    # Calculate price changes
    delta = data[price_col].diff()
    
    # Separate gains and losses
    gain = delta.copy()
    loss = delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    loss = abs(loss)
    
    # Calculate average gain and loss over the period
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Calculate RS (Relative Strength)
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    data[new_column_name] = 100 - (100 / (1 + rs))

def process_data(data):
    # Drop rows with zero volume
    data.drop(data[data['Volume'] == 0].index, inplace=True)
    heikin_ashi(data)
    adx(data, period=14, use_ha=True)
    supertrend(data, period=14, multiplier=3, use_ha=False)
    rsi(data, period=14, column='ha_close', use_ha=False)
    #atr(data)

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
                    stoploss = data['Close'].iloc[i] - (data['ATR_14'].iloc[i] * 0.5)
                    stoploss2 = data['Close'].iloc[i] * 0.98
                    stoploss = max(stoploss, stoploss2)
            # elif data['HA_ADX_Plus_DI'].iloc[i] < data['HA_ADX_Minus_DI'].iloc[i]:
            #     if data['HA_ADX_14'].iloc[i] > 30:
            #         signal[i] = -1
            #         sum += 1
            #         stoploss = data['Close'].iloc[i] + (data['ATR_14'].iloc[i] * 0.5)

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
                stoploss = data['Close'].iloc[i] - (data['ATR_14'].iloc[i] * 0.5)
                stoploss2 = data['Close'].iloc[i] * 0.98
                stoploss = max(stoploss, stoploss2)


    data['signals'] = signal
    data['stoploss'] = stoploss_hits

    data = data[['Open', 'High', 'Low', 'Close', 'Volume', 'signals', 'stoploss']].copy()
    data.columns = data.columns.str.lower()
    data.reset_index(inplace=True)
    return data
    # try only trading when fast > slow

def main():
    data = pd.read_csv('data/TSLA.csv', parse_dates=['Price'], index_col='Price')
    
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