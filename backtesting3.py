import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def run_backtest(file_path, initial_capital=1000, commission_rate=0.00):
    """
    Run a backtest on trading strategy results.
    
    Args:
        file_path (str): Path to the CSV file containing trading data
        initial_capital (float): Initial portfolio value
        commission_rate (float): Commission rate as a percentage
        
    Returns:
        None: Displays metrics and a chart
    """
    # Load trading data
    trade_data = pd.read_csv(file_path)
    trading_signals = trade_data['signals']
    price_series = trade_data['close']
    stop_loss_prices = trade_data['stoploss']
    timestamp_series = trade_data['Price']
    
    # Initialize tracking variables
    realized_equity = []  # Realized equity curve
    portfolio_valuation = []  # Unrealized equity curve (portfolio value over time)
    benchmark_values = []  # Buy and hold strategy performance
    trade_profits = []  # PnL for each completed trade
    
    # Setup initial state
    current_capital = initial_capital
    realized_equity.append(float(current_capital))
    current_position = 0  # 0: no position, 1: long, -1: short
    long_trade_count = 0
    short_trade_count = 0

    # Calculate initial buy-and-hold amount
    benchmark_shares = initial_capital / price_series[0]
    
    # Main backtest loop
    for i in range(len(trade_data)):
        # Update benchmark buy-and-hold value
        benchmark_values.append(benchmark_shares * price_series[i])

        if trading_signals[i] in (+1, -1, +2, -2):
            # Open long position
            if trading_signals[i] == +1 and current_position == 0:
                current_position = 1
                long_trade_count += 1
                entry_price = price_series[i]
                transaction_fee = (commission_rate * current_capital) / 100
                shares_purchased = current_capital / entry_price
                portfolio_valuation.append(round(float(current_capital), 2))

            # Open short position
            elif trading_signals[i] == -1 and current_position == 0:
                current_position = -1
                short_trade_count += 1
                entry_price = price_series[i]
                transaction_fee = (commission_rate * current_capital) / 100           
                shares_sold = current_capital / entry_price
                portfolio_valuation.append(round(float(current_capital), 2))
            
            # Close long position
            elif trading_signals[i] == -1 and current_position == 1:
                if stop_loss_prices[i] > 0:
                    exit_price = stop_loss_prices[i]
                else:
                    exit_price = price_series[i]
                current_position = 0
                trade_profit = (exit_price - entry_price) * shares_purchased - transaction_fee
                trade_profits.append(trade_profit)
                current_capital += trade_profit
                realized_equity.append(float(current_capital))
                portfolio_valuation.append(round(float(current_capital), 2))
            
            # Close short position
            elif trading_signals[i] == +1 and current_position == -1:
                current_position = 0
                exit_price = price_series[i]
                trade_profit = (entry_price - exit_price) * shares_sold - transaction_fee
                trade_profits.append(trade_profit)
                current_capital += trade_profit
                realized_equity.append(float(current_capital))
                portfolio_valuation.append(round(float(current_capital), 2))
            
            # Switch from long to short
            elif trading_signals[i] == -2:
                exit_price = price_series[i]
                trade_profit = (exit_price - entry_price) * shares_purchased - transaction_fee
                trade_profits.append(trade_profit)
                current_capital += trade_profit
                realized_equity.append(float(current_capital))

                current_position = -1
                short_trade_count += 1
                entry_price = price_series[i]
                transaction_fee = (commission_rate * current_capital) / 100
                shares_sold = current_capital / entry_price
                portfolio_valuation.append(round(float(current_capital), 2))
            
            # Switch from short to long
            elif trading_signals[i] == +2:
                exit_price = price_series[i]
                trade_profit = (entry_price - exit_price) * shares_sold - transaction_fee
                trade_profits.append(trade_profit)
                current_capital += trade_profit
                realized_equity.append(float(current_capital))

                current_position = +1
                long_trade_count += 1
                transaction_fee = (commission_rate * current_capital) / 100
                entry_price = price_series[i]
                shares_purchased = current_capital / entry_price
                portfolio_valuation.append(round(float(current_capital), 2))

        elif current_position == +1 and trading_signals[i] == 0:
            unrealized_profit = price_series[i] * shares_purchased - current_capital
            portfolio_valuation.append(round(float(current_capital + unrealized_profit), 2))

        elif current_position == -1 and trading_signals[i] == 0:
            unrealized_profit = -(price_series[i] * shares_sold - current_capital)
            portfolio_valuation.append(round(float(current_capital + unrealized_profit), 2))
        else:
            portfolio_valuation.append(round(float(current_capital), 2))

    # Calculate performance metrics
    starting_balance = initial_capital
    ending_balance = current_capital
    return_percentage = ((ending_balance - starting_balance) / starting_balance) * 100
    benchmark_return = ((benchmark_values[-1] - benchmark_values[0]) / benchmark_values[0]) * 100
    winning_trades = len([profit for profit in trade_profits if profit > 0])
    losing_trades = len(trade_profits) - winning_trades
    win_rate = (winning_trades / len(trade_profits)) * 100 if trade_profits else 0
    total_trades = long_trade_count + short_trade_count
    avg_winning_trade = np.mean([profit for profit in trade_profits if profit > 0]) if any(profit > 0 for profit in trade_profits) else 0
    avg_losing_trade = np.mean([profit for profit in trade_profits if profit < 0]) if any(profit < 0 for profit in trade_profits) else 0
    total_fees = sum((commission_rate * value / 100) for value in realized_equity[:-1])
    max_winning_trade = max(trade_profits) if trade_profits else 0
    max_losing_trade = min(trade_profits) if trade_profits else 0

    # Calculate maximum drawdown percentage only (removed TTR calculations)
    peak_value = portfolio_valuation[0]
    max_drawdown_pct = 0
    
    for i in range(len(portfolio_valuation)):
        if portfolio_valuation[i] > peak_value:
            peak_value = portfolio_valuation[i]
        else:
            current_drawdown = (peak_value - portfolio_valuation[i]) / peak_value * 100
            if current_drawdown > max_drawdown_pct:
                max_drawdown_pct = current_drawdown

    # Calculate Sharpe ratio
    daily_returns = np.array(trade_profits) / np.array(realized_equity[:-1])
    risk_free_rate = 0.02 / 252  # Daily risk-free rate
    excess_returns = daily_returns - risk_free_rate
    avg_excess_return = np.mean(excess_returns)
    std_excess_return = np.std(excess_returns)
    sharpe_ratio = (avg_excess_return / std_excess_return) * np.sqrt(252) if std_excess_return != 0 else 0

    # Display performance metrics
    print(f'Initial Balance: {starting_balance:.2f}$')
    print(f'Final Balance: {ending_balance:.2f}$')
    print(f'ROI: {return_percentage:.2f}%')
    print(f'Benchmark ROI: {benchmark_return:.2f}%')
    print(f'Number of Trades: {total_trades}')
    print(f'Win Rate: {win_rate:.2f}%')
    print(f'Average Win: {avg_winning_trade:.2f}$')
    print(f'Average Loss: {avg_losing_trade:.2f}$')
    print(f'Total Fees: {total_fees:.2f}$')
    print(f'Maximum Win: {max_winning_trade:.2f}$')
    print(f'Maximum Loss in a single trade: {max_losing_trade:.2f}$')
    print(f'Number of Winning Trades: {winning_trades}')
    print(f'Number of Losing Trades: {losing_trades}')
    print(f'Number of Long Trades: {long_trade_count}')
    print(f'Number of Short Trades: {short_trade_count}')
    print(f'Maximum Drawdown Percent: {max_drawdown_pct:.2f}%')
    print(f'Sharpe Ratio: {sharpe_ratio:.2f}')

    # Create performance visualization
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    fig.add_trace(go.Scatter(x=timestamp_series, y=portfolio_valuation, name="Portfolio Value"), secondary_y=False)
    fig.add_trace(go.Scatter(x=timestamp_series, y=benchmark_values, name="Buy and Hold Strategy"), secondary_y=False)
    
    fig.update_layout(
        title="Backtesting Results",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        template="plotly_dark"
    )
    
    fig.show()


if __name__ == '__main__':
    run_backtest('trading_results_new_12h (1).csv')