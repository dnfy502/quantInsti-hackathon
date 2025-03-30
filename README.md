Please access the following file: https://colab.research.google.com/drive/11qWJ5ud10SXIyNkiqfK1LzjeZWHqZ4B_?usp=sharing
All details as well as graphs are present in that. Additionally, all code used is present in this repository as well.

# AAPL Momentum Strategy Documentation

## Strategy Overview

This strategy implements a momentum-based approach to trade AAPL stock using the Average Directional Index (ADX) as the primary indicator to identify strong trends. The strategy takes advantage of directional movement by entering long positions when the market shows strong upward momentum and utilizing a trailing stop-loss mechanism to protect profits.

## Technical Indicators Used

1. **Average Directional Index (ADX)**: 
   - Used to determine the strength of a trend, regardless of its direction
   - The strategy uses ADX with a 14-period setting
   - Trades are only taken when ADX is above 30, indicating a strong trend

2. **Directional Movement Indicators (DI+ and DI-)**:
   - The Positive Directional Indicator (DI+) measures upward price movement
   - The Negative Directional Indicator (DI-) measures downward price movement
   - The strategy enters long positions when DI+ is greater than DI-

3. **Average True Range (ATR)**:
   - Used to measure market volatility
   - Serves as the basis for setting dynamic stop-loss levels
   - The strategy uses ATR with a 14-period setting

4. **Heikin-Ashi Candles**:
   - Modified candlestick charts that filter out market noise
   - Used in the ADX calculation to provide smoother signals

## Trading Rules

### Entry Conditions (Long Only)
1. DI+ is greater than DI- (upward momentum)
2. ADX is above 30 (strong trend)

### Exit Conditions
1. DI+ falls below DI- (momentum reversal)
2. Stop-loss is hit (price protection)

### Stop-Loss Management
- Initial stop-loss set at the maximum of:
  - Current price minus (ATR × 0.50)
  - 0.5% below entry price (tighter than the TSLA strategy)
- Trailing stop-loss: As the position moves in favor, the stop-loss is raised to protect profits, taking the maximum of:
  - Previous stop-loss level
  - Current price minus (ATR × 0.50)
  - 0.5% below current price

## Implementation Notes

The strategy employs Heikin-Ashi candles for calculating ADX, which helps filter noise and produces more reliable signals. The stop-loss incorporates both ATR-based volatility adjustment and a percentage-based approach, taking the more conservative of the two values.

Key differences from the TSLA strategy:
1. Tighter stop-loss at 0.5% below entry price (vs. 2% for TSLA)
2. More aggressive trailing stop-loss that considers the current price level

## Performance Metrics

| Metric | Value |
|--------|-------|
| Initial Balance | $1,000.00 |
| Final Balance | $11,429.48 |
| ROI | 1,042.95% |
| Benchmark ROI | 935.85% |
| Number of Trades | 425 |
| Win Rate | 39.62% |
| Average Win | $102.36 |
| Average Loss | -$26.43 |
| Total Fees | $0.00 |
| Maximum Win | $575.61 |
| Maximum Loss | -$58.56 |
| Number of Winning Trades | 168 |
| Number of Losing Trades | 256 |
| Number of Long Trades | 425 |
| Number of Short Trades | 0 |
| Maximum Drawdown | 6.47% |
| Sharpe Ratio | 4.87 |

## Performance Analysis

### Strengths
1. **Strong Risk-Adjusted Returns**: With a Sharpe Ratio of 4.87, the strategy demonstrates exceptional risk-adjusted performance, significantly better than the TSLA strategy (3.61), indicating more consistent returns relative to volatility.

2. **Favorable Reward-to-Risk Ratio**: Average winning trades ($102.36) are approximately 3.9 times larger than average losing trades (-$26.43), showing effective risk management.

3. **Limited Drawdown**: The maximum drawdown of only 6.47% is remarkably low, especially compared to the TSLA strategy's 20.86%, indicating superior capital preservation.

4. **Market Outperformance**: The strategy achieved a 1,042.95% ROI, outperforming the benchmark ROI of 935.85%, demonstrating alpha generation.

### Weaknesses
1. **Low Win Rate**: At 39.62%, the strategy wins fewer than half of its trades, suggesting frequent whipsaws or false signals.

2. **Long-Only Approach**: The strategy only takes long positions, potentially missing opportunities during bearish market conditions.

3. **Smaller Absolute Returns**: While still impressive, the 1,042.95% ROI is significantly lower than the TSLA strategy's 6,359.88%, reflecting the different volatility profiles of the underlying assets.

## Future Improvements

1. **Incorporate Short Positions**: Expand the strategy to include short positions during strong downtrends.

2. **Add Volume Filters**: Consider volume confirmation for entry signals to ensure sufficient market interest.

3. **Position Sizing**: Implement a more sophisticated position sizing model based on market volatility.

4. **Market Regime Detection**: Add filters to determine overall market conditions and adjust strategy parameters accordingly.

5. **Parameter Optimization**: Conduct formal optimization of ADX threshold and stop-loss parameters to improve performance.

# TSLA Momentum Strategy Documentation

## Strategy Overview

This strategy implements a momentum-based approach to trade TSLA stock using the Average Directional Index (ADX) as the primary indicator to identify strong trends. The strategy takes advantage of directional movement by entering long positions when the market shows strong upward momentum and utilizing a trailing stop-loss mechanism to protect profits.

## Technical Indicators Used

1. **Average Directional Index (ADX)**: 
   - Used to determine the strength of a trend, regardless of its direction
   - The strategy uses ADX with a 14-period setting
   - Trades are only taken when ADX is above 30, indicating a strong trend

2. **Directional Movement Indicators (DI+ and DI-)**:
   - The Positive Directional Indicator (DI+) measures upward price movement
   - The Negative Directional Indicator (DI-) measures downward price movement
   - The strategy enters long positions when DI+ crosses above DI-

3. **Average True Range (ATR)**:
   - Used to measure market volatility
   - Serves as the basis for setting dynamic stop-loss levels
   - The strategy uses ATR with a 14-period setting

4. **Heikin-Ashi Candles**:
   - Modified candlestick charts that filter out market noise
   - Used in the ADX calculation to provide smoother signals

## Trading Rules

### Entry Conditions (Long Only)
1. DI+ is greater than DI- (upward momentum)
2. ADX is above 30 (strong trend)

### Exit Conditions
1. DI+ crosses below DI- (momentum reversal)
2. Stop-loss is hit (price protection)

### Stop-Loss Management
- Initial stop-loss set at the maximum of:
  - Current price minus (ATR × 0.5)
  - 2% below entry price
- Trailing stop-loss: As the position moves in favor, the stop-loss is raised to protect profits

## Performance Metrics

| Metric | Value |
|--------|-------|
| Initial Balance | $1,000.00 |
| Final Balance | $64,598.80 |
| ROI | 6,359.88% |
| Benchmark ROI | 2,754.93% |
| Number of Trades | 330 |
| Win Rate | 40.61% |
| Average Win | $783.67 |
| Average Loss | -$211.29 |
| Total Fees | $0.00 |
| Maximum Win | $13,653.25 |
| Maximum Loss | -$1,318.34 |
| Number of Winning Trades | 134 |
| Number of Losing Trades | 196 |
| Number of Long Trades | 330 |
| Number of Short Trades | 0 |
| Maximum Drawdown | 20.86% |
| Sharpe Ratio | 3.61 |

## Performance Analysis

### Strengths
1. **Exceptional ROI**: The strategy delivered a 6,359.88% return, significantly outperforming the benchmark (2,754.93%), demonstrating strong alpha generation.

2. **Favorable Reward-to-Risk Ratio**: With average winning trades ($783.67) being 3.7 times larger than average losing trades (-$211.29), the strategy effectively captures large moves while limiting losses.

3. **Strong Risk-Adjusted Returns**: A Sharpe Ratio of 3.61 indicates excellent risk-adjusted performance, showing that the returns compensate well for the volatility experienced.

4. **Trailing Stop Mechanism**: The implementation of a trailing stop-loss helps protect profits while allowing winning trades to run, contributing to the high average win amount.

### Weaknesses
1. **Low Win Rate**: At 40.61%, the strategy wins less than half of its trades, suggesting frequent whipsaws or false signals.

2. **Maximum Drawdown**: A 20.86% drawdown indicates periods of significant account value decline, which could cause psychological challenges for traders implementing this strategy.

3. **Long-Only Approach**: The strategy only takes long positions, potentially missing opportunities during bearish market conditions.

## Implementation Notes

The strategy employs Heikin-Ashi candles for calculating ADX, which helps filter noise and produces more reliable signals. The stop-loss incorporates both ATR-based volatility adjustment and a percentage-based approach, taking the more conservative of the two values.
