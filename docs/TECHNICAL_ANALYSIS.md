# 🔧 Technical Analysis Guide

Understanding the technical indicators and analysis methods used by IDX Stock Analyzer.

---

## Table of Contents

- [Support & Resistance](#support--resistance)
- [Moving Averages](#moving-averages)
- [RSI (Relative Strength Index)](#rsi-relative-strength-index)
- [Bollinger Bands](#bollinger-bands)
- [Volume Profile](#volume-profile)
- [Trend Analysis](#trend-analysis)
- [Trading Recommendations](#trading-recommendations)

---

## Support & Resistance

### Definition

- **Support**: Price level where buying pressure overcomes selling pressure. The floor that prevents the price from falling further.
- **Resistance**: Price level where selling pressure overcomes buying pressure. The ceiling that prevents the price from rising further.

### Types of Levels

| Type | Strength | Description |
|------|----------|-------------|
| **52-week high/low** | Strong | Major psychological levels over a year |
| **Recent swing high/low** | Moderate | Local peaks/troughs from recent price action |
| **Psychological levels** | Weak | Round numbers (e.g., 7000, 7500) |

### How They're Calculated

1. **52-week low** is automatically the strongest support
2. **52-week high** is automatically the strongest resistance
3. **Recent swing points** from last 30 days are identified
4. **Psychological levels** are generated based on current price

---

## Moving Averages

### Simple Moving Average (SMA)

The average price over a specific period. Each day has equal weight.

**Formula:**
```
SMA = (P1 + P2 + ... + Pn) / n
```

### Timeframes

| SMA | Period | Significance |
|-----|--------|--------------|
| **SMA 20** | 20 days | Short-term trend |
| **SMA 50** | 50 days | Medium-term trend |
| **SMA 200** | 200 days | Long-term trend / major support or resistance |

### Golden Cross & Death Cross

**Golden Cross** 📈
- SMA 50 crosses **above** SMA 200
- Long-term bullish signal
- Indicates potential sustained uptrend

**Death Cross** 📉
- SMA 50 crosses **below** SMA 200
- Long-term bearish signal
- Indicates potential sustained downtrend

### Strategy Signals

| Price Position | Trend Signal |
|----------------|--------------|
| Price > SMA 20 > SMA 50 > SMA 200 | Strong uptrend (Bullish) |
| Price < SMA 20 < SMA 50 < SMA 200 | Strong downtrend (Bearish) |
| Price above all MAs | Bullish momentum |
| Price below all MAs | Bearish momentum |
| Price between MAs | Mixed/Consolidating |

---

## RSI (Relative Strength Index)

### Definition

Momentum oscillator that measures the speed and change of price movements. Ranges from 0 to 100.

**Formula:**
```
RSI = 100 - (100 / (1 + RS))
RS = Average Gain / Average Loss
```

### Interpretation

| RSI Level | Condition | Signal |
|-----------|-----------|--------|
| **> 70** | Overbought | Potential sell signal |
| **> 75** | Extremely overbought | High probability pullback |
| **< 30** | Oversold | Potential buy signal |
| **< 25** | Extremely oversold | High probability bounce |
| **30-70** | Neutral | No clear signal |

### RSI Divergence

- **Bullish Divergence**: Price makes lower low, RSI makes higher low
- **Bearish Divergence**: Price makes higher high, RSI makes lower high

---

## Bollinger Bands

### Definition

Volatility bands placed above and below a moving average. Consists of:
- **Middle Band**: SMA 20
- **Upper Band**: SMA 20 + (2 × Standard Deviation)
- **Lower Band**: SMA 20 - (2 × Standard Deviation)

### Interpretation

| Position | Signal |
|----------|--------|
| **Price > Upper Band** | Overbought / Extended |
| **Price near Upper Band** | Strong uptrend |
| **Price near Middle Band** | Neutral / Balanced |
| **Price near Lower Band** | Weak downtrend |
| **Price < Lower Band** | Oversold / Extended |

### Band Width

- **Wide bands**: High volatility
- **Narrow bands**: Low volatility ("squeeze") - often precedes big moves

---

## Volume Profile

### Definition

Shows trading activity (volume) at specific price levels over a given time period.

### Key Concepts

| Concept | Definition |
|---------|------------|
| **POC (Point of Control)** | Price level with highest trading volume |
| **Value Area** | Price range containing ~70% of total volume |
| **Value Area High (VAH)** | Upper boundary of value area |
| **Value Area Low (VAL)** | Lower boundary of value area |

### Trading Implications

- **Price above POC**: Bullish bias
- **Price below POC**: Bearish bias
- **Price in Value Area**: Fair value, equilibrium
- **Price outside Value Area**: Extended, potential mean reversion

---

## Trend Analysis

### Trend Determination

The tool evaluates multiple factors:

1. **Price vs Moving Averages**
   - Above = Bullish points
   - Below = Bearish points

2. **Moving Average Crosses**
   - Golden Cross = Bullish
   - Death Cross = Bearish

3. **RSI Conditions**
   - Overbought (>70) = Caution on longs
   - Oversold (<30) = Caution on shorts

### Trend Classification

| Classification | Criteria |
|----------------|----------|
| **Strong Uptrend** | Price > SMA 20 > SMA 50 > SMA 200 |
| **Uptrend** | Price > SMA 20 and SMA 50 |
| **Mixed/Neutral** | Price between moving averages |
| **Downtrend** | Price < SMA 20 and SMA 50 |
| **Strong Downtrend** | Price < SMA 20 < SMA 50 < SMA 200 |

---

## Trading Recommendations

### Risk/Reward Ratio

Calculated as:
```
Risk = Current Price - Nearest Support
Reward = Nearest Resistance - Current Price
R/R Ratio = Reward / Risk
```

**Interpretation:**
- **1:1** - Even risk/reward (neutral)
- **1:2** - Good setup (twice the reward vs risk)
- **1:3** - Excellent setup (three times the reward vs risk)

### Recommendation Types

| Recommendation | Condition | Action |
|----------------|-----------|--------|
| **BUY** | Golden Cross + RSI < 70 | Enter long position |
| **BUY THE DIP** | Uptrend + RSI < 30 | Add to position on weakness |
| **SELL/AVOID** | Death Cross + RSI > 30 | Exit or avoid longs |
| **SELL THE RALLY** | Downtrend + RSI > 70 | Short or exit longs |
| **TAKE PROFITS** | RSI > 75 | Reduce position size |
| **WATCH CLOSELY** | RSI < 25 | Prepare for reversal |
| **HOLD/BUY** | Bullish trend, not overbought | Maintain/add on support |
| **AVOID/HOLD** | Bearish trend | Stay in cash |
| **WAIT** | No clear bias | No action |

### Position Sizing Considerations

1. **Never risk more than 1-2%** of portfolio on single trade
2. **Use stop loss** at support level (or 5-8% below entry)
3. **Take partial profits** at first resistance
4. **Trail stop** as price moves in your favor

---

## Combining Indicators

### High Probability Setups

**Long Setup:**
- Golden Cross active (SMA 50 > SMA 200)
- Price near support or lower Bollinger Band
- RSI 30-50 (not overbought)
- Volume profile POC below current price

**Short Setup:**
- Death Cross active (SMA 50 < SMA 200)
- Price near resistance or upper Bollinger Band
- RSI 50-70 (not oversold)
- Volume profile POC above current price

### Confluence is Key

The strongest signals occur when multiple indicators align:
- Price at support + RSI oversold + Near lower BB
- Golden Cross + Price breakout + Volume spike
- Death Cross + Resistance test + RSI overbought

---

*For more information, see [README.md](../README.md) and [USAGE.md](USAGE.md)*
