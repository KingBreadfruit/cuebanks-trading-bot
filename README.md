# CueBot — SMC-Based Algorithmic Trading Bot
### Trained on Smart Money Concepts (Cue Banks Framework)

## Strategy Foundation
This bot implements Cue Banks' core trading methodology:
- **Fair Value Gaps (FVGs)** — price imbalance zones that act as magnets
- **Order Blocks** — institutional buy/sell zones (last opposing candle before a big move)
- **Market Structure** — Break of Structure (BOS) + Change of Character (CHoCH)
- **H4 Bias** — top-down analysis, higher timeframe direction filter
- **Session Timing** — London (3am–5am EST) and NY Open (9:30am EST) windows

## Instruments
- **SPY** (S&P 500 ETF)
- **DIA** (Dow Jones ETF) / US30

## Architecture
```
trading_bot/
├── data/           # Market data fetching (yfinance + Alpaca)
├── features/       # SMC feature engineering (FVGs, OBs, Structure)
├── models/         # LSTM + Random Forest models
├── backtest/       # Historical backtesting engine
├── paper_trading/  # Live paper trading simulation
├── live/           # Alpaca live execution
├── dashboard/      # Real-time monitoring UI
└── utils/          # Shared helpers
```

## Quick Start
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Fetch historical data & train model
python train.py --symbol SPY --symbol DIA

# 3. Run backtester
python backtest/run_backtest.py

# 4. Start paper trading dashboard
python paper_trading/run_paper.py

# 5. Go live (after paper trading validation)
python live/run_live.py
```

## Environment Variables (.env)
```
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # Switch to live when ready
```
