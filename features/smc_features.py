"""
features/smc_features.py
Smart Money Concepts Feature Engineering — Cue Banks Framework

Implements:
  1. Fair Value Gaps (FVGs)
  2. Order Blocks (Bullish + Bearish)
  3. Market Structure (BOS + CHoCH)
  4. H4 Bias (top-down directional filter)
  5. Session Timing (London / NY Open)
"""

import pandas as pd
import numpy as np
import pytz


# ─────────────────────────────────────────────
# 1. FAIR VALUE GAPS (FVGs)
# ─────────────────────────────────────────────
def detect_fvg(df: pd.DataFrame) -> pd.DataFrame:
    """
    A Fair Value Gap occurs when candle[i-1].high < candle[i+1].low (bullish FVG)
    or candle[i-1].low > candle[i+1].high (bearish FVG).
    This is the gap/imbalance candle Cue targets for entries.

    CAUSALITY FIX (2026-06-17): The FVG at bar i is only KNOWABLE at the close
    of bar i+1 (since it requires nxt.low / nxt.high). The original code wrote
    the FVG status at the gap candle's index, which constituted a 1-bar look-
    ahead leak. The fix: shift fvg_* by +1 bar after detection so the value
    indexed at bar t represents "FVG just confirmed at close of bar t".
    """
    df = df.copy()
    df["fvg_bull"] = 0
    df["fvg_bear"] = 0
    df["fvg_bull_top"] = np.nan
    df["fvg_bull_bot"] = np.nan
    df["fvg_bear_top"] = np.nan
    df["fvg_bear_bot"] = np.nan

    for i in range(1, len(df) - 1):
        prev  = df.iloc[i - 1]
        curr  = df.iloc[i]
        nxt   = df.iloc[i + 1]

        # Bullish FVG: gap between prev high and next low
        if prev["high"] < nxt["low"]:
            df.at[df.index[i], "fvg_bull"]     = 1
            df.at[df.index[i], "fvg_bull_top"] = nxt["low"]
            df.at[df.index[i], "fvg_bull_bot"] = prev["high"]

        # Bearish FVG: gap between prev low and next high
        if prev["low"] > nxt["high"]:
            df.at[df.index[i], "fvg_bear"]     = 1
            df.at[df.index[i], "fvg_bear_top"] = prev["low"]
            df.at[df.index[i], "fvg_bear_bot"] = nxt["high"]

    # CAUSALITY FIX: shift the raw FVG detection forward by 1 bar so the value
    # indexed at bar t is the FVG that was confirmable AT the close of bar t
    # (i.e. derived from bars t-2 and t-1, both fully known at t).
    _fvg_cols = ["fvg_bull", "fvg_bear",
                 "fvg_bull_top", "fvg_bull_bot",
                 "fvg_bear_top", "fvg_bear_bot"]
    df[_fvg_cols] = df[_fvg_cols].shift(1)
    df[["fvg_bull", "fvg_bear"]] = df[["fvg_bull", "fvg_bear"]].fillna(0).astype(int)

    # Distance from current close to nearest FVG (as % of price)
    df["fvg_bull_dist"] = (df["close"] - df["fvg_bull_bot"]).fillna(0) / df["close"]
    df["fvg_bear_dist"] = (df["fvg_bear_top"] - df["close"]).fillna(0) / df["close"]

    return df


# ─────────────────────────────────────────────
# 2. ORDER BLOCKS
# ─────────────────────────────────────────────
def detect_order_blocks(df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    """
    Bullish OB: Last bearish (red) candle before a strong bullish move up.
    Bearish OB: Last bullish (green) candle before a strong bearish move down.
    Cue uses these as key entry zones.
    """
    df = df.copy()
    df["ob_bull"] = 0
    df["ob_bear"] = 0
    df["ob_bull_high"] = np.nan
    df["ob_bull_low"]  = np.nan
    df["ob_bear_high"] = np.nan
    df["ob_bear_low"]  = np.nan

    for i in range(lookback, len(df)):
        window = df.iloc[i - lookback:i]
        curr   = df.iloc[i]

        # Bullish OB: find last red candle in window before current bullish impulse
        if curr["close"] > curr["open"]:  # current is bullish
            red_candles = window[window["close"] < window["open"]]
            if not red_candles.empty:
                last_red = red_candles.iloc[-1]
                df.at[df.index[i], "ob_bull"]      = 1
                df.at[df.index[i], "ob_bull_high"] = last_red["high"]
                df.at[df.index[i], "ob_bull_low"]  = last_red["low"]

        # Bearish OB: find last green candle before current bearish impulse
        if curr["close"] < curr["open"]:  # current is bearish
            green_candles = window[window["close"] > window["open"]]
            if not green_candles.empty:
                last_green = green_candles.iloc[-1]
                df.at[df.index[i], "ob_bear"]      = 1
                df.at[df.index[i], "ob_bear_high"] = last_green["high"]
                df.at[df.index[i], "ob_bear_low"]  = last_green["low"]

    # Price relative to OB zone (inside OB = potential entry)
    df["in_bull_ob"] = (
        (df["close"] >= df["ob_bull_low"]) &
        (df["close"] <= df["ob_bull_high"])
    ).astype(int)

    df["in_bear_ob"] = (
        (df["close"] >= df["ob_bear_low"]) &
        (df["close"] <= df["ob_bear_high"])
    ).astype(int)

    return df


# ─────────────────────────────────────────────
# 3. MARKET STRUCTURE (BOS + CHoCH)
# ─────────────────────────────────────────────
def detect_market_structure(df: pd.DataFrame, swing_length: int = 5) -> pd.DataFrame:
    """
    Detects:
    - Swing highs / swing lows
    - Break of Structure (BOS): price breaks previous swing high/low in trend direction
    - Change of Character (CHoCH): price breaks structure AGAINST the trend (reversal signal)
    Cue uses CHoCH as a key signal that momentum is shifting.

    CAUSALITY FIX (2026-06-17): The original implementation used a centered
    swing window (df.iloc[i-n:i+n+1]) and wrote the swing flag at the swing
    bar's own index. That bar's swing status was only confirmable n bars
    LATER (we need to see bars i+1..i+n to know bar i is the max). This was a
    multi-bar look-ahead leak; on H4 (where this function feeds compute_h4_bias),
    n=5 → 20 hours of future info bleeding into the model. Live trading could
    not replicate this, which likely contributed to the live vs. backtest gap.

    The fix: identify swings centered, then SHIFT the flags AND swing prices
    forward by `swing_length` bars so the value at bar t represents what was
    confirmable at the close of bar t. BOS/CHoCH detection then runs on the
    shifted (causal) swings, comparing each newly-confirmed swing's price to
    the prior confirmed swing's price — same semantics as before but no peek.
    """
    df = df.copy()
    n = swing_length

    # ── Step 1: identify swings using centered window (uses future bars) ──
    raw_swing_high = np.zeros(len(df), dtype=np.int8)
    raw_swing_low  = np.zeros(len(df), dtype=np.int8)
    highs = df["high"].to_numpy()
    lows  = df["low"].to_numpy()
    for i in range(n, len(df) - n):
        if highs[i] == highs[i - n:i + n + 1].max():
            raw_swing_high[i] = 1
        if lows[i]  == lows[i - n:i + n + 1].min():
            raw_swing_low[i]  = 1

    # ── Step 2: shift swing flags AND swing prices forward by n bars ──
    # After shift: swing_high at bar t means "at bar t we just confirmed that
    # bar (t-n) was a swing high". The actual swing PRICE is bar (t-n)'s high.
    raw_sh_series  = pd.Series(raw_swing_high, index=df.index)
    raw_sl_series  = pd.Series(raw_swing_low,  index=df.index)
    df["swing_high"] = raw_sh_series.shift(n).fillna(0).astype(int)
    df["swing_low"]  = raw_sl_series.shift(n).fillna(0).astype(int)

    # Track the actual swing PRICE (the high/low of the bar that was the swing,
    # which is `n` bars before the confirmation bar).
    swing_high_price = df["high"].shift(n)
    swing_low_price  = df["low"].shift(n)
    swing_high_price = swing_high_price.where(df["swing_high"] == 1, np.nan)
    swing_low_price  = swing_low_price.where(df["swing_low"]  == 1, np.nan)

    # ── Step 3: BOS/CHoCH detection using causal (shifted) swings ──
    df["bos_bull"] = 0
    df["bos_bear"] = 0
    df["choch"]    = 0

    last_sh = None  # high of the last confirmed swing
    last_sl = None  # low  of the last confirmed swing
    trend   = 0     # 1 = bullish, -1 = bearish

    sh_flags = df["swing_high"].to_numpy()
    sl_flags = df["swing_low"].to_numpy()
    sh_prices = swing_high_price.to_numpy()
    sl_prices = swing_low_price.to_numpy()

    for i in range(len(df)):
        if sh_flags[i] == 1:
            cur_sh = sh_prices[i]
            if last_sh is not None and cur_sh > last_sh:
                if trend == 1:
                    df.at[df.index[i], "bos_bull"] = 1
                else:
                    df.at[df.index[i], "choch"] = 1
                    trend = 1
            last_sh = cur_sh

        if sl_flags[i] == 1:
            cur_sl = sl_prices[i]
            if last_sl is not None and cur_sl < last_sl:
                if trend == -1:
                    df.at[df.index[i], "bos_bear"] = 1
                else:
                    df.at[df.index[i], "choch"] = 1
                    trend = -1
            last_sl = cur_sl

    # Trend label: 1 = bullish structure, -1 = bearish
    df["trend"] = 0
    df.loc[df["bos_bull"] == 1, "trend"]  =  1
    df.loc[df["bos_bear"] == 1, "trend"]  = -1
    df["trend"] = df["trend"].replace(0, np.nan).ffill().fillna(0)

    return df


# ─────────────────────────────────────────────
# 4. H4 BIAS
# ─────────────────────────────────────────────
def compute_h4_bias(df_h4: pd.DataFrame, df_target: pd.DataFrame,
                    h4_is_close_stamped: bool = False) -> pd.DataFrame:
    """
    Compute the H4 directional bias and merge it onto the target timeframe.
    Cue always checks H4 first to determine if he's looking for buys or sells.

    CAUSALITY FIX (2026-06-17, take 2): MT5 H4 bars are timestamped at their
    OPEN time (a bar at 04:00 means the bar that OPENED at 04:00 and CLOSES
    at 08:00). The previous version of this function used merge_asof(direction=
    "backward") on the raw H4 OPEN-indexed series, which matched the 04:00 H4
    bar onto M15 bar 04:15 — even though that H4 bar wouldn't close until 08:00.
    Result: every M15 bar between an H4 open and the next H4 open received a
    h4_bias value derived from an H4 bar that was still in progress (avg
    ~120 min of forward info per M15 bar). The h4_bias_causality harness test
    caught 400/400 leaks on this code.

    The fix: shift the bias index forward by 4 hours so it represents the
    CLOSE time of each H4 bar. After this shift, merge_asof(direction=
    "backward") only matches H4 bars whose CLOSE is <= the M15 bar's
    timestamp, which is the true causal constraint.

    DOUBLE-SHIFT FIX (2026-07-15): the ablation pipeline builds df_h4 via
    aggregate_to_tf(), which ALREADY produces a close-time-indexed series.
    Applying the internal +4h shift on that input stamped h4_bias +4h LATER
    than the true H4 close, causing M15/H1 bars in [close, close+4h) to
    receive stale bias (previous H4 bar) or 0 fallback.  Callers that pass a
    close-stamped H4 series must set h4_is_close_stamped=True to skip the
    internal shift.
    """
    # Use market structure trend on H4 as bias (causally-correct at H4 close
    # time thanks to the swing-shift fix in detect_market_structure).
    df_h4 = detect_market_structure(df_h4)
    bias = df_h4[["trend"]].rename(columns={"trend": "h4_bias"})

    # Resample bias to target timeframe index via forward fill
    df_target = df_target.copy()
    bias.index = bias.index.tz_convert(df_target.index.tz) if bias.index.tz else bias.index

    # CAUSALITY FIX: shift the bias index forward by 4 hours so it represents
    # the H4 bar's CLOSE time (not its OPEN time). This ensures the merge
    # below only attaches an H4 bias value to M15 bars at or AFTER the H4
    # bar has fully closed.  Skip when caller already passes close-stamped H4.
    bias = bias.copy()
    if not h4_is_close_stamped:
        bias.index = bias.index + pd.Timedelta(hours=4)

    # Merge on nearest H4 timestamp
    df_target = df_target.sort_index()
    bias = bias.sort_index()
    df_target = pd.merge_asof(
        df_target,
        bias,
        left_index=True,
        right_index=True,
        direction="backward"
    )
    df_target["h4_bias"] = df_target["h4_bias"].fillna(0)
    return df_target


# ─────────────────────────────────────────────
# 5. SESSION TIMING
# ─────────────────────────────────────────────
def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cue primarily trades London Open (3–5 AM EST) and NY Open (9:30–11 AM EST).
    These sessions have the highest institutional volume and clearest FVG fills.
    """
    df = df.copy()
    eastern = pytz.timezone("US/Eastern")

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    local_times = df.index.tz_convert(eastern)
    hours = local_times.hour + local_times.minute / 60

    # London Open: 3:00 AM – 5:00 AM EST
    df["london_session"] = ((hours >= 3.0) & (hours < 5.0)).astype(int)

    # NY Open: 9:30 AM – 11:00 AM EST
    df["ny_session"] = ((hours >= 9.5) & (hours < 11.0)).astype(int)

    # Combined high-probability window
    df["prime_session"] = ((df["london_session"] == 1) | (df["ny_session"] == 1)).astype(int)

    # Hour of day (cyclical encoding for ML)
    df["hour_sin"] = np.sin(2 * np.pi * local_times.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * local_times.hour / 24)

    # Day of week (Mon=0, Fri=4) — avoid Mondays/Fridays (lower quality setups)
    df["day_of_week"] = local_times.dayofweek
    df["is_prime_day"] = df["day_of_week"].isin([1, 2, 3]).astype(int)  # Tue-Thu

    return df


# ─────────────────────────────────────────────
# 6. SUPPORTING INDICATORS (context for ML)
# ─────────────────────────────────────────────
def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """RSI, ATR, candle body/wick ratios — supporting context for the LSTM."""
    df = df.copy()

    # RSI (14)
    delta  = df["close"].diff()
    gain   = delta.clip(lower=0).rolling(14).mean()
    loss   = (-delta.clip(upper=0)).rolling(14).mean()
    rs     = gain / (loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    # ATR (14) — volatility
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"]  - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    # Candle body and wick ratios
    df["body_size"]  = (df["close"] - df["open"]).abs() / (df["atr"] + 1e-9)
    df["upper_wick"] = (df["high"]  - df[["open", "close"]].max(axis=1)) / (df["atr"] + 1e-9)
    df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / (df["atr"] + 1e-9)
    df["is_bullish_candle"] = (df["close"] > df["open"]).astype(int)

    # Volume relative to 20-period average
    df["volume_ratio"] = df["volume"] / (df["volume"].rolling(20).mean() + 1e-9)

    # Price momentum
    df["momentum_5"]  = df["close"].pct_change(5)
    df["momentum_10"] = df["close"].pct_change(10)

    return df

# NOTE: Scale-invariant vol/autocorr features (atr_pct, realized_vol_*,
# autocorr_*, variance_ratio_10) were evaluated for V75 on 2026-04-23.
# Result: all had |corr with label| < 0.02 and adding them dropped WF
# Sharpe 8.88 -> 6.90 with spread; re-tune still picked the same shallow
# tree params. V75 is near-IID GBM — its edge comes from hour encoding
# and microstructure, not vol structure. Keeping the feature set lean.


# ─────────────────────────────────────────────
# 6b. ICT PRICE ACTION MODEL FEATURES
#     Source: ICT Charter Price Action Models 1-12
#     Decoded from video via AI Video Decoder, integrated 2026-06-07
# ─────────────────────────────────────────────

def detect_premium_discount(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    ICT Premium/Discount + OTE (Optimal Trade Entry) zones.

    Core ICT concept: divide the current range into premium (top half) and
    discount (bottom half). Institutional buyers accumulate in discount,
    sellers distribute in premium.

    OTE zone: the 62–79% Fibonacci retracement — ICT's highest-probability
    entry zone. In a bullish context, the OTE is in the lower portion of
    the range (21–38% from bottom). In bearish, it's 62–79% from bottom.

    Uses a rolling 20-bar range to approximate the IPDA (Interbank Price
    Delivery Algorithm) 20-day data range from ICT Model #1.
    """
    df = df.copy()

    # Rolling range (IPDA 20-bar proxy on whatever TF we're on)
    df["ipda_high"] = df["high"].rolling(lookback, min_periods=lookback).max()
    df["ipda_low"]  = df["low"].rolling(lookback, min_periods=lookback).min()
    range_size = df["ipda_high"] - df["ipda_low"]

    # Where price sits: 0 = at range low, 1 = at range high
    df["range_position"] = (df["close"] - df["ipda_low"]) / (range_size + 1e-9)
    df["range_position"] = df["range_position"].clip(0, 1)

    # Premium (top half) / Discount (bottom half)
    df["in_premium"]  = (df["range_position"] > 0.5).astype(int)
    df["in_discount"] = (df["range_position"] <= 0.5).astype(int)

    # OTE zones — Fib 62–79% retracement
    # Bullish OTE: price in discount OTE (21–38% of range from bottom)
    df["in_ote_bull"] = (
        (df["range_position"] >= 0.21) & (df["range_position"] <= 0.38)
    ).astype(int)
    # Bearish OTE: price in premium OTE (62–79% of range from bottom)
    df["in_ote_bear"] = (
        (df["range_position"] >= 0.62) & (df["range_position"] <= 0.79)
    ).astype(int)

    # Distance from IPDA extremes in ATR units (how far from key liquidity)
    df["dist_ipda_high"] = (df["ipda_high"] - df["close"]) / (df["atr"] + 1e-9)
    df["dist_ipda_low"]  = (df["close"] - df["ipda_low"])  / (df["atr"] + 1e-9)

    return df


def detect_liquidity_sweeps(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    ICT Liquidity Sweep (Stop Hunt / Turtle Soup).

    A liquidity sweep occurs when price spikes beyond a previous high or low
    (taking out stop-loss clusters / resting orders) but then closes back
    inside the range. This is a strong reversal signal.

    From ICT Models #1, #9: "The market seeks liquidity. External range
    liquidity (old highs/lows) gets swept before the real move."

    Uses previous bar's rolling high/low to avoid lookahead.
    """
    df = df.copy()

    # Previous bar's rolling extremes (shifted to avoid lookahead)
    prev_high = df["high"].rolling(lookback, min_periods=lookback).max().shift(1)
    prev_low  = df["low"].rolling(lookback, min_periods=lookback).min().shift(1)

    # Buyside sweep: wick above previous high, close back below it
    # (stop hunts above equal highs → bearish reversal signal)
    df["liq_sweep_high"] = (
        (df["high"] > prev_high) & (df["close"] < prev_high)
    ).astype(int).fillna(0)

    # Sellside sweep: wick below previous low, close back above it
    # (stop hunts below equal lows → bullish reversal signal)
    df["liq_sweep_low"] = (
        (df["low"] < prev_low) & (df["close"] > prev_low)
    ).astype(int).fillna(0)

    return df


def detect_displacement(df: pd.DataFrame, atr_mult: float = 1.5) -> pd.DataFrame:
    """
    ICT Displacement / Expansion Swing.

    A displacement candle has a body > 1.5× ATR — it signals strong
    institutional momentum. From ICT Model #12: "Wait for the order block
    to prove there is expansion... an expansion swing taking price higher."

    These candles often precede FVG formation and mark the start of
    a genuine institutional move vs. retail chop.
    """
    df = df.copy()

    body = (df["close"] - df["open"]).abs()

    # Bullish displacement: large bullish candle
    df["displacement_bull"] = (
        (body > atr_mult * df["atr"]) & (df["close"] > df["open"])
    ).astype(int).fillna(0)

    # Bearish displacement: large bearish candle
    df["displacement_bear"] = (
        (body > atr_mult * df["atr"]) & (df["close"] < df["open"])
    ).astype(int).fillna(0)

    return df


def add_prev_day_levels(df: pd.DataFrame) -> pd.DataFrame:
    """
    ICT Previous Day High/Low — key liquidity targets.

    From ICT Model #1: "We'll be buying... while anticipating a rally to
    the previous day's high" and "shorting... while anticipating a decline
    to the previous day's low."

    These are the most immediate external liquidity pools that price draws
    towards during each session.
    """
    df = df.copy()

    # Extract date from index for grouping
    dates = df.index.normalize()

    # Compute daily high/low, then shift by 1 day to get PREVIOUS day
    daily_high = df.groupby(dates)["high"].transform("max")
    daily_low  = df.groupby(dates)["low"].transform("min")

    # Shift forward by the number of bars in the first full day to align
    # "previous day" correctly. Use a date-based approach instead:
    unique_dates = dates.unique()
    prev_day_high = pd.Series(np.nan, index=df.index, dtype=float)
    prev_day_low  = pd.Series(np.nan, index=df.index, dtype=float)

    for i in range(1, len(unique_dates)):
        today_mask    = dates == unique_dates[i]
        prev_day_mask = dates == unique_dates[i - 1]
        prev_hi = df.loc[prev_day_mask, "high"].max()
        prev_lo = df.loc[prev_day_mask, "low"].min()
        prev_day_high.loc[today_mask] = prev_hi
        prev_day_low.loc[today_mask]  = prev_lo

    # Distance in ATR units (positive = price below target for high, above for low)
    df["dist_prev_high"] = (prev_day_high - df["close"]) / (df["atr"] + 1e-9)
    df["dist_prev_low"]  = (df["close"] - prev_day_low)  / (df["atr"] + 1e-9)

    return df


def detect_fvg_in_displacement(df: pd.DataFrame) -> pd.DataFrame:
    """
    ICT Model #12 entry technique: FVG within an Expansion Swing.

    From the video: "What we're doing is showing you how to find fair value
    gaps in the middle of price runs." After a displacement candle, a
    subsequent FVG is especially high-quality because it formed during
    institutional momentum, not random chop.

    This feature flags FVGs that formed within 3 bars of a displacement.
    """
    df = df.copy()

    # Check if a displacement occurred in the last 3 bars
    disp_bull_recent = df["displacement_bull"].rolling(3, min_periods=1).max()
    disp_bear_recent = df["displacement_bear"].rolling(3, min_periods=1).max()

    # FVG that formed during/after bullish displacement = high-quality bullish FVG
    df["fvg_displacement_bull"] = (
        (df["fvg_bull"] == 1) & (disp_bull_recent == 1)
    ).astype(int)

    # FVG that formed during/after bearish displacement = high-quality bearish FVG
    df["fvg_displacement_bear"] = (
        (df["fvg_bear"] == 1) & (disp_bear_recent == 1)
    ).astype(int)

    return df


# ─────────────────────────────────────────────
# 7. SIGNAL LABELS (for supervised learning)
# ─────────────────────────────────────────────
def create_labels(df: pd.DataFrame, forward_bars: int = 24,
                  atr_tp_mult: float = 3.0, atr_sl_mult: float = 1.5) -> pd.DataFrame:
    """
    Triple-barrier labelling that mirrors the backtest executor exactly.
    For each bar we simulate BOTH a long and a short bracket trade and mark
    which one (if any) hits its take-profit before its stop-loss inside the
    next `forward_bars` bars.
      1 = BUY   long  TP hit first, short would have stopped out or timed out
     -1 = SELL  short TP hit first, long  would have stopped out or timed out
      0 = HOLD  neither side would have won, or both did (ambiguous)
    Requires df["atr"] to be present (added by add_momentum_indicators).
    """
    df = df.copy()
    n = len(df)
    close = df["close"].to_numpy()
    high  = df["high"].to_numpy()
    low   = df["low"].to_numpy()
    atr   = df["atr"].to_numpy()

    # Use float so we can mark the tail (incomplete forward window) as NaN.
    labels = np.zeros(n, dtype=np.float32)
    labels[-forward_bars:] = np.nan

    for i in range(n - forward_bars):
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        entry = close[i]
        tp_long  = entry + atr_tp_mult * a
        sl_long  = entry - atr_sl_mult * a
        tp_short = entry - atr_tp_mult * a
        sl_short = entry + atr_sl_mult * a

        long_win  = False
        short_win = False
        # Walk forward bar by bar. Whichever barrier is touched first wins.
        # If both TP and SL are inside the same bar's range, assume worst case (SL
        # hit first) — keeps labels conservative and matches how the executor
        # would handle gapping.
        for j in range(i + 1, i + 1 + forward_bars):
            hj, lj = high[j], low[j]
            if not long_win:
                if lj <= sl_long:
                    long_result = -1
                    long_win = True   # decided (stopped)
                    long_hit = False
                elif hj >= tp_long:
                    long_result = 1
                    long_win = True
                    long_hit = True
            if not short_win:
                if hj >= sl_short:
                    short_result = -1
                    short_win = True
                    short_hit = False
                elif lj <= tp_short:
                    short_result = 1
                    short_win = True
                    short_hit = True
            if long_win and short_win:
                break

        # Final label: prefer the side that hit TP; if both/neither, HOLD.
        long_hit_tp  = long_win  and long_result == 1
        short_hit_tp = short_win and short_result == 1
        if long_hit_tp and not short_hit_tp:
            labels[i] = 1
        elif short_hit_tp and not long_hit_tp:
            labels[i] = -1
        # else 0 (HOLD)

    df["label"] = labels

    # NOTE: Cue's "never trade against H4 bias" rule is enforced at EXECUTION time
    # (in SMCTrader.predict), not at labeling time. This lets the model learn both
    # bullish and bearish setups from raw price action even when the H4 window is
    # one-sided — the filter still gets applied before any trade actually fires.

    return df


# ─────────────────────────────────────────────
# 8. FULL PIPELINE
# ─────────────────────────────────────────────
def build_features(df_target: pd.DataFrame, df_h4: pd.DataFrame = None) -> pd.DataFrame:
    """
    Run the full SMC + ICT feature pipeline on a target timeframe DataFrame.
    Optionally pass H4 data for bias computation.

    Pipeline order matters: momentum indicators (ATR) must come before the
    ICT features that depend on ATR for normalization.
    """
    df = df_target.copy()

    print("[Features] Detecting Fair Value Gaps...")
    df = detect_fvg(df)

    print("[Features] Detecting Order Blocks...")
    df = detect_order_blocks(df)

    print("[Features] Detecting Market Structure...")
    df = detect_market_structure(df)

    print("[Features] Adding Session Timing...")
    df = add_session_features(df)

    print("[Features] Adding Momentum Indicators...")
    df = add_momentum_indicators(df)

    if df_h4 is not None:
        print("[Features] Computing H4 Bias...")
        df = compute_h4_bias(df_h4, df)

    # ── ICT Price Action Model features (require ATR to be computed first) ──
    print("[Features] ICT: Premium/Discount + OTE zones...")
    df = detect_premium_discount(df)

    print("[Features] ICT: Liquidity Sweeps...")
    df = detect_liquidity_sweeps(df)

    print("[Features] ICT: Displacement / Expansion Swings...")
    df = detect_displacement(df)

    print("[Features] ICT: Previous Day High/Low targets...")
    df = add_prev_day_levels(df)

    print("[Features] ICT: FVG within Displacement...")
    df = detect_fvg_in_displacement(df)

    print("[Features] Creating Trade Labels...")
    df = create_labels(df)

    # Drop rows where actual training columns or label are NaN.
    # Do NOT use bare dropna() — intermediate columns like fvg_bull_top and
    # ob_bull_high are NaN on most rows by design (only set when a pattern fires).
    df.dropna(subset=FEATURE_COLS + ["label"], inplace=True)

    print(f"[Features] Done. {len(df)} labeled samples.")
    print(f"  Buy: {(df['label']==1).sum()} | Sell: {(df['label']==-1).sum()} | Hold: {(df['label']==0).sum()}")

    return df


FEATURE_COLS = [
    # ── Original SMC features ──
    "fvg_bull", "fvg_bear", "fvg_bull_dist", "fvg_bear_dist",
    "ob_bull", "ob_bear", "in_bull_ob", "in_bear_ob",
    "swing_high", "swing_low", "bos_bull", "bos_bear", "choch", "trend",
    "london_session", "ny_session", "prime_session",
    "hour_sin", "hour_cos", "is_prime_day",
    "rsi", "atr", "body_size", "upper_wick", "lower_wick",
    "is_bullish_candle", "volume_ratio", "momentum_5", "momentum_10",
    "h4_bias",
    # ── ICT Price Action Model features (added 2026-06-07) ──
    "range_position",                              # IPDA range context (0=low, 1=high)
    "in_premium", "in_discount",                   # ICT Premium/Discount zones
    "in_ote_bull", "in_ote_bear",                  # Optimal Trade Entry zones (Fib 62-79%)
    "dist_ipda_high", "dist_ipda_low",             # Distance from IPDA 20-bar extremes
    "liq_sweep_high", "liq_sweep_low",             # Liquidity sweeps (stop hunts)
    "displacement_bull", "displacement_bear",       # Expansion swings (institutional momentum)
    "dist_prev_high", "dist_prev_low",             # Previous day high/low targets
    "fvg_displacement_bull", "fvg_displacement_bear",  # FVG within displacement (Model #12)
]
