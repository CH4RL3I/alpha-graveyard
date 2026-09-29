"""Fixed experimental design. Changing anything here changes the experiment."""

UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "VNQ", "HYG", "LQD"]

# Discovery: every date strictly before SPLIT_DATE. Validation: SPLIT_DATE onwards.
# The search never receives a price dated on or after SPLIT_DATE.
SPLIT_DATE = "2017-01-01"

# One-way cost per unit of turnover: 2 bps commission/half-spread + 3 bps slippage.
COST_BPS = 5.0
TRADING_DAYS = 252

# A candidate needs at least this many position changes in discovery to be tradeable.
MIN_CHANGES = 20
