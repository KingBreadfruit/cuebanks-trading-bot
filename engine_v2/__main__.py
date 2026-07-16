"""Allow `python -m trading_bot.engine_v2 ...` invocation."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
