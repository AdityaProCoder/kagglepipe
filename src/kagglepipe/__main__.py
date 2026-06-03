"""Allow `python -m kagglepipe ...`."""

from kagglepipe.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
