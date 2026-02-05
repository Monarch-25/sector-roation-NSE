# Strategy Documentation

This folder contains detailed documentation for each trading strategy.

## Available Strategies

| Strategy | File | Description |
|----------|------|-------------|
| [RRG Strategy](rrg_strategy.md) | `rrg_strategy.py` | Relative Rotation Graph momentum-based rotation |
| [Volume Collapse](volume_collapse_strategy.md) | `volume_collapse_strategy.py` | RRG + geometric volume regime filter |

## Adding New Strategies

1. Implement in `strategies/` following `STRATEGY_GUIDE.md`
2. Add documentation here as `{strategy_name}.md`
3. Export from `strategies/__init__.py`
