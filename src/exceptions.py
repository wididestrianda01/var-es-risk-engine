class RiskEngineError(Exception):
    """Base exception for all risk engine errors."""


class DataError(RiskEngineError):
    """Data fetch or quality failure."""


class ConvergenceError(RiskEngineError):
    """GARCH or optimizer failed to converge."""


class ValidationError(RiskEngineError, ValueError):
    """Input validation failed."""
