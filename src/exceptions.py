class RiskEngineError(Exception):
    """Base exception for all risk engine errors."""
    pass


class DataError(RiskEngineError):
    """Data fetch or quality failure."""
    pass


class ConvergenceError(RiskEngineError):
    """GARCH or optimizer failed to converge."""
    pass


class ValidationError(RiskEngineError, ValueError):
    """Input validation failed."""
    pass
