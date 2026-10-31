class GoldPanException(Exception):
    """Base exception for GoldPan"""

    pass


class ExtractionError(GoldPanException):
    """Error during extraction phase"""

    pass


class FilterError(GoldPanException):
    """Error during filtering phase"""

    pass


class FormattingError(GoldPanException):
    """Error during formatting phase"""

    pass


class PromptInjectionBlockedError(GoldPanException):
    """Raised when AgentContextFirewall in 'strict' mode blocks a high-risk prompt injection payload."""

    def __init__(self, message: str, risk_score: float = 1.0, signals: list[str] | None = None) -> None:
        super().__init__(message)
        self.risk_score = risk_score
        self.signals = signals or []
