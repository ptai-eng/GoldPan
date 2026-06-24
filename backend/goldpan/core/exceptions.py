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
