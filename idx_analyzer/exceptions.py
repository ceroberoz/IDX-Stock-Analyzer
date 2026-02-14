"""
Custom exceptions for IDX Analyzer

Provides specific exception types for different error scenarios
to enable better error handling and user feedback.
"""


class IDXAnalyzerError(Exception):
    """Base exception for all IDX Analyzer errors"""

    def __init__(self, message: str, details: str = None):
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self):
        if self.details:
            return f"{self.message}\n  Details: {self.details}"
        return self.message


class ConfigurationError(IDXAnalyzerError):
    """Raised when there's an issue with configuration"""

    pass


class DataFetchError(IDXAnalyzerError):
    """Raised when data fetching fails"""

    def __init__(self, message: str, ticker: str = None, details: str = None):
        super().__init__(message, details)
        self.ticker = ticker


class NetworkError(DataFetchError):
    """Raised when network request fails"""

    def __init__(
        self,
        message: str = "Network request failed",
        ticker: str = None,
        retry_count: int = 0,
        details: str = None,
    ):
        super().__init__(message, ticker, details)
        self.retry_count = retry_count


class InvalidTickerError(DataFetchError):
    """Raised when ticker symbol is invalid or not found"""

    def __init__(self, ticker: str, details: str = None):
        message = f"Invalid or unknown ticker: '{ticker}'"
        super().__init__(message, ticker, details)


class InsufficientDataError(DataFetchError):
    """Raised when there's not enough data for analysis"""

    def __init__(
        self,
        ticker: str,
        data_points: int = 0,
        required: int = 0,
        details: str = None,
    ):
        message = f"Insufficient data for {ticker}"
        if data_points and required:
            message += f" ({data_points} points, {required} required)"
        super().__init__(message, ticker, details)
        self.data_points = data_points
        self.required = required


class AnalysisError(IDXAnalyzerError):
    """Raised when analysis calculation fails"""

    pass


class ChartError(IDXAnalyzerError):
    """Raised when chart generation fails"""

    pass


class ExportError(IDXAnalyzerError):
    """Raised when data export fails"""

    pass


def format_error_for_user(error: Exception) -> str:
    """Format an error message for end-user display"""
    if isinstance(error, InvalidTickerError):
        return f"""❌ {error.message}

💡 Tips:
   • Check the ticker spelling (e.g., 'BBCA' not 'BB CA')
   • Use the stock code without '.JK' suffix
   • Verify the stock is listed on IDX
   • Try: BBCA, BBRI, TLKM, ASII, UNVR"""

    elif isinstance(error, NetworkError):
        msg = f"❌ {error.message}\n\n💡 Tips:\n   • Check your internet connection\n   • Yahoo Finance may be temporarily unavailable\n   • Try again in a few moments"
        if error.retry_count > 0:
            msg += f"\n   • Failed after {error.retry_count} retry attempts"
        return msg

    elif isinstance(error, InsufficientDataError):
        return f"""❌ {error.message}

💡 Tips:
   • Try a longer period: --period 1y or --period 2y
   • The stock may be newly listed
   • Try a different ticker"""

    elif isinstance(error, ConfigurationError):
        return f"""❌ Configuration Error: {error.message}

💡 Tips:
   • Check your config file syntax
   • Run with --help to see default options
   • Delete config file to reset to defaults"""

    elif isinstance(error, DataFetchError):
        return f"❌ Failed to fetch data for {error.ticker}: {error.message}"

    elif isinstance(error, IDXAnalyzerError):
        return f"❌ {error.message}"

    else:
        # Generic error
        return f"❌ Unexpected error: {error}"
