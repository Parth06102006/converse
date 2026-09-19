"""Typed domain exceptions for Automatic Speech Recognition (ASR) subsystem."""


class AsrError(Exception):
    """Base exception for all ASR pipeline failures."""

    def __init__(self, message: str, code: str = "ASR_ERROR") -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class AsrSessionNotFoundError(AsrError):
    """Raised when an operation is attempted on an unknown or expired session."""

    def __init__(self, session_id: str) -> None:
        super().__init__(f"ASR session not found: {session_id}", code="SESSION_NOT_FOUND")
        self.session_id = session_id


class AsrAudioDecodingError(AsrError):
    """Raised when audio payload cannot be decoded or validated."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="AUDIO_DECODING_ERROR")


class AwsAsrError(AsrError):
    """Base exception for Amazon Transcribe Streaming errors."""

    def __init__(self, message: str, code: str = "AWS_ASR_ERROR") -> None:
        super().__init__(message, code=code)


class AwsAsrAuthError(AwsAsrError):
    """Raised when AWS credentials, IAM permissions, or subscriptions are missing/invalid."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="AUTHENTICATION_FAILED")


class AwsAsrServiceUnavailableError(AwsAsrError):
    """Raised when Amazon Transcribe Streaming service or network connection is unavailable."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="SERVICE_UNAVAILABLE")


class AwsAsrConfigurationError(AwsAsrError):
    """Raised when AWS Transcribe parameters (region, language, sample rate) are invalid."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="CONFIGURATION_ERROR")
