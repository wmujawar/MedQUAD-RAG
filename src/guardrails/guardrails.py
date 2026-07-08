from dataclasses import dataclass

import structlog
from guardrails.errors import ValidationError
from guardrails.hub import DetectPII, GibberishText, ToxicLanguage

from guardrails import Guard, OnFailAction

TOXIC_THRESHOLD = 0.5  # 0.0 (strict) → 1.0 (lenient)
GIBBERISH_THRESHOLD = 0.5  # same scale
TOXIC_METHOD = "sentence"  # "sentence" = per sentence | "full" = whole text
PII_INPUT_ENTITIES = [  # block these in user input
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SSN",
    "CREDIT_CARD",
]
PII_OUTPUT_ENTITIES = [  # redact these from LLM output
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
]


@dataclass
class GuardResult:
    passed: bool
    text: str
    error: str | None


def _input_validator() -> Guard:
    return (
        Guard()
        .use(
            GibberishText(
                threshold=GIBBERISH_THRESHOLD,
                validation_method=TOXIC_METHOD,
                on_fail=OnFailAction.FILTER,
            )
        )
        .use(
            ToxicLanguage(
                threshold=TOXIC_THRESHOLD,
                validation_method=TOXIC_METHOD,
                on_fail=OnFailAction.FILTER,
            )
        )
        .use(
            DetectPII(
                pii_entities=PII_OUTPUT_ENTITIES,
                on_fail=OnFailAction.FIX,
            )
        )
    )


def _output_validator() -> Guard:
    return (
        Guard()
        .use(
            DetectPII(
                pii_entities=PII_OUTPUT_ENTITIES,
                on_fail=OnFailAction.FIX,
            )
        )
        .use(
            ToxicLanguage(
                threshold=TOXIC_THRESHOLD,
                validation_method=TOXIC_METHOD,
                on_fail=OnFailAction.FIX,
            )
        )
    )


logger = structlog.get_logger(__file__)


class GuardRailChecker:
    """
    Stateless utility class that runs input and output guards
    and returns a clean GuardResult — never raises to the caller.

    Usage:
        checker = GuardRailChecker()
        result = checker.validate_input(user_message)
        if not result.passed:
            return result.error   # blocked

        llm_response = llm.invoke(result.text)

        result = runner.validate_output(llm_response)
        return result.text        # safe, cleaned response
    """

    def __init__(self):
        self._input_validator = _input_validator()
        self._output_validator = _output_validator()

    def validate_input(self, text: str) -> GuardResult:
        """
        Returns GuardResult(passed=False) if any validator raises.
        """
        try:
            result = self._input_validator.validate(text)
            logger.debug(
                "input_guard.passed",
                query=text[:100],
            )
            return GuardResult(
                passed=True,
                error=None,
                text=result.validated_output
                if result.validated_output is not None
                else "",
            )

        except ValidationError as e:
            logger.exception(
                "input_guard.failed",
                error=str(e),
            )
            return GuardResult(passed=False, error=str(e), text="")

    def validate_output(self, text: str) -> GuardResult:
        """
        Run output guard
        """
        try:
            result = self._output_validator.validate(text)
            logger.debug(
                "output_guard.passed",
                query=text[:100],
            )
            return GuardResult(
                passed=True,
                error=None,
                text=result.validated_output
                if result.validated_output is not None
                else "",
            )

        except ValidationError as e:
            logger.exception(
                "output_guard.failed",
                error=str(e),
            )
            return GuardResult(passed=False, error=str(e), text="")
