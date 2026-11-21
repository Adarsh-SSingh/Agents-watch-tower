"""Argument schemas. LangChain validates every tool call against these."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    # Unknown arguments are rejected (additionalProperties: false).
    model_config = ConfigDict(extra="forbid")


class CalculatorInput(_Strict):
    expression: str = Field(
        description="Mathematical expression, e.g. sqrt(144) or 25 * 4"
    )


class SystemInfoInput(_Strict):
    system_info: str = Field(description="Pass an empty string.")


class WeatherInput(_Strict):
    city: str = Field(description="City name, e.g. Bengaluru")


class WikipediaInput(_Strict):
    topic: str = Field(description="Topic to search on Wikipedia")


class FlakyToolInput(_Strict):
    mode: Literal["success", "delay", "error", "timeout", "intermittent"] = Field(
        description="Execution behavior"
    )
    delay_seconds: float = Field(
        default=0, description="Delay duration between 0 and 30 seconds"
    )
    failure_message: str = Field(
        default="Simulated tool failure",
        description="Custom error message for error mode",
    )
    timeout_seconds: float = Field(
        default=3,
        description=(
            "Maximum allowed execution time in seconds (0 to 30). "
            "Used by timeout mode."
        ),
    )