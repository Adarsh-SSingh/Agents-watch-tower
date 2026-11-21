"""The Tool Agent's tools, registered as LangChain StructuredTools."""

from langchain_core.tools import StructuredTool

from agentops.agents.tool.tools.calculator import calculate
from agentops.agents.tool.tools.flaky import flaky_tool
from agentops.agents.tool.tools.schemas import (
    CalculatorInput,
    FlakyToolInput,
    SystemInfoInput,
    WeatherInput,
    WikipediaInput,
)
from agentops.agents.tool.tools.system_info import get_system_info
from agentops.agents.tool.tools.weather import get_weather
from agentops.agents.tool.tools.wikipedia import wikipedia_lookup


TOOLS = [
    StructuredTool.from_function(
        func=calculate,
        name="calculator",
        description="Perform mathematical and scientific calculations.",
        args_schema=CalculatorInput,
    ),
    StructuredTool.from_function(
        func=get_system_info,
        name="get_system_info",
        description=(
            "Retrieve CPU, RAM, disk, network, uptime "
            "and process information of the local system."
        ),
        args_schema=SystemInfoInput,
    ),
    StructuredTool.from_function(
        func=get_weather,
        name="get_weather",
        description="Get current weather information for a city.",
        args_schema=WeatherInput,
    ),
    StructuredTool.from_function(
        func=wikipedia_lookup,
        name="wikipedia_lookup",
        description="Search Wikipedia and retrieve an introductory summary about a topic.",
        args_schema=WikipediaInput,
    ),
    StructuredTool.from_function(
        func=flaky_tool,
        name="flaky_tool",
        description=(
            "Testing tool that simulates success, delay, error, "
            "timeout or intermittent failure."
        ),
        args_schema=FlakyToolInput,
    ),
]

TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}

__all__ = ["TOOLS", "TOOLS_BY_NAME"]