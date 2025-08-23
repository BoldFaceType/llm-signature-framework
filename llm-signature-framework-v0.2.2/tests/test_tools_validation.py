import asyncio
import pytest
from llm_signature_framework.tools import (
    Tool,
    ToolRegistry,
    InputValidationError,
    OutputValidationError,
    FatalToolError,
)


@Tool(name="adder")
def add(a: int, b: int) -> int:
    return a + b


@Tool(name="badadder")
def bad(a: int, b: int) -> int:
    return "not an int"


def run(coro):
    return asyncio.run(coro)


def test_tool_success_and_validation_errors():
    assert run(ToolRegistry.call("adder", a=1, b=2)) == 3
    with pytest.raises(InputValidationError):
        run(ToolRegistry.call("adder", a="x", b=2))
    with pytest.raises(OutputValidationError):
        run(ToolRegistry.call("badadder", a=1, b=2))
    with pytest.raises(FatalToolError):
        run(ToolRegistry.call("unknown"))
