from llm_signature_framework.core import LLMFunction, ToolRegistry

llm = LLMFunction()

@llm
def summarize(text: str) -> str:
    """Summarize: {text}"""
    pass

print("Summ:", summarize("This framework rocks."))
print("Tools:", ToolRegistry.list_tools())
