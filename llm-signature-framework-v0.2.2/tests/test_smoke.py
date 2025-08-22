from llm_signature_framework import LLMFunction

def test_summarize():
    llm = LLMFunction()
    @llm
    def summarize(text: str) -> str:
        """Summarize: {text}"""
        pass
    out = summarize("ok")
    assert isinstance(out, str)
