import pytest
from llm_signature_framework.templates import _MiniTemplate


def test_minitemplate_loop_and_if():
    tmpl_text = (
        "Hi {name}!"
        "{% for n in nums %}[{n}]{% endfor %}"
        "{% if flag %}Y{% elif alt %}N{% else %}Z{% endif %}"
    )
    tmpl = _MiniTemplate(tmpl_text)
    result = tmpl.render({"name": "Ann", "nums": [1, 2], "flag": False, "alt": True})
    assert result == "Hi Ann![1][2]N"


def test_minitemplate_unclosed_for():
    tmpl = _MiniTemplate("{% for x in items %}{x}")
    with pytest.raises(ValueError):
        tmpl.render({"items": [1, 2]})
