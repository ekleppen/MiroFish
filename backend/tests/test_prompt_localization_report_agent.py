import re
import string

import pytest

from app.services import report_agent as ra
from app.services.report_agent import ReportAgent
from app.utils.locale import localized, set_locale

HAN = re.compile(r'[\u4e00-\u9fff]')

PAIRS = [
    'TOOL_DESC_INSIGHT_FORGE',
    'TOOL_DESC_PANORAMA_SEARCH',
    'TOOL_DESC_QUICK_SEARCH',
    'TOOL_DESC_INTERVIEW_AGENTS',
    'PLAN_SYSTEM_PROMPT',
    'PLAN_USER_PROMPT_TEMPLATE',
    'SECTION_SYSTEM_PROMPT_TEMPLATE',
    'SECTION_USER_PROMPT_TEMPLATE',
    'REACT_OBSERVATION_TEMPLATE',
    'REACT_INSUFFICIENT_TOOLS_MSG',
    'REACT_INSUFFICIENT_TOOLS_MSG_ALT',
    'REACT_TOOL_LIMIT_MSG',
    'REACT_UNUSED_TOOLS_HINT',
    'REACT_FORCE_FINAL_MSG',
    'CHAT_SYSTEM_PROMPT_TEMPLATE',
    'CHAT_OBSERVATION_SUFFIX',
]

FORMAT_TEMPLATES = {
    'PLAN_USER_PROMPT_TEMPLATE',
    'SECTION_SYSTEM_PROMPT_TEMPLATE',
    'SECTION_USER_PROMPT_TEMPLATE',
    'REACT_OBSERVATION_TEMPLATE',
    'REACT_INSUFFICIENT_TOOLS_MSG',
    'REACT_INSUFFICIENT_TOOLS_MSG_ALT',
    'REACT_TOOL_LIMIT_MSG',
    'REACT_UNUSED_TOOLS_HINT',
    'CHAT_SYSTEM_PROMPT_TEMPLATE',
}


def _fields(template):
    return sorted(f for _, f, _, _ in string.Formatter().parse(template) if f is not None)


@pytest.fixture(autouse=True)
def _reset_locale():
    yield
    set_locale('zh')


@pytest.mark.parametrize('name', PAIRS)
def test_english_prompt_has_no_chinese(name):
    en = getattr(ra, f'{name}_EN')
    assert not HAN.search(en), HAN.findall(en)


@pytest.mark.parametrize('name', sorted(FORMAT_TEMPLATES))
def test_format_fields_match(name):
    zh = getattr(ra, name)
    en = getattr(ra, f'{name}_EN')
    assert _fields(en) == _fields(zh)
    kwargs = {f: 'x' for f in _fields(zh)}
    zh.format(**kwargs)
    en.format(**kwargs)


@pytest.mark.parametrize('name', [n for n in PAIRS if 'TOOL_DESC' in n or n.startswith('REACT') or 'SECTION' in n])
def test_final_answer_and_tool_names_preserved(name):
    zh = getattr(ra, name)
    en = getattr(ra, f'{name}_EN')
    assert ('Final Answer:' in zh) == ('Final Answer:' in en)
    assert zh.count('<tool_call>') == en.count('<tool_call>')
    for tool in ('insight_forge', 'panorama_search', 'quick_search', 'interview_agents'):
        assert zh.count(tool) == en.count(tool)


def test_localized_picks_by_locale():
    set_locale('en')
    assert localized(ra.SECTION_SYSTEM_PROMPT_TEMPLATE, ra.SECTION_SYSTEM_PROMPT_TEMPLATE_EN) == ra.SECTION_SYSTEM_PROMPT_TEMPLATE_EN
    set_locale('zh')
    assert localized(ra.SECTION_SYSTEM_PROMPT_TEMPLATE, ra.SECTION_SYSTEM_PROMPT_TEMPLATE_EN) == ra.SECTION_SYSTEM_PROMPT_TEMPLATE


def test_tool_descriptions_follow_locale():
    agent = object.__new__(ReportAgent)

    set_locale('en')
    agent.tools = agent._define_tools()
    en_desc = agent._get_tools_description()
    assert not HAN.search(en_desc), HAN.findall(en_desc)
    assert en_desc.startswith('Available tools:')

    set_locale('zh')
    agent.tools = agent._define_tools()
    zh_desc = agent._get_tools_description()
    assert zh_desc.startswith('可用工具：')
    assert ra.TOOL_DESC_INSIGHT_FORGE in zh_desc
