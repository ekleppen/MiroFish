import re

import pytest

from app.services.zep_tools import (
    AgentInterview,
    EdgeInfo,
    InsightForgeResult,
    InterviewResult,
    NodeInfo,
    PanoramaResult,
    SearchResult,
    ZepToolsService,
)
from app.utils.locale import set_locale

HAN = re.compile(r'[\u4e00-\u9fff]')


@pytest.fixture(autouse=True)
def reset_locale():
    yield
    set_locale('zh')


def make_search():
    return SearchResult(facts=['Alice posted about the policy'], edges=[], nodes=[], query='policy', total_count=1)


def make_node():
    return NodeInfo(uuid='n1', name='Alice', labels=['Entity', 'Student'], summary='A student', attributes={})


def make_edge():
    return EdgeInfo(
        uuid='e1', name='SUPPORTS', fact='Alice supports the policy',
        source_node_uuid='n1', target_node_uuid='n2',
        source_node_name='Alice', target_node_name='Bob',
        expired_at='2026-01-01',
    )


def make_insight():
    return InsightForgeResult(
        query='How will students react?',
        simulation_requirement='Campus policy change',
        sub_queries=['Who reacts first?'],
        semantic_facts=['Alice supports the policy'],
        entity_insights=[{'name': 'Alice', 'type': 'Student', 'summary': 'A student', 'related_facts': ['f1', 'f2']}],
        relationship_chains=['Alice --[SUPPORTS]--> Bob'],
        total_facts=1, total_entities=1, total_relationships=1,
    )


def make_panorama():
    return PanoramaResult(
        query='policy',
        all_nodes=[make_node()],
        active_facts=['Alice supports the policy'],
        historical_facts=['[2025 - 2026] Bob opposed the policy'],
        total_nodes=1, total_edges=2, active_count=1, historical_count=1,
    )


def make_interview(response):
    agent = AgentInterview(
        agent_name='Alice', agent_role='Student', agent_bio='Sophomore',
        question='1. What do you think?', response=response,
        key_quotes=['This policy will change how we study on campus every day'],
    )
    return InterviewResult(
        interview_topic='Policy views', interview_questions=['What do you think?'],
        interviews=[agent], selection_reasoning='', summary='',
        total_agents=3, interviewed_count=1,
    )


EN_LABELS = {
    'search': ['Search query: policy', 'Found 1 relevant results', '\n### Related facts:\n'],
    'insight': [
        '## In-Depth Future Prediction Analysis', 'Analysis question: How will students react?',
        'Prediction scenario: Campus policy change', '\n### Prediction Data Statistics',
        '- Relevant prediction facts: 1', '- Entities involved: 1', '- Relationship chains: 1',
        '\n### Analyzed Sub-questions\n', '\n### [Key Facts] (quote these verbatim in the report)\n',
        '\n### [Core Entities]\n', '- **Alice** (Student)', '  Summary: "A student"', '  Related facts: 2',
        '\n### [Relationship Chains]\n',
    ],
    'panorama': [
        '## Breadth Search Results (Future Panoramic View)', 'Query: policy', '\n### Statistics',
        '- Total nodes: 1', '- Total edges: 2', '- Currently valid facts: 1', '- Historical/expired facts: 1',
        '\n### [Currently Valid Facts] (original simulation output)\n',
        '\n### [Historical/Expired Facts] (record of how events evolved)\n',
        '\n### [Entities Involved]\n', '- **Alice** (Student)',
    ],
    'interview': [
        '## In-Depth Interview Report', '**Interview topic:** Policy views',
        '**Interviewees:** 1 / 3 simulated agents', '\n### Interviewee Selection Rationale\n(Automatically selected)',
        '\n### Interview Transcripts', '\n#### Interview #1: Alice', '_Bio: Sophomore_\n',
        '\n**Key Quotes:**\n', '\n### Interview Summary and Key Viewpoints\n(No summary)',
    ],
}

ZH_LABELS = {
    'search': ['搜索查询: policy', '找到 1 条相关信息', '\n### 相关事实:\n'],
    'insight': [
        '## 未来预测深度分析', '分析问题: ', '预测场景: ', '- 相关预测事实: 1条', '- 涉及实体: 1个',
        '- 关系链: 1条', '\n### 分析的子问题\n', '\n### 【关键事实】(请在报告中引用这些原文)\n',
        '\n### 【核心实体】\n', '  摘要: "A student"', '  相关事实: 2条', '\n### 【关系链】\n',
    ],
    'panorama': [
        '## 广度搜索结果（未来全景视图）', '查询: policy', '- 总节点数: 1', '- 总边数: 2',
        '- 当前有效事实: 1条', '- 历史/过期事实: 1条', '\n### 【当前有效事实】(模拟结果原文)\n',
        '\n### 【历史/过期事实】(演变过程记录)\n', '\n### 【涉及实体】\n',
    ],
    'interview': [
        '## 深度采访报告', '**采访主题:** Policy views', '**采访人数:** 1 / 3 位模拟Agent',
        '\n### 采访对象选择理由\n（自动选择）', '\n### 采访实录', '\n#### 采访 #1: Alice', '_简介: Sophomore_\n',
        '\n**关键引言:**\n', '\n### 采访摘要与核心观点\n（无摘要）',
    ],
}


def render_all():
    return {
        'search': make_search().to_text(),
        'insight': make_insight().to_text(),
        'panorama': make_panorama().to_text(),
        'interview': make_interview('answer').to_text(),
    }


def test_to_text_english_has_no_chinese_and_frontend_labels():
    set_locale('en')
    texts = render_all()
    for kind, text in texts.items():
        assert not HAN.search(text), (kind, HAN.findall(text))
        for label in EN_LABELS[kind]:
            assert label in text, (kind, label)

    assert make_node().to_text() == 'Entity: Alice (Type: Student)\nSummary: A student'
    edge_text = make_edge().to_text(include_temporal=True)
    assert edge_text == (
        'Relation: Alice --[SUPPORTS]--> Bob\nFact: Alice supports the policy\n'
        'Validity: Unknown - Present (Expired: 2026-01-01)'
    )


def test_to_text_chinese_labels_unchanged():
    set_locale('zh')
    texts = render_all()
    for kind, text in texts.items():
        for label in ZH_LABELS[kind]:
            assert label in text, (kind, label)

    assert make_node().to_text() == '实体: Alice (类型: Student)\n摘要: A student'
    assert make_edge().to_text(include_temporal=True) == (
        '关系: Alice --[SUPPORTS]--> Bob\n事实: Alice supports the policy\n'
        '时效: 未知 - 至今 (已过期: 2026-01-01)'
    )


def test_key_quotes_strip_english_question_labels():
    set_locale('en')
    response = (
        'Question 1: I think the new campus policy will make studying much harder for everyone. '
        'It is short.\n\nQuestion 2: My classmates are already organizing a petition against it.'
    )
    quotes = ZepToolsService._extract_key_quotes(response, '')
    assert quotes
    assert all(not q.lower().startswith('question') for q in quotes)
    assert 'I think the new campus policy will make studying much harder for everyone.' in quotes


def test_key_quotes_strip_chinese_question_labels():
    set_locale('zh')
    response = '问题1：我认为这项新政策会让校园里每一位同学的学习变得更加困难和紧张。\n\n问题2：我的同学们已经开始组织联名请愿活动来反对它了。'
    quotes = ZepToolsService._extract_key_quotes(response, '')
    assert quotes
    assert all(not q.startswith('问题') and q.endswith('。') for q in quotes)


class RecordingLLM:
    def __init__(self, json_response=None, text_response='summary'):
        self.calls = []
        self.json_response = json_response
        self.text_response = text_response

    def chat_json(self, messages, **kwargs):
        self.calls.append(messages)
        if self.json_response is None:
            raise RuntimeError('boom')
        return self.json_response

    def chat(self, messages, **kwargs):
        self.calls.append(messages)
        return self.text_response


def make_service(llm):
    service = ZepToolsService.__new__(ZepToolsService)
    service._llm_client = llm
    return service


def prompt_text(llm):
    return '\n'.join(m['content'] for call in llm.calls for m in call)


@pytest.mark.parametrize('locale', ['en', 'zh'])
def test_service_prompts_follow_locale(locale):
    set_locale(locale)
    llm = RecordingLLM(json_response={'sub_queries': ['a'], 'selected_indices': [0], 'reasoning': 'r', 'questions': ['q']})
    service = make_service(llm)
    profiles = [{'realname': 'Alice', 'bio': 'Student'}]

    service._generate_sub_queries('policy', 'Campus policy change', report_context='ctx')
    service._select_agents_for_interview(profiles, 'views', '', 1)
    service._generate_interview_questions('views', '', profiles)
    service._generate_interview_summary([make_interview('answer').interviews[0]], 'views')

    text = prompt_text(llm)
    if locale == 'en':
        assert not HAN.search(text), sorted(set(HAN.findall(text)))
        assert 'Report context: ctx' in text
        assert 'Not provided' in text
        assert '"profession": "Unknown"' in text
    else:
        assert '报告上下文：ctx' in text
        assert '未提供' in text


def test_service_fallbacks_english():
    set_locale('en')
    service = make_service(RecordingLLM(json_response=None))
    profiles = [{'realname': 'Alice'}]

    for item in service._generate_sub_queries('policy', 'req'):
        assert not HAN.search(item)
    _, _, reasoning = service._select_agents_for_interview(profiles, 'views', '', 1)
    assert reasoning == 'Used the default selection strategy'
    for q in service._generate_interview_questions('views', '', profiles):
        assert not HAN.search(q)
    assert service._generate_interview_summary([], 'views') == 'No interviews were completed'
