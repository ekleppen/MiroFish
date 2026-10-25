import re
from types import SimpleNamespace

import pytest

from app.services.oasis_profile_generator import OasisProfileGenerator
from app.services.zep_entity_reader import EntityNode
from app.utils.locale import set_locale

CJK = re.compile(r"[\u3400-\u9fff\uf900-\ufaff\u3000-\u303f\uff00-\uffef]")


class _FakeGraph:
    def search(self, *, scope, **_kwargs):
        if scope == "edges":
            return SimpleNamespace(edges=[SimpleNamespace(fact="Alice criticized the policy")])
        return SimpleNamespace(nodes=[SimpleNamespace(summary="A city council", name="Council")])


def _generator():
    generator = object.__new__(OasisProfileGenerator)
    generator.zep_client = SimpleNamespace(graph=_FakeGraph())
    generator.graph_id = "graph-1"
    return generator


def _entity():
    return EntityNode(
        uuid="e1",
        name="Alice",
        labels=["Entity", "Person"],
        summary="A local activist",
        attributes={"role": "activist"},
        related_edges=[
            {"fact": "Alice joined the protest", "edge_name": "JOINED", "direction": "outgoing"},
            {"fact": "", "edge_name": "KNOWS", "direction": "outgoing"},
            {"fact": "", "edge_name": "FOLLOWS", "direction": "incoming"},
        ],
        related_nodes=[
            {"name": "Bob", "labels": ["Entity", "Person"], "summary": "A journalist"},
            {"name": "Carol", "labels": ["Entity"], "summary": ""},
        ],
    )


def _build_all(generator):
    entity = _entity()
    context = generator._build_entity_context(entity)
    search = generator._search_zep_for_entity(entity)
    return {
        "context": context,
        "search_context": search["context"],
        "search_summaries": "\n".join(search["node_summaries"]),
        "system_individual": generator._get_system_prompt(True),
        "system_group": generator._get_system_prompt(False),
        "individual": generator._build_individual_persona_prompt(
            "Alice", "Person", "A local activist", {"role": "activist"}, context
        ),
        "individual_empty": generator._build_individual_persona_prompt(
            "Alice", "Person", "A local activist", {}, ""
        ),
        "group": generator._build_group_persona_prompt(
            "Council", "Organization", "A city council", {"seats": 9}, context
        ),
        "group_empty": generator._build_group_persona_prompt(
            "Council", "Organization", "A city council", {}, ""
        ),
    }


@pytest.fixture(autouse=True)
def _reset_locale():
    yield
    set_locale("zh")


def test_english_locale_profile_prompts_contain_no_chinese():
    set_locale("en")
    prompts = _build_all(_generator())

    for name, text in prompts.items():
        assert text, name
        assert not CJK.search(text), f"{name}: {CJK.findall(text)}"

    assert "### Entity Attributes" in prompts["context"]
    assert "(related entity) --[FOLLOWS]--> Alice" in prompts["context"]
    assert "Related entity: Council" in prompts["search_summaries"]
    assert prompts["search_context"].startswith("Facts:\n")
    assert "Entity attributes: None" in prompts["individual_empty"]
    assert "No additional context" in prompts["group_empty"]
    assert '"male" or "female"' in prompts["individual"]
    assert 'gender must be the string "other"' in prompts["group"]


def test_chinese_locale_profile_prompts_are_unchanged():
    set_locale("zh")
    prompts = _build_all(_generator())

    assert "### 实体属性\n- role: activist" in prompts["context"]
    assert "- Alice --[KNOWS]--> (相关实体)" in prompts["context"]
    assert "- (相关实体) --[FOLLOWS]--> Alice" in prompts["context"]
    assert "### 相关事实和关系" in prompts["context"]
    assert "### 关联实体信息" in prompts["context"]
    assert "### Zep检索到的事实信息\n- Alice criticized the policy" in prompts["context"]
    assert "### Zep检索到的相关节点" in prompts["context"]
    assert "相关实体: Council" in prompts["search_summaries"]
    assert prompts["search_context"].startswith("事实信息:\n")
    assert prompts["system_individual"].startswith(
        "你是社交媒体用户画像生成专家。生成详细、真实的人设用于舆论模拟,最大程度还原已有现实情况。"
    )
    assert prompts["individual"].startswith("为实体生成详细的社交媒体用户人设,最大程度还原已有现实情况。")
    assert "(gender字段必须用英文male/female)" in prompts["individual"]
    assert "实体属性: 无\n" in prompts["individual_empty"]
    assert "无额外上下文" in prompts["individual_empty"]
    assert prompts["group"].startswith("为机构/群体实体生成详细的社交媒体账号设定,最大程度还原已有现实情况。")
    assert prompts["group"].endswith("- 机构账号发言要符合其身份定位")
    assert '6. country: 国家（使用中文，如"中国"）' in prompts["group"]
