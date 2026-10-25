import ast
import re
import string
from collections import Counter
from pathlib import Path

import pytest

from app.api import simulation as simulation_api
from app.services import ontology_generator as ontology_module
from app.services.ontology_generator import OntologyGenerator
from app.services.simulation_config_generator import SimulationConfigGenerator
from app.services.zep_entity_reader import EntityNode
from app.utils.locale import set_locale

CJK = re.compile(r"[\u3400-\u9fff\uf900-\ufaff\u3000-\u303f\uff00-\uffef]")
APP_DIR = Path(__file__).resolve().parents[1] / "app"
LOCALIZED_FILES = [
    APP_DIR / "services" / "ontology_generator.py",
    APP_DIR / "services" / "simulation_config_generator.py",
    APP_DIR / "api" / "simulation.py",
]


@pytest.fixture(autouse=True)
def _reset_locale():
    yield
    set_locale("zh")


class _RecordingLLMClient:
    def __init__(self):
        self.calls = []

    def chat_json(self, **kwargs):
        self.calls.append(kwargs)
        return {"entity_types": [], "edge_types": [], "analysis_summary": "ok"}


def _entities():
    return [
        EntityNode(
            uuid=f"e{i}",
            name=f"Student {i}",
            labels=["Entity", "Student"],
            summary="A student who posts about the campus controversy",
            attributes={},
        )
        for i in range(25)
    ] + [
        EntityNode(
            uuid="u1",
            name="City University",
            labels=["Entity", "University"],
            summary="The university at the center of the event",
            attributes={},
        )
    ]


def _config_generator(captured):
    generator = object.__new__(SimulationConfigGenerator)

    def fake_call(prompt, system_prompt):
        captured.append((prompt, system_prompt))
        return {}

    generator._call_llm_with_retry = fake_call
    return generator


def _render_config_prompts():
    captured = []
    generator = _config_generator(captured)
    entities = _entities()
    context = generator._build_context("Simulate the debate", "D" * 60000, entities)
    generator._generate_time_config(context, len(entities))
    generator._generate_event_config(context, "Simulate the debate", entities)
    generator._generate_agent_configs_batch(context, entities[:2], 0, "Simulate the debate")
    return context, captured


def _render_ontology_prompts():
    llm = _RecordingLLMClient()
    generator = OntologyGenerator(llm_client=llm)
    generator.generate(
        document_texts=["A short source document."],
        simulation_requirement="Simulate the debate",
        additional_context="Focus on students",
    )
    messages = llm.calls[0]["messages"]

    chunked = OntologyGenerator(llm_client=object())
    chunked.MAX_TEXT_LENGTH_FOR_LLM = 2000
    chunked.LONG_TEXT_CHUNK_SIZE = 500
    chunked.LONG_TEXT_CHUNK_OVERLAP = 0
    chunked.MAX_LONG_TEXT_CHUNKS = 3
    chunked.MIN_LONG_TEXT_EXCERPT = 120
    long_text = "".join(f"CHUNK{i:02d}-" + (str(i % 10) * 490) for i in range(14))
    chunk_context = chunked._build_document_context([long_text])

    code = generator.generate_python_code({
        "entity_types": [{"name": "Student", "description": "A student", "attributes": []}],
        "edge_types": [{"name": "STUDIES_AT", "description": "Studies at", "attributes": [],
                        "source_targets": [{"source": "Student", "target": "University"}]}],
    })
    return {
        "system": messages[0]["content"],
        "user": messages[1]["content"],
        "chunk_context": chunk_context,
        "excerpt": OntologyGenerator._excerpt_text("x" * 500, 200),
        "code": code,
    }


@pytest.mark.parametrize("locale", ["en", "de"])
def test_non_chinese_locales_use_english_generator_prompts(locale):
    set_locale(locale)
    ontology = _render_ontology_prompts()
    context, config_calls = _render_config_prompts()

    texts = dict(ontology)
    texts["config_context"] = context
    for i, (prompt, system_prompt) in enumerate(config_calls):
        texts[f"config_prompt_{i}"] = prompt
        # The appended language instruction may legitimately be non-English.
        texts[f"config_system_{i}"] = system_prompt.split("\n\n")[0]
    texts["interview"] = simulation_api.optimize_interview_prompt("How do you feel?")

    for name, text in texts.items():
        assert text, name
        assert not CJK.search(text), f"{name}: {CJK.findall(text)[:20]}"

    assert ontology["system"].startswith(ontology_module.ONTOLOGY_SYSTEM_PROMPT_EN)
    assert "## Simulation Requirement\n\nSimulate the debate" in ontology["user"]
    assert "## Additional Notes\n\nFocus on students" in ontology["user"]
    assert "[Automatic chunked summary of long text]" in ontology["chunk_context"]
    assert ontology["chunk_context"].count("--- Document 1 / Chunk") == 3
    assert "(middle of this chunk omitted)" in ontology["excerpt"]
    assert "# ============== Entity Type Definitions ==============" in ontology["code"]
    compile(ontology["code"], "<generated>", "exec")

    assert "## Entity Information (26 entities)" in context
    assert "... 5 more" in context
    assert "...(document truncated)" in context
    assert "## Original Document Content" in context
    assert "(range: 1-23)" in config_calls[0][0]
    assert '"poster_type"' in config_calls[1][0]
    assert '"stance": "<supportive/opposing/neutral/observer>"' in config_calls[2][0]
    assert texts["interview"] == simulation_api.INTERVIEW_PROMPT_PREFIX_EN + "How do you feel?"
    assert simulation_api.optimize_interview_prompt(texts["interview"]) == texts["interview"]


def test_chinese_locale_generator_prompts_are_unchanged():
    set_locale("zh")
    ontology = _render_ontology_prompts()
    context, config_calls = _render_config_prompts()

    assert ontology["system"].startswith(ontology_module.ONTOLOGY_SYSTEM_PROMPT)
    assert ontology["user"].startswith("## 模拟需求\n\nSimulate the debate")
    assert "## 额外说明\n\nFocus on students" in ontology["user"]
    assert "(本分块中间内容省略)" in ontology["excerpt"]
    assert ontology["code"].startswith('"""\n自定义实体类型定义\n由MiroFish自动生成，用于社会舆论模拟\n"""')
    assert "## 实体信息 (26个)" in context
    assert "  ... 还有 5 个" in context
    assert "\n...(文档已截断)" in context
    assert config_calls[0][0].startswith("基于以下模拟需求，生成时间模拟配置。")
    assert config_calls[0][1].startswith("你是社交媒体模拟专家。返回纯JSON格式")
    assert config_calls[1][0].startswith("基于以下模拟需求，生成事件配置。")
    assert config_calls[1][1].startswith("你是舆论分析专家。返回纯JSON格式。")
    assert config_calls[2][0].startswith("基于以下信息，为每个实体生成社交媒体活动配置。")
    assert config_calls[2][1].startswith("你是社交媒体行为分析专家。返回纯JSON")
    assert simulation_api.optimize_interview_prompt("你好") == simulation_api.INTERVIEW_PROMPT_PREFIX + "你好"


def _placeholders(node):
    if isinstance(node, ast.JoinedStr):
        return "fstring", Counter(
            ast.unparse(v.value) for v in node.values if isinstance(v, ast.FormattedValue)
        )
    return "format", Counter(
        f for _, f, _, _ in string.Formatter().parse(node.value) if f is not None
    )


def _localized_pairs(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    constants = {
        n.targets[0].id: n.value
        for n in tree.body
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "localized":
            zh, en = (constants.get(a.id) if isinstance(a, ast.Name) else a for a in node.args)
            yield node.lineno, zh, en


@pytest.mark.parametrize("path", LOCALIZED_FILES, ids=lambda p: p.name)
def test_localized_pairs_have_matching_fields_and_english_text(path):
    pairs = list(_localized_pairs(path))
    assert pairs
    for lineno, zh, en in pairs:
        assert isinstance(zh, (ast.Constant, ast.JoinedStr)), lineno
        assert isinstance(en, (ast.Constant, ast.JoinedStr)), lineno
        assert _placeholders(zh) == _placeholders(en), f"{path.name}:{lineno}"
        en_text = "".join(
            v.value for v in ([en] if isinstance(en, ast.Constant) else en.values)
            if isinstance(v, ast.Constant)
        )
        assert not CJK.search(en_text), f"{path.name}:{lineno}: {CJK.findall(en_text)[:20]}"
