"""
Report Agent service
Implements ReACT-style simulation report generation with LangChain + Zep

Features:
1. Generate reports from the simulation requirement and Zep graph information
2. Plan the outline structure first, then generate section by section
3. Each section uses a multi-round ReACT think-and-reflect pattern
4. Support chatting with the user, autonomously calling retrieval tools during the conversation
"""

import os
import json
import time
import re
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..config import Config
from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger
from ..utils.locale import get_language_instruction, localized, t
from .zep_tools import (
    ZepToolsService, 
    SearchResult, 
    InsightForgeResult, 
    PanoramaResult,
    InterviewResult
)

logger = get_logger('mirofish.report_agent')


class ReportLogger:
    """
    Report Agent detailed logger
    
    Generates an agent_log.jsonl file in the report folder, recording each detailed step.
    Each line is a complete JSON object containing timestamp, action type, details, etc.
    """
    
    def __init__(self, report_id: str):
        """
        Initialize the logger
        
        Args:
            report_id: Report ID, used to determine the log file path
        """
        self.report_id = report_id
        self.log_file_path = os.path.join(
            Config.UPLOAD_FOLDER, 'reports', report_id, 'agent_log.jsonl'
        )
        self.start_time = datetime.now()
        self._ensure_log_file()
    
    def _ensure_log_file(self):
        """Ensure the log file's directory exists"""
        log_dir = os.path.dirname(self.log_file_path)
        os.makedirs(log_dir, exist_ok=True)
    
    def _get_elapsed_time(self) -> float:
        """Get elapsed time since start (seconds)"""
        return (datetime.now() - self.start_time).total_seconds()
    
    def log(
        self, 
        action: str, 
        stage: str,
        details: Dict[str, Any],
        section_title: str = None,
        section_index: int = None
    ):
        """
        Record a log entry
        
        Args:
            action: Action type, e.g. 'start', 'tool_call', 'llm_response', 'section_complete'
            stage: Current stage, e.g. 'planning', 'generating', 'completed'
            details: Details dict, not truncated
            section_title: Current section title (optional)
            section_index: Current section index (optional)
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(self._get_elapsed_time(), 2),
            "report_id": self.report_id,
            "action": action,
            "stage": stage,
            "section_title": section_title,
            "section_index": section_index,
            "details": details
        }
        
        # Append to the JSONL file
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    def log_start(self, simulation_id: str, graph_id: str, simulation_requirement: str):
        """Log report generation start"""
        self.log(
            action="report_start",
            stage="pending",
            details={
                "simulation_id": simulation_id,
                "graph_id": graph_id,
                "simulation_requirement": simulation_requirement,
                "message": t('report.taskStarted')
            }
        )
    
    def log_planning_start(self):
        """Log outline planning start"""
        self.log(
            action="planning_start",
            stage="planning",
            details={"message": t('report.planningStart')}
        )
    
    def log_planning_context(self, context: Dict[str, Any]):
        """Log context information fetched during planning"""
        self.log(
            action="planning_context",
            stage="planning",
            details={
                "message": t('report.fetchSimContext'),
                "context": context
            }
        )
    
    def log_planning_complete(self, outline_dict: Dict[str, Any]):
        """Log outline planning completion"""
        self.log(
            action="planning_complete",
            stage="planning",
            details={
                "message": t('report.planningComplete'),
                "outline": outline_dict
            }
        )
    
    def log_section_start(self, section_title: str, section_index: int):
        """Log section generation start"""
        self.log(
            action="section_start",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={"message": t('report.sectionStart', title=section_title)}
        )
    
    def log_react_thought(self, section_title: str, section_index: int, iteration: int, thought: str):
        """Log the ReACT thought process"""
        self.log(
            action="react_thought",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "thought": thought,
                "message": t('report.reactThought', iteration=iteration)
            }
        )
    
    def log_tool_call(
        self, 
        section_title: str, 
        section_index: int,
        tool_name: str, 
        parameters: Dict[str, Any],
        iteration: int
    ):
        """Log a tool call"""
        self.log(
            action="tool_call",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "tool_name": tool_name,
                "parameters": parameters,
                "message": t('report.toolCall', toolName=tool_name)
            }
        )
    
    def log_tool_result(
        self,
        section_title: str,
        section_index: int,
        tool_name: str,
        result: str,
        iteration: int
    ):
        """Log a tool call result (full content, not truncated)"""
        self.log(
            action="tool_result",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "tool_name": tool_name,
                "result": result,  # Full result, not truncated
                "result_length": len(result),
                "message": t('report.toolResult', toolName=tool_name)
            }
        )
    
    def log_llm_response(
        self,
        section_title: str,
        section_index: int,
        response: str,
        iteration: int,
        has_tool_calls: bool,
        has_final_answer: bool
    ):
        """Log an LLM response (full content, not truncated)"""
        self.log(
            action="llm_response",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "response": response,  # Full response, not truncated
                "response_length": len(response),
                "has_tool_calls": has_tool_calls,
                "has_final_answer": has_final_answer,
                "message": t('report.llmResponse', hasToolCalls=has_tool_calls, hasFinalAnswer=has_final_answer)
            }
        )
    
    def log_section_content(
        self,
        section_title: str,
        section_index: int,
        content: str,
        tool_calls_count: int
    ):
        """Log section content generation done (records content only; does not mean the whole section is complete)"""
        self.log(
            action="section_content",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "content": content,  # Full content, not truncated
                "content_length": len(content),
                "tool_calls_count": tool_calls_count,
                "message": t('report.sectionContentDone', title=section_title)
            }
        )
    
    def log_section_full_complete(
        self,
        section_title: str,
        section_index: int,
        full_content: str
    ):
        """
        Log section generation completion

        The frontend should listen for this log to determine whether a section is truly complete and to get the full content
        """
        self.log(
            action="section_complete",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "content": full_content,
                "content_length": len(full_content),
                "message": t('report.sectionComplete', title=section_title)
            }
        )
    
    def log_report_complete(self, total_sections: int, total_time_seconds: float):
        """Log report generation completion"""
        self.log(
            action="report_complete",
            stage="completed",
            details={
                "total_sections": total_sections,
                "total_time_seconds": round(total_time_seconds, 2),
                "message": t('report.reportComplete')
            }
        )
    
    def log_error(self, error_message: str, stage: str, section_title: str = None):
        """Log an error"""
        self.log(
            action="error",
            stage=stage,
            section_title=section_title,
            section_index=None,
            details={
                "error": error_message,
                "message": t('report.errorOccurred', error=error_message)
            }
        )


class ReportConsoleLogger:
    """
    Report Agent console logger
    
    Writes console-style logs (INFO, WARNING, etc.) to console_log.txt in the report folder.
    Unlike agent_log.jsonl, these logs are plain-text console output.
    """
    
    def __init__(self, report_id: str):
        """
        Initialize the console logger
        
        Args:
            report_id: Report ID, used to determine the log file path
        """
        self.report_id = report_id
        self.log_file_path = os.path.join(
            Config.UPLOAD_FOLDER, 'reports', report_id, 'console_log.txt'
        )
        self._ensure_log_file()
        self._file_handler = None
        self._setup_file_handler()
    
    def _ensure_log_file(self):
        """Ensure the log file's directory exists"""
        log_dir = os.path.dirname(self.log_file_path)
        os.makedirs(log_dir, exist_ok=True)
    
    def _setup_file_handler(self):
        """Set up a file handler so logs are also written to file"""
        import logging
        
        # Create file handler
        self._file_handler = logging.FileHandler(
            self.log_file_path,
            mode='a',
            encoding='utf-8'
        )
        self._file_handler.setLevel(logging.INFO)
        
        # Use the same concise format as the console
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        self._file_handler.setFormatter(formatter)
        
        # Attach to report_agent-related loggers
        loggers_to_attach = [
            'mirofish.report_agent',
            'mirofish.zep_tools',
        ]
        
        for logger_name in loggers_to_attach:
            target_logger = logging.getLogger(logger_name)
            # Avoid adding duplicates
            if self._file_handler not in target_logger.handlers:
                target_logger.addHandler(self._file_handler)
    
    def close(self):
        """Close the file handler and remove it from loggers"""
        import logging
        
        if self._file_handler:
            loggers_to_detach = [
                'mirofish.report_agent',
                'mirofish.zep_tools',
            ]
            
            for logger_name in loggers_to_detach:
                target_logger = logging.getLogger(logger_name)
                if self._file_handler in target_logger.handlers:
                    target_logger.removeHandler(self._file_handler)
            
            self._file_handler.close()
            self._file_handler = None
    
    def __del__(self):
        """Ensure the file handler is closed on destruction"""
        self.close()


class ReportStatus(str, Enum):
    """Report status"""
    PENDING = "pending"
    PLANNING = "planning"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ReportSection:
    """Report section"""
    title: str
    content: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content
        }

    def to_markdown(self, level: int = 2) -> str:
        """Convert to Markdown format"""
        md = f"{'#' * level} {self.title}\n\n"
        if self.content:
            md += f"{self.content}\n\n"
        return md


@dataclass
class ReportOutline:
    """Report outline"""
    title: str
    summary: str
    sections: List[ReportSection]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "summary": self.summary,
            "sections": [s.to_dict() for s in self.sections]
        }
    
    def to_markdown(self) -> str:
        """Convert to Markdown format"""
        md = f"# {self.title}\n\n"
        md += f"> {self.summary}\n\n"
        for section in self.sections:
            md += section.to_markdown()
        return md


@dataclass
class Report:
    """Complete report"""
    report_id: str
    simulation_id: str
    graph_id: str
    simulation_requirement: str
    status: ReportStatus
    outline: Optional[ReportOutline] = None
    markdown_content: str = ""
    created_at: str = ""
    completed_at: str = ""
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "simulation_id": self.simulation_id,
            "graph_id": self.graph_id,
            "simulation_requirement": self.simulation_requirement,
            "status": self.status.value,
            "outline": self.outline.to_dict() if self.outline else None,
            "markdown_content": self.markdown_content,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "error": self.error
        }


# ═══════════════════════════════════════════════════════════════
# Prompt template constants
# ═══════════════════════════════════════════════════════════════

# ── Tool descriptions ──

TOOL_DESC_INSIGHT_FORGE = """\
【深度洞察检索 - 强大的检索工具】
这是我们强大的检索函数，专为深度分析设计。它会：
1. 自动将你的问题分解为多个子问题
2. 从多个维度检索模拟图谱中的信息
3. 整合语义搜索、实体分析、关系链追踪的结果
4. 返回最全面、最深度的检索内容

【使用场景】
- 需要深入分析某个话题
- 需要了解事件的多个方面
- 需要获取支撑报告章节的丰富素材

【返回内容】
- 相关事实原文（可直接引用）
- 核心实体洞察
- 关系链分析"""

TOOL_DESC_INSIGHT_FORGE_EN = """\
[Deep Insight Retrieval - Powerful Retrieval Tool]
This is our powerful retrieval function, designed for in-depth analysis. It will:
1. Automatically break your question down into multiple sub-questions
2. Retrieve information from the simulation graph across multiple dimensions
3. Integrate the results of semantic search, entity analysis, and relationship-chain tracing
4. Return the most comprehensive, in-depth retrieval content

[Use Cases]
- You need to analyze a topic in depth
- You need to understand multiple aspects of an event
- You need rich material to support a report section

[Returns]
- Relevant facts in their original wording (can be quoted directly)
- Core entity insights
- Relationship-chain analysis"""

TOOL_DESC_PANORAMA_SEARCH = """\
【广度搜索 - 获取全貌视图】
这个工具用于获取模拟结果的完整全貌，特别适合了解事件演变过程。它会：
1. 获取所有相关节点和关系
2. 区分当前有效的事实和历史/过期的事实
3. 帮助你了解舆情是如何演变的

【使用场景】
- 需要了解事件的完整发展脉络
- 需要对比不同阶段的舆情变化
- 需要获取全面的实体和关系信息

【返回内容】
- 当前有效事实（模拟最新结果）
- 历史/过期事实（演变记录）
- 所有涉及的实体"""

TOOL_DESC_PANORAMA_SEARCH_EN = """\
[Breadth Search - Get the Full Picture]
This tool retrieves the complete, panoramic view of the simulation results and is especially suited to understanding how an event evolved. It will:
1. Retrieve all relevant nodes and relationships
2. Distinguish currently valid facts from historical/expired facts
3. Help you understand how public opinion evolved

[Use Cases]
- You need to understand the complete course of an event's development
- You need to compare changes in public opinion across different stages
- You need comprehensive entity and relationship information

[Returns]
- Currently valid facts (latest simulation results)
- Historical/expired facts (record of the evolution)
- All entities involved"""

TOOL_DESC_QUICK_SEARCH = """\
【简单搜索 - 快速检索】
轻量级的快速检索工具，适合简单、直接的信息查询。

【使用场景】
- 需要快速查找某个具体信息
- 需要验证某个事实
- 简单的信息检索

【返回内容】
- 与查询最相关的事实列表"""

TOOL_DESC_QUICK_SEARCH_EN = """\
[Simple Search - Quick Retrieval]
A lightweight, fast retrieval tool suited to simple, direct information queries.

[Use Cases]
- You need to quickly look up a specific piece of information
- You need to verify a fact
- Simple information retrieval

[Returns]
- A list of the facts most relevant to the query"""

TOOL_DESC_INTERVIEW_AGENTS = """\
【深度采访 - 真实Agent采访（双平台）】
调用OASIS模拟环境的采访API，对正在运行的模拟Agent进行真实采访！
这不是LLM模拟，而是调用真实的采访接口获取模拟Agent的原始回答。
默认在Twitter和Reddit两个平台同时采访，获取更全面的观点。

功能流程：
1. 自动读取人设文件，了解所有模拟Agent
2. 智能选择与采访主题最相关的Agent（如学生、媒体、官方等）
3. 自动生成采访问题
4. 调用 /api/simulation/interview/batch 接口在双平台进行真实采访
5. 整合所有采访结果，提供多视角分析

【使用场景】
- 需要从不同角色视角了解事件看法（学生怎么看？媒体怎么看？官方怎么说？）
- 需要收集多方意见和立场
- 需要获取模拟Agent的真实回答（来自OASIS模拟环境）
- 想让报告更生动，包含"采访实录"

【返回内容】
- 被采访Agent的身份信息
- 各Agent在Twitter和Reddit两个平台的采访回答
- 关键引言（可直接引用）
- 采访摘要和观点对比

【重要】需要OASIS模拟环境正在运行才能使用此功能！"""

TOOL_DESC_INTERVIEW_AGENTS_EN = """\
[In-Depth Interviews - Real Agent Interviews (Dual Platform)]
Calls the OASIS simulation environment's interview API to conduct real interviews with the running simulated agents!
This is not an LLM simulation; it calls the real interview interface to obtain the simulated agents' original answers.
By default, agents are interviewed on both Twitter and Reddit at the same time to obtain a more comprehensive range of views.

Workflow:
1. Automatically read the persona files to learn about all simulated agents
2. Intelligently select the agents most relevant to the interview topic (e.g., students, media, officials)
3. Automatically generate interview questions
4. Call the /api/simulation/interview/batch endpoint to conduct real interviews on both platforms
5. Integrate all interview results and provide multi-perspective analysis

[Use Cases]
- You need to understand views on the event from the perspectives of different roles (What do students think? What does the media think? What do officials say?)
- You need to collect opinions and positions from multiple parties
- You need real answers from simulated agents (from the OASIS simulation environment)
- You want to make the report more vivid by including "interview transcripts"

[Returns]
- Identity information of the interviewed agents
- Each agent's interview answers on both Twitter and Reddit
- Key quotes (can be quoted directly)
- Interview summary and comparison of viewpoints

[Important] The OASIS simulation environment must be running to use this feature!"""

# ── Outline planning prompt ──

PLAN_SYSTEM_PROMPT = """\
你是一个「未来预测报告」的撰写专家，拥有对模拟世界的「上帝视角」——你可以洞察模拟中每一位Agent的行为、言论和互动。

【核心理念】
我们构建了一个模拟世界，并向其中注入了特定的「模拟需求」作为变量。模拟世界的演化结果，就是对未来可能发生情况的预测。你正在观察的不是"实验数据"，而是"未来的预演"。

【你的任务】
撰写一份「未来预测报告」，回答：
1. 在我们设定的条件下，未来发生了什么？
2. 各类Agent（人群）是如何反应和行动？
3. 这个模拟揭示了哪些值得关注的未来趋势和风险？

【报告定位】
- ✅ 这是一份基于模拟的未来预测报告，揭示"如果这样，未来会怎样"
- ✅ 聚焦于预测结果：事件走向、群体反应、涌现现象、潜在风险
- ✅ 模拟世界中的Agent言行就是对未来人群行为的预测
- ❌ 不是对现实世界现状的分析
- ❌ 不是泛泛而谈的舆情综述

【章节数量限制】
- 最少2个章节，最多5个章节
- 不需要子章节，每个章节直接撰写完整内容
- 内容要精炼，聚焦于核心预测发现
- 章节结构由你根据预测结果自主设计

请输出JSON格式的报告大纲，格式如下：
{
    "title": "报告标题",
    "summary": "报告摘要（一句话概括核心预测发现）",
    "sections": [
        {
            "title": "章节标题",
            "description": "章节内容描述"
        }
    ]
}

注意：sections数组最少2个，最多5个元素！"""

PLAN_SYSTEM_PROMPT_EN = """\
You are an expert writer of "future prediction reports", with a "god's-eye view" of the simulated world: you can observe the behavior, statements, and interactions of every agent in the simulation.

[Core Concept]
We have built a simulated world and injected a specific "simulation requirement" into it as a variable. The way the simulated world evolves is a prediction of what may happen in the future. What you are observing is not "experimental data" but "a rehearsal of the future".

[Your Task]
Write a "future prediction report" that answers:
1. Under the conditions we set, what happened in the future?
2. How did the various types of agents (groups of people) react and act?
3. What future trends and risks worth attention does this simulation reveal?

[Report Positioning]
- ✅ This is a simulation-based future prediction report, revealing "if this happens, what will the future look like"
- ✅ Focus on the prediction results: how events unfold, group reactions, emergent phenomena, potential risks
- ✅ The words and actions of agents in the simulated world are predictions of future human behavior
- ❌ It is not an analysis of the current state of the real world
- ❌ It is not a generic overview of public opinion

[Section Count Limits]
- Minimum 2 sections, maximum 5 sections
- No subsections are needed; write the complete content directly for each section
- Content should be concise and focused on the core prediction findings
- You design the section structure yourself based on the prediction results

Please output the report outline in JSON format, as follows:
{
    "title": "Report title",
    "summary": "Report summary (one sentence summarizing the core prediction findings)",
    "sections": [
        {
            "title": "Section title",
            "description": "Description of the section content"
        }
    ]
}

Note: the sections array must have at least 2 and at most 5 elements!"""

PLAN_USER_PROMPT_TEMPLATE = """\
【预测场景设定】
我们向模拟世界注入的变量（模拟需求）：{simulation_requirement}

【模拟世界规模】
- 参与模拟的实体数量: {total_nodes}
- 实体间产生的关系数量: {total_edges}
- 实体类型分布: {entity_types}
- 活跃Agent数量: {total_entities}

【模拟预测到的部分未来事实样本】
{related_facts_json}

请以「上帝视角」审视这个未来预演：
1. 在我们设定的条件下，未来呈现出了什么样的状态？
2. 各类人群（Agent）是如何反应和行动的？
3. 这个模拟揭示了哪些值得关注的未来趋势？

根据预测结果，设计最合适的报告章节结构。

【再次提醒】报告章节数量：最少2个，最多5个，内容要精炼聚焦于核心预测发现。"""

PLAN_USER_PROMPT_TEMPLATE_EN = """\
[Prediction Scenario Setup]
The variable we injected into the simulated world (simulation requirement): {simulation_requirement}

[Simulated World Scale]
- Number of entities participating in the simulation: {total_nodes}
- Number of relationships generated between entities: {total_edges}
- Entity type distribution: {entity_types}
- Number of active agents: {total_entities}

[Sample of Future Facts Predicted by the Simulation]
{related_facts_json}

Examine this rehearsal of the future from a "god's-eye view":
1. Under the conditions we set, what state did the future take?
2. How did the various groups of people (agents) react and act?
3. What future trends worth attention does this simulation reveal?

Based on the prediction results, design the most appropriate report section structure.

[Reminder] Number of report sections: minimum 2, maximum 5; content should be concise and focused on the core prediction findings."""

# ── Section generation prompt ──

SECTION_SYSTEM_PROMPT_TEMPLATE = """\
你是一个「未来预测报告」的撰写专家，正在撰写报告的一个章节。

报告标题: {report_title}
报告摘要: {report_summary}
预测场景（模拟需求）: {simulation_requirement}

当前要撰写的章节: {section_title}

═══════════════════════════════════════════════════════════════
【核心理念】
═══════════════════════════════════════════════════════════════

模拟世界是对未来的预演。我们向模拟世界注入了特定条件（模拟需求），
模拟中Agent的行为和互动，就是对未来人群行为的预测。

你的任务是：
- 揭示在设定条件下，未来发生了什么
- 预测各类人群（Agent）是如何反应和行动的
- 发现值得关注的未来趋势、风险和机会

❌ 不要写成对现实世界现状的分析
✅ 要聚焦于"未来会怎样"——模拟结果就是预测的未来

═══════════════════════════════════════════════════════════════
【最重要的规则 - 必须遵守】
═══════════════════════════════════════════════════════════════

1. 【必须调用工具观察模拟世界】
   - 你正在以「上帝视角」观察未来的预演
   - 所有内容必须来自模拟世界中发生的事件和Agent言行
   - 禁止使用你自己的知识来编写报告内容
   - 每个章节至少调用3次工具（最多5次）来观察模拟的世界，它代表了未来

2. 【必须引用Agent的原始言行】
   - Agent的发言和行为是对未来人群行为的预测
   - 在报告中使用引用格式展示这些预测，例如：
     > "某类人群会表示：原文内容..."
   - 这些引用是模拟预测的核心证据

3. 【语言一致性 - 引用内容必须翻译为报告语言】
   - 工具返回的内容可能包含与报告语言不同的表述
   - 报告必须全部使用与用户指定语言一致的语言撰写
   - 当你引用工具返回的其他语言内容时，必须将其翻译为报告语言后再写入
   - 翻译时保持原意不变，确保表述自然通顺
   - 这一规则同时适用于正文和引用块（> 格式）中的内容

4. 【忠实呈现预测结果】
   - 报告内容必须反映模拟世界中的代表未来的模拟结果
   - 不要添加模拟中不存在的信息
   - 如果某方面信息不足，如实说明

5. 【禁止捏造数据】
   - ❌ 禁止捏造用户名、引用、统计数字或互动数据
   - ❌ 禁止在回复中包含 <tool_result> 块 — 只有系统会提供工具结果
   - ✅ 只能引用真实出现在工具结果中的实体、引用和数据
   - 如果工具结果中没有相关内容，应如实说明，而非编造

═══════════════════════════════════════════════════════════════
【⚠️ 格式规范 - 极其重要！】
═══════════════════════════════════════════════════════════════

【一个章节 = 最小内容单位】
- 每个章节是报告的最小分块单位
- ❌ 禁止在章节内使用任何 Markdown 标题（#、##、###、#### 等）
- ❌ 禁止在内容开头添加章节主标题
- ✅ 章节标题由系统自动添加，你只需撰写纯正文内容
- ✅ 使用**粗体**、段落分隔、引用、列表来组织内容，但不要用标题

【正确示例】
```
本章节分析了事件的舆论传播态势。通过对模拟数据的深入分析，我们发现...

**首发引爆阶段**

微博作为舆情的第一现场，承担了信息首发的核心功能：

> "微博贡献了68%的首发声量..."

**情绪放大阶段**

抖音平台进一步放大了事件影响力：

- 视觉冲击力强
- 情绪共鸣度高
```

【错误示例】
```
## 执行摘要          ← 错误！不要添加任何标题
### 一、首发阶段     ← 错误！不要用###分小节
#### 1.1 详细分析   ← 错误！不要用####细分

本章节分析了...
```

═══════════════════════════════════════════════════════════════
【可用检索工具】（每章节调用3-5次）
═══════════════════════════════════════════════════════════════

{tools_description}

【工具使用建议 - 请混合使用不同工具，不要只用一种】
- insight_forge: 深度洞察分析，自动分解问题并多维度检索事实和关系
- panorama_search: 广角全景搜索，了解事件全貌、时间线和演变过程
- quick_search: 快速验证某个具体信息点
- interview_agents: 采访模拟Agent，获取不同角色的第一人称观点和真实反应

═══════════════════════════════════════════════════════════════
【工作流程】
═══════════════════════════════════════════════════════════════

每次回复你只能做以下两件事之一（不可同时做）：

选项A - 调用工具：
输出你的思考，然后用以下格式调用一个工具：
<tool_call>
{{"name": "工具名称", "parameters": {{"参数名": "参数值"}}}}
</tool_call>
系统会执行工具并把结果返回给你。你不需要也不能自己编写工具返回结果。

选项B - 输出最终内容：
当你已通过工具获取了足够信息，以 "Final Answer:" 开头输出章节内容。

⚠️ 严格禁止：
- 禁止在一次回复中同时包含工具调用和 Final Answer
- 禁止自己编造工具返回结果（Observation），所有工具结果由系统注入
- 每次回复最多调用一个工具

═══════════════════════════════════════════════════════════════
【章节内容要求】
═══════════════════════════════════════════════════════════════

1. 内容必须基于工具检索到的模拟数据
2. 大量引用原文来展示模拟效果
3. 使用Markdown格式（但禁止使用标题）：
   - 使用 **粗体文字** 标记重点（代替子标题）
   - 使用列表（-或1.2.3.）组织要点
   - 使用空行分隔不同段落
   - ❌ 禁止使用 #、##、###、#### 等任何标题语法
4. 【引用格式规范 - 必须单独成段】
   引用必须独立成段，前后各有一个空行，不能混在段落中：

   ✅ 正确格式：
   ```
   校方的回应被认为缺乏实质内容。

   > "校方的应对模式在瞬息万变的社交媒体环境中显得僵化和迟缓。"

   这一评价反映了公众的普遍不满。
   ```

   ❌ 错误格式：
   ```
   校方的回应被认为缺乏实质内容。> "校方的应对模式..." 这一评价反映了...
   ```
5. 保持与其他章节的逻辑连贯性
6. 【避免重复】仔细阅读下方已完成的章节内容，不要重复描述相同的信息
7. 【再次强调】不要添加任何标题！用**粗体**代替小节标题"""

SECTION_SYSTEM_PROMPT_TEMPLATE_EN = """\
You are an expert writer of "future prediction reports", and you are writing one section of a report.

Report title: {report_title}
Report summary: {report_summary}
Prediction scenario (simulation requirement): {simulation_requirement}

Section to write now: {section_title}

═══════════════════════════════════════════════════════════════
[Core Concept]
═══════════════════════════════════════════════════════════════

The simulated world is a rehearsal of the future. We injected specific conditions (the simulation requirement) into the simulated world;
the behavior and interactions of the agents in the simulation are predictions of future human behavior.

Your task is to:
- Reveal what happened in the future under the given conditions
- Predict how the various groups of people (agents) reacted and acted
- Discover future trends, risks, and opportunities worth attention

❌ Do not write it as an analysis of the current state of the real world
✅ Focus on "what the future will look like" — the simulation results are the predicted future

═══════════════════════════════════════════════════════════════
[Most Important Rules - Must Be Followed]
═══════════════════════════════════════════════════════════════

1. [You Must Call Tools to Observe the Simulated World]
   - You are observing a rehearsal of the future from a "god's-eye view"
   - All content must come from events that occurred and agents' words and actions in the simulated world
   - Do not use your own knowledge to write the report content
   - Call tools at least 3 times (at most 5 times) per section to observe the simulated world, which represents the future

2. [You Must Quote the Agents' Original Words and Actions]
   - The agents' statements and behavior are predictions of future human behavior
   - Present these predictions in the report using quotation format, for example:
     > "A certain group would say: original text..."
   - These quotes are the core evidence of the simulation's predictions

3. [Language Consistency - Quoted Content Must Be Translated into the Report Language]
   - Content returned by tools may contain wording in a language different from the report language
   - The report must be written entirely in the language specified by the user
   - When you quote content in another language returned by a tool, you must translate it into the report language before writing it
   - Preserve the original meaning when translating, and make sure the wording reads naturally
   - This rule applies both to the body text and to content in quote blocks (> format)

4. [Faithfully Present the Prediction Results]
   - The report content must reflect the simulation results in the simulated world, which represent the future
   - Do not add information that does not exist in the simulation
   - If information on some aspect is insufficient, say so honestly

5. [Do Not Fabricate Data]
   - ❌ Do not fabricate usernames, quotes, statistics, or engagement data
   - ❌ Do not include <tool_result> blocks in your reply — only the system provides tool results
   - ✅ Only cite entities, quotes, and data that actually appear in the tool results
   - If the tool results contain nothing relevant, say so honestly instead of making something up

═══════════════════════════════════════════════════════════════
[⚠️ Formatting Rules - Extremely Important!]
═══════════════════════════════════════════════════════════════

[One Section = Smallest Content Unit]
- Each section is the smallest chunk of the report
- ❌ Do not use any Markdown headings (#, ##, ###, ####, etc.) within a section
- ❌ Do not add the section's main title at the beginning of the content
- ✅ Section titles are added automatically by the system; you only write the plain body text
- ✅ Use **bold**, paragraph breaks, quotes, and lists to organize the content, but do not use headings

[Correct Example]
```
This section analyzes how public opinion about the event spread. Through in-depth analysis of the simulation data, we found...

**Initial Outbreak Stage**

As the front line of public opinion, Weibo served the core function of breaking the news first:

> "Weibo contributed 68% of the initial volume..."

**Emotional Amplification Stage**

The Douyin platform further amplified the event's impact:

- Strong visual impact
- High emotional resonance
```

[Incorrect Example]
```
## Executive Summary          ← Wrong! Do not add any headings
### 1. Initial Stage          ← Wrong! Do not use ### for subsections
#### 1.1 Detailed Analysis    ← Wrong! Do not use #### for further subdivision

This section analyzes...
```

═══════════════════════════════════════════════════════════════
[Available Retrieval Tools] (call 3-5 times per section)
═══════════════════════════════════════════════════════════════

{tools_description}

[Tool Usage Suggestions - Mix different tools; do not use only one]
- insight_forge: Deep insight analysis; automatically breaks the question down and retrieves facts and relationships across multiple dimensions
- panorama_search: Wide-angle panoramic search; understand the full picture, timeline, and evolution of the event
- quick_search: Quickly verify a specific piece of information
- interview_agents: Interview simulated agents to obtain first-person views and genuine reactions from different roles

═══════════════════════════════════════════════════════════════
[Workflow]
═══════════════════════════════════════════════════════════════

In each reply you may do only one of the following two things (never both):

Option A - Call a tool:
Output your thinking, then call one tool using the following format:
<tool_call>
{{"name": "tool_name", "parameters": {{"parameter_name": "parameter_value"}}}}
</tool_call>
The system will execute the tool and return the result to you. You do not need to, and must not, write tool results yourself.

Option B - Output the final content:
Once you have obtained enough information through tools, output the section content starting with "Final Answer:".

⚠️ Strictly prohibited:
- Do not include both a tool call and a Final Answer in a single reply
- Do not fabricate tool results (Observation) yourself; all tool results are injected by the system
- Call at most one tool per reply

═══════════════════════════════════════════════════════════════
[Section Content Requirements]
═══════════════════════════════════════════════════════════════

1. The content must be based on the simulation data retrieved by the tools
2. Quote the original text extensively to show the simulation results
3. Use Markdown formatting (but headings are prohibited):
   - Use **bold text** to mark key points (instead of subheadings)
   - Use lists (- or 1. 2. 3.) to organize points
   - Use blank lines to separate paragraphs
   - ❌ Do not use any heading syntax such as #, ##, ###, ####
4. [Quote Formatting Rules - Quotes Must Be Standalone Paragraphs]
   Quotes must stand as separate paragraphs, with a blank line before and after; they must not be mixed into a paragraph:

   ✅ Correct format:
   ```
   The school's response was seen as lacking substance.

   > "The school's response pattern appeared rigid and sluggish in the fast-changing social media environment."

   This assessment reflects widespread public dissatisfaction.
   ```

   ❌ Incorrect format:
   ```
   The school's response was seen as lacking substance. > "The school's response pattern..." This assessment reflects...
   ```
5. Maintain logical coherence with the other sections
6. [Avoid Repetition] Carefully read the completed section content below and do not repeat the same information
7. [Emphasis Again] Do not add any headings! Use **bold** instead of subsection headings"""

SECTION_USER_PROMPT_TEMPLATE = """\
已完成的章节内容（请仔细阅读，避免重复）：
{previous_content}

═══════════════════════════════════════════════════════════════
【当前任务】撰写章节: {section_title}
═══════════════════════════════════════════════════════════════

【重要提醒】
1. 仔细阅读上方已完成的章节，避免重复相同的内容！
2. 开始前必须先调用工具获取模拟数据
3. 请混合使用不同工具，不要只用一种
4. 报告内容必须来自检索结果，不要使用自己的知识

【⚠️ 格式警告 - 必须遵守】
- ❌ 不要写任何标题（#、##、###、####都不行）
- ❌ 不要写"{section_title}"作为开头
- ✅ 章节标题由系统自动添加
- ✅ 直接写正文，用**粗体**代替小节标题

请开始：
1. 首先思考（Thought）这个章节需要什么信息
2. 然后调用工具（Action）获取模拟数据
3. 收集足够信息后输出 Final Answer（纯正文，无任何标题）"""

SECTION_USER_PROMPT_TEMPLATE_EN = """\
Completed section content (read carefully to avoid repetition):
{previous_content}

═══════════════════════════════════════════════════════════════
[Current Task] Write section: {section_title}
═══════════════════════════════════════════════════════════════

[Important Reminders]
1. Carefully read the completed sections above and avoid repeating the same content!
2. You must call tools to obtain simulation data before you start
3. Mix different tools; do not use only one
4. The report content must come from the retrieval results; do not use your own knowledge

[⚠️ Formatting Warning - Must Be Followed]
- ❌ Do not write any headings (#, ##, ###, #### are all prohibited)
- ❌ Do not start with "{section_title}"
- ✅ Section titles are added automatically by the system
- ✅ Write the body text directly, using **bold** instead of subsection headings

Please begin:
1. First think (Thought) about what information this section needs
2. Then call tools (Action) to obtain simulation data
3. Once you have gathered enough information, output the Final Answer (plain body text, no headings)"""

# ── Message templates used inside the ReACT loop ──

REACT_OBSERVATION_TEMPLATE = """\
Observation（检索结果）:

═══ 工具 {tool_name} 返回 ═══
{result}

═══════════════════════════════════════════════════════════════
已调用工具 {tool_calls_count}/{max_tool_calls} 次（已用: {used_tools_str}）{unused_hint}
- 如果信息充分：以 "Final Answer:" 开头输出章节内容（必须引用上述原文）
- 如果需要更多信息：调用一个工具继续检索
═══════════════════════════════════════════════════════════════"""

REACT_OBSERVATION_TEMPLATE_EN = """\
Observation (retrieval results):

═══ Tool {tool_name} returned ═══
{result}

═══════════════════════════════════════════════════════════════
Tools called {tool_calls_count}/{max_tool_calls} times (used: {used_tools_str}){unused_hint}
- If the information is sufficient: output the section content starting with "Final Answer:" (you must quote the original text above)
- If you need more information: call one tool to continue retrieving
═══════════════════════════════════════════════════════════════"""

REACT_INSUFFICIENT_TOOLS_MSG = (
    "【注意】你只调用了{tool_calls_count}次工具，至少需要{min_tool_calls}次。"
    "请再调用工具获取更多模拟数据，然后再输出 Final Answer。{unused_hint}"
)

REACT_INSUFFICIENT_TOOLS_MSG_EN = (
    "[Note] You have called tools only {tool_calls_count} time(s); at least {min_tool_calls} are required. "
    "Please call tools again to obtain more simulation data, and only then output the Final Answer. {unused_hint}"
)

REACT_INSUFFICIENT_TOOLS_MSG_ALT = (
    "当前只调用了 {tool_calls_count} 次工具，至少需要 {min_tool_calls} 次。"
    "请调用工具获取模拟数据。{unused_hint}"
)

REACT_INSUFFICIENT_TOOLS_MSG_ALT_EN = (
    "Tools have been called only {tool_calls_count} time(s) so far; at least {min_tool_calls} are required. "
    "Please call tools to obtain simulation data. {unused_hint}"
)

REACT_TOOL_LIMIT_MSG = (
    "工具调用次数已达上限（{tool_calls_count}/{max_tool_calls}），不能再调用工具。"
    '请立即基于已获取的信息，以 "Final Answer:" 开头输出章节内容。'
)

REACT_TOOL_LIMIT_MSG_EN = (
    "The tool call limit has been reached ({tool_calls_count}/{max_tool_calls}); no more tools can be called. "
    'Based on the information already obtained, immediately output the section content starting with "Final Answer:".'
)

REACT_UNUSED_TOOLS_HINT = "\n💡 你还没有使用过: {unused_list}，建议尝试不同工具获取多角度信息"

REACT_UNUSED_TOOLS_HINT_EN = "\n💡 You have not used yet: {unused_list}. Try different tools to get information from multiple angles"

REACT_FORCE_FINAL_MSG = "已达到工具调用限制，请直接输出 Final Answer: 并生成章节内容。"

REACT_FORCE_FINAL_MSG_EN = "The tool call limit has been reached. Output Final Answer: directly and generate the section content."

# ── Chat prompt ──

CHAT_SYSTEM_PROMPT_TEMPLATE = """\
你是一个简洁高效的模拟预测助手。

【背景】
预测条件: {simulation_requirement}

【已生成的分析报告】
{report_content}

【规则】
1. 优先基于上述报告内容回答问题
2. 直接回答问题，避免冗长的思考论述
3. 仅在报告内容不足以回答时，才调用工具检索更多数据
4. 回答要简洁、清晰、有条理

【可用工具】（仅在需要时使用，最多调用1-2次）
{tools_description}

【工具调用格式】
<tool_call>
{{"name": "工具名称", "parameters": {{"参数名": "参数值"}}}}
</tool_call>

【回答风格】
- 简洁直接，不要长篇大论
- 使用 > 格式引用关键内容
- 优先给出结论，再解释原因"""

CHAT_SYSTEM_PROMPT_TEMPLATE_EN = """\
You are a concise and efficient simulation prediction assistant.

[Background]
Prediction conditions: {simulation_requirement}

[Generated Analysis Report]
{report_content}

[Rules]
1. Prioritize answering questions based on the report content above
2. Answer questions directly; avoid lengthy reasoning
3. Only call tools to retrieve more data when the report content is insufficient to answer
4. Answers should be concise, clear, and well organized

[Available Tools] (use only when needed, at most 1-2 calls)
{tools_description}

[Tool Call Format]
<tool_call>
{{"name": "tool_name", "parameters": {{"parameter_name": "parameter_value"}}}}
</tool_call>

[Answer Style]
- Concise and direct; avoid long-winded answers
- Use > format to quote key content
- Give the conclusion first, then explain the reasons"""

CHAT_OBSERVATION_SUFFIX = "\n\n请简洁回答问题。"

CHAT_OBSERVATION_SUFFIX_EN = "\n\nPlease answer the question concisely."


# ═══════════════════════════════════════════════════════════════
# ReportAgent main class
# ═══════════════════════════════════════════════════════════════


class ReportAgent:
    """
    Report Agent - simulation report generation agent

    Uses the ReACT (Reasoning + Acting) pattern:
    1. Planning phase: analyze the simulation requirement and plan the report outline
    2. Generation phase: generate content section by section; each section may call tools multiple times to gather information
    3. Reflection phase: check content completeness and accuracy
    """
    
    # Max tool calls (per section)
    MAX_TOOL_CALLS_PER_SECTION = 5
    
    # Max reflection rounds
    MAX_REFLECTION_ROUNDS = 3
    
    # Max tool calls in a chat
    MAX_TOOL_CALLS_PER_CHAT = 2
    
    def __init__(
        self, 
        graph_id: str,
        simulation_id: str,
        simulation_requirement: str,
        llm_client: Optional[LLMClient] = None,
        zep_tools: Optional[ZepToolsService] = None
    ):
        """
        Initialize the Report Agent
        
        Args:
            graph_id: Graph ID
            simulation_id: Simulation ID
            simulation_requirement: Simulation requirement description
            llm_client: LLM client (optional)
            zep_tools: Zep tools service (optional)
        """
        self.graph_id = graph_id
        self.simulation_id = simulation_id
        self.simulation_requirement = simulation_requirement
        
        self.llm = llm_client or LLMClient()
        self.zep_tools = zep_tools or ZepToolsService()
        
        # Tool definitions
        self.tools = self._define_tools()
        
        # Logger (initialized in generate_report)
        self.report_logger: Optional[ReportLogger] = None
        # Console logger (initialized in generate_report)
        self.console_logger: Optional[ReportConsoleLogger] = None
        
        logger.info(t('report.agentInitDone', graphId=graph_id, simulationId=simulation_id))
    
    def _define_tools(self) -> Dict[str, Dict[str, Any]]:
        """Define available tools"""
        return {
            "insight_forge": {
                "name": "insight_forge",
                "description": localized(TOOL_DESC_INSIGHT_FORGE, TOOL_DESC_INSIGHT_FORGE_EN),
                "parameters": {
                    "query": localized("你想深入分析的问题或话题", "The question or topic you want to analyze in depth"),
                    "report_context": localized(
                        "当前报告章节的上下文（可选，有助于生成更精准的子问题）",
                        "Context of the current report section (optional; helps generate more precise sub-questions)",
                    )
                }
            },
            "panorama_search": {
                "name": "panorama_search",
                "description": localized(TOOL_DESC_PANORAMA_SEARCH, TOOL_DESC_PANORAMA_SEARCH_EN),
                "parameters": {
                    "query": localized("搜索查询，用于相关性排序", "Search query, used for relevance ranking"),
                    "include_expired": localized(
                        "是否包含过期/历史内容（默认True）",
                        "Whether to include expired/historical content (default True)",
                    )
                }
            },
            "quick_search": {
                "name": "quick_search",
                "description": localized(TOOL_DESC_QUICK_SEARCH, TOOL_DESC_QUICK_SEARCH_EN),
                "parameters": {
                    "query": localized("搜索查询字符串", "Search query string"),
                    "limit": localized("返回结果数量（可选，默认10）", "Number of results to return (optional, default 10)")
                }
            },
            "interview_agents": {
                "name": "interview_agents",
                "description": localized(TOOL_DESC_INTERVIEW_AGENTS, TOOL_DESC_INTERVIEW_AGENTS_EN),
                "parameters": {
                    "interview_topic": localized(
                        "采访主题或需求描述（如：'了解学生对宿舍甲醛事件的看法'）",
                        "Interview topic or description of the need (e.g., 'understand students' views on the dormitory formaldehyde incident')",
                    ),
                    "max_agents": localized(
                        "最多采访的Agent数量（可选，默认5，最大10）",
                        "Maximum number of agents to interview (optional, default 5, max 10)",
                    )
                }
            }
        }
    
    def _execute_tool(self, tool_name: str, parameters: Dict[str, Any], report_context: str = "") -> str:
        """
        Execute a tool call
        
        Args:
            tool_name: Tool name
            parameters: Tool parameters
            report_context: Report context (used by InsightForge)
            
        Returns:
            Tool execution result (text format)
        """
        logger.info(t('report.executingTool', toolName=tool_name, params=parameters))
        
        try:
            if tool_name == "insight_forge":
                query = parameters.get("query", "")
                ctx = parameters.get("report_context", "") or report_context
                result = self.zep_tools.insight_forge(
                    graph_id=self.graph_id,
                    query=query,
                    simulation_requirement=self.simulation_requirement,
                    report_context=ctx
                )
                return result.to_text()
            
            elif tool_name == "panorama_search":
                # Breadth search - get the full picture
                query = parameters.get("query", "")
                include_expired = parameters.get("include_expired", True)
                if isinstance(include_expired, str):
                    include_expired = include_expired.lower() in ['true', '1', 'yes']
                result = self.zep_tools.panorama_search(
                    graph_id=self.graph_id,
                    query=query,
                    include_expired=include_expired
                )
                return result.to_text()
            
            elif tool_name == "quick_search":
                # Simple search - quick retrieval
                query = parameters.get("query", "")
                limit = parameters.get("limit", 10)
                if isinstance(limit, str):
                    limit = int(limit)
                result = self.zep_tools.quick_search(
                    graph_id=self.graph_id,
                    query=query,
                    limit=limit
                )
                return result.to_text()
            
            elif tool_name == "interview_agents":
                # In-depth interview - call the real OASIS interview API to get simulated agents' answers (both platforms)
                interview_topic = parameters.get("interview_topic", parameters.get("query", ""))
                max_agents = parameters.get("max_agents", 5)
                if isinstance(max_agents, str):
                    max_agents = int(max_agents)
                max_agents = min(max_agents, 10)
                result = self.zep_tools.interview_agents(
                    simulation_id=self.simulation_id,
                    interview_requirement=interview_topic,
                    simulation_requirement=self.simulation_requirement,
                    max_agents=max_agents
                )
                return result.to_text()
            
            # ========== Backward-compatible legacy tools (internally redirected to new tools) ==========
            
            elif tool_name == "search_graph":
                # Redirect to quick_search
                logger.info(t('report.redirectToQuickSearch'))
                return self._execute_tool("quick_search", parameters, report_context)
            
            elif tool_name == "get_graph_statistics":
                result = self.zep_tools.get_graph_statistics(self.graph_id)
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            elif tool_name == "get_entity_summary":
                entity_name = parameters.get("entity_name", "")
                result = self.zep_tools.get_entity_summary(
                    graph_id=self.graph_id,
                    entity_name=entity_name
                )
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            elif tool_name == "get_simulation_context":
                # Redirect to insight_forge, since it is more powerful
                logger.info(t('report.redirectToInsightForge'))
                query = parameters.get("query", self.simulation_requirement)
                return self._execute_tool("insight_forge", {"query": query}, report_context)
            
            elif tool_name == "get_entities_by_type":
                entity_type = parameters.get("entity_type", "")
                nodes = self.zep_tools.get_entities_by_type(
                    graph_id=self.graph_id,
                    entity_type=entity_type
                )
                result = [n.to_dict() for n in nodes]
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            else:
                return localized(
                    f"未知工具: {tool_name}。请使用以下工具之一: insight_forge, panorama_search, quick_search",
                    f"Unknown tool: {tool_name}. Please use one of the following tools: insight_forge, panorama_search, quick_search",
                )
                
        except Exception as e:
            logger.error(t('report.toolExecFailed', toolName=tool_name, error=str(e)))
            return localized(f"工具执行失败: {str(e)}", f"Tool execution failed: {str(e)}")
    
    # Set of valid tool names, used for validation in the bare-JSON fallback parsing
    VALID_TOOL_NAMES = {"insight_forge", "panorama_search", "quick_search", "interview_agents"}

    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse tool calls from the LLM response

        Supported formats (by priority):
        1. <tool_call>{"name": "tool_name", "parameters": {...}}</tool_call>
        2. Bare JSON (the whole response or a single line is a tool call JSON)
        """
        tool_calls = []

        # Format 1: XML style (standard format)
        xml_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        for match in re.finditer(xml_pattern, response, re.DOTALL):
            try:
                call_data = json.loads(match.group(1))
                tool_calls.append(call_data)
            except json.JSONDecodeError:
                pass

        if tool_calls:
            return tool_calls

        # Format 2: fallback - LLM output bare JSON directly (not wrapped in <tool_call> tags)
        # Only tried when format 1 didn't match, to avoid falsely matching JSON in body text
        stripped = response.strip()
        if stripped.startswith('{') and stripped.endswith('}'):
            try:
                call_data = json.loads(stripped)
                if self._is_valid_tool_call(call_data):
                    tool_calls.append(call_data)
                    return tool_calls
            except json.JSONDecodeError:
                pass

        # The response may contain reasoning text + bare JSON; try to extract the last JSON object
        json_pattern = r'(\{"(?:name|tool)"\s*:.*?\})\s*$'
        match = re.search(json_pattern, stripped, re.DOTALL)
        if match:
            try:
                call_data = json.loads(match.group(1))
                if self._is_valid_tool_call(call_data):
                    tool_calls.append(call_data)
            except json.JSONDecodeError:
                pass

        return tool_calls

    def _is_valid_tool_call(self, data: dict) -> bool:
        """Validate whether the parsed JSON is a valid tool call"""
        # Supports both {"name": ..., "parameters": ...} and {"tool": ..., "params": ...} key styles
        tool_name = data.get("name") or data.get("tool")
        if tool_name and tool_name in self.VALID_TOOL_NAMES:
            # Normalize keys to name / parameters
            if "tool" in data:
                data["name"] = data.pop("tool")
            if "params" in data and "parameters" not in data:
                data["parameters"] = data.pop("params")
            return True
        return False
    
    def _get_tools_description(self) -> str:
        """Generate tool description text"""
        desc_parts = [localized("可用工具：", "Available tools:")]
        for name, tool in self.tools.items():
            params_desc = ", ".join([f"{k}: {v}" for k, v in tool["parameters"].items()])
            desc_parts.append(f"- {name}: {tool['description']}")
            if params_desc:
                desc_parts.append(localized(f"  参数: {params_desc}", f"  Parameters: {params_desc}"))
        return "\n".join(desc_parts)

    @staticmethod
    def _strip_fake_tool_results(response: str) -> str:
        """Strip any <tool_result> blocks the LLM fabricated in its response.

        When the LLM generates a <tool_call> block and then continues to generate
        a <tool_result> block in the same response, we must strip the fake result
        before appending to message history. The real tool result will be injected
        separately by the system.
        """
        tag_pattern = re.compile(r'</?tool_result\b[^>]*>', flags=re.IGNORECASE)
        parts = []
        cursor = 0
        depth = 0

        for match in tag_pattern.finditer(response):
            if depth == 0:
                parts.append(response[cursor:match.start()])

            if match.group(0).lstrip().startswith('</'):
                depth = max(0, depth - 1)
            else:
                depth += 1
            cursor = match.end()

        if depth == 0:
            parts.append(response[cursor:])

        cleaned = ''.join(parts)
        # Treat a malformed opening tag without a closing `>` as unsafe too.
        cleaned = re.sub(r'<tool_result\b.*$', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        return cleaned.strip()

    def plan_outline(
        self, 
        progress_callback: Optional[Callable] = None
    ) -> ReportOutline:
        """
        Plan the report outline
        
        Use the LLM to analyze the simulation requirement and plan the report structure
        
        Args:
            progress_callback: Progress callback function
            
        Returns:
            ReportOutline: Report outline
        """
        logger.info(t('report.startPlanningOutline'))
        
        if progress_callback:
            progress_callback("planning", 0, t('progress.analyzingRequirements'))
        
        # First fetch the simulation context
        context = self.zep_tools.get_simulation_context(
            graph_id=self.graph_id,
            simulation_requirement=self.simulation_requirement
        )
        
        if progress_callback:
            progress_callback("planning", 30, t('progress.generatingOutline'))
        
        system_prompt = f"{localized(PLAN_SYSTEM_PROMPT, PLAN_SYSTEM_PROMPT_EN)}\n\n{get_language_instruction()}"
        user_prompt = localized(PLAN_USER_PROMPT_TEMPLATE, PLAN_USER_PROMPT_TEMPLATE_EN).format(
            simulation_requirement=self.simulation_requirement,
            total_nodes=context.get('graph_statistics', {}).get('total_nodes', 0),
            total_edges=context.get('graph_statistics', {}).get('total_edges', 0),
            entity_types=list(context.get('graph_statistics', {}).get('entity_types', {}).keys()),
            total_entities=context.get('total_entities', 0),
            related_facts_json=json.dumps(context.get('related_facts', [])[:10], ensure_ascii=False, indent=2),
        )

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            
            if progress_callback:
                progress_callback("planning", 80, t('progress.parsingOutline'))
            
            # Parse outline
            sections = []
            for section_data in response.get("sections", []):
                sections.append(ReportSection(
                    title=section_data.get("title", ""),
                    content=""
                ))
            
            outline = ReportOutline(
                title=response.get("title", t('backend.reportAgent.defaultReportTitle')),
                summary=response.get("summary", ""),
                sections=sections
            )
            
            if progress_callback:
                progress_callback("planning", 100, t('progress.outlinePlanComplete'))
            
            logger.info(t('report.outlinePlanDone', count=len(sections)))
            return outline
            
        except Exception as e:
            logger.error(t('report.outlinePlanFailed', error=str(e)))
            # Return a default outline (3 sections, as a fallback)
            return ReportOutline(
                title=t('backend.reportAgent.fallbackOutlineTitle'),
                summary=t('backend.reportAgent.fallbackOutlineSummary'),
                sections=[
                    ReportSection(title=t('backend.reportAgent.fallbackSectionScenario')),
                    ReportSection(title=t('backend.reportAgent.fallbackSectionBehavior')),
                    ReportSection(title=t('backend.reportAgent.fallbackSectionOutlook'))
                ]
            )
    
    def _generate_section_react(
        self, 
        section: ReportSection,
        outline: ReportOutline,
        previous_sections: List[str],
        progress_callback: Optional[Callable] = None,
        section_index: int = 0
    ) -> str:
        """
        Generate a single section's content using the ReACT pattern
        
        ReACT loop:
        1. Thought - analyze what information is needed
        2. Action - call tools to gather information
        3. Observation - analyze tool results
        4. Repeat until information is sufficient or the max count is reached
        5. Final Answer - generate the section content
        
        Args:
            section: Section to generate
            outline: Full outline
            previous_sections: Content of previous sections (for coherence)
            progress_callback: Progress callback
            section_index: Section index (for logging)
            
        Returns:
            Section content (Markdown format)
        """
        logger.info(t('report.reactGenerateSection', title=section.title))
        
        # Log section start
        if self.report_logger:
            self.report_logger.log_section_start(section.title, section_index)
        
        system_prompt = localized(SECTION_SYSTEM_PROMPT_TEMPLATE, SECTION_SYSTEM_PROMPT_TEMPLATE_EN).format(
            report_title=outline.title,
            report_summary=outline.summary,
            simulation_requirement=self.simulation_requirement,
            section_title=section.title,
            tools_description=self._get_tools_description(),
        )
        system_prompt = f"{system_prompt}\n\n{get_language_instruction()}"

        # Build user prompt - pass up to 4000 chars from each completed section
        if previous_sections:
            previous_parts = []
            for sec in previous_sections:
                # Max 4000 chars per section
                truncated = sec[:4000] + "..." if len(sec) > 4000 else sec
                previous_parts.append(truncated)
            previous_content = "\n\n---\n\n".join(previous_parts)
        else:
            previous_content = localized("（这是第一个章节）", "(This is the first section)")
        
        user_prompt = localized(SECTION_USER_PROMPT_TEMPLATE, SECTION_USER_PROMPT_TEMPLATE_EN).format(
            previous_content=previous_content,
            section_title=section.title,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # ReACT loop
        tool_calls_count = 0
        max_iterations = 5  # Max iteration rounds
        min_tool_calls = 3  # Min tool calls
        conflict_retries = 0  # Consecutive conflicts where a tool call and Final Answer appear together
        used_tools = set()  # Names of tools already called
        all_tools = {"insight_forge", "panorama_search", "quick_search", "interview_agents"}

        # Report context, used for InsightForge sub-question generation
        report_context = localized(
            f"章节标题: {section.title}\n模拟需求: {self.simulation_requirement}",
            f"Section title: {section.title}\nSimulation requirement: {self.simulation_requirement}",
        )
        
        for iteration in range(max_iterations):
            if progress_callback:
                progress_callback(
                    "generating", 
                    int((iteration / max_iterations) * 100),
                    t('progress.deepSearchAndWrite', current=tool_calls_count, max=self.MAX_TOOL_CALLS_PER_SECTION)
                )
            
            # Call LLM
            response = self.llm.chat(
                messages=messages,
                temperature=0.5,
                max_tokens=4096
            )

            # Check whether the LLM returned None (API error or empty content)
            if response is None:
                logger.warning(t('report.sectionIterNone', title=section.title, iteration=iteration + 1))
                # If iterations remain, add messages and retry
                if iteration < max_iterations - 1:
                    messages.append({"role": "assistant", "content": localized("（响应为空）", "(Empty response)")})
                    messages.append({"role": "user", "content": localized("请继续生成内容。", "Please continue generating the content.")})
                    continue
                # The last iteration also returned None; break out and force a wrap-up
                break

            logger.debug(f"LLM response: {response[:200]}...")

            # Parse once and reuse the result
            tool_calls = self._parse_tool_calls(response)
            has_tool_calls = bool(tool_calls)
            has_final_answer = "Final Answer:" in response

            # ── Conflict handling: LLM output both a tool call and a Final Answer ──
            if has_tool_calls and has_final_answer:
                conflict_retries += 1
                logger.warning(
                    t('report.sectionConflict', title=section.title, iteration=iteration+1, conflictCount=conflict_retries)
                )

                if conflict_retries <= 2:
                    # First two times: discard this response and ask the LLM to reply again
                    cleaned_response = ReportAgent._strip_fake_tool_results(response)
                    messages.append({"role": "assistant", "content": cleaned_response})
                    messages.append({
                        "role": "user",
                        "content": localized(
                            (
                                "【格式错误】你在一次回复中同时包含了工具调用和 Final Answer，这是不允许的。\n"
                                "每次回复只能做以下两件事之一：\n"
                                "- 调用一个工具（输出一个 <tool_call> 块，不要写 Final Answer）\n"
                                "- 输出最终内容（以 'Final Answer:' 开头，不要包含 <tool_call>）\n"
                                "请重新回复，只做其中一件事。"
                            ),
                            (
                                "[Format Error] Your reply contained both a tool call and a Final Answer, which is not allowed.\n"
                                "Each reply may do only one of the following two things:\n"
                                "- Call one tool (output one <tool_call> block; do not write a Final Answer)\n"
                                "- Output the final content (starting with 'Final Answer:'; do not include <tool_call>)\n"
                                "Please reply again, doing only one of these."
                            ),
                        ),
                    })
                    continue
                else:
                    # Third time: degrade - truncate to the first tool call and force execution
                    logger.warning(
                        t('report.sectionConflictDowngrade', title=section.title, conflictCount=conflict_retries)
                    )
                    first_tool_end = response.find('</tool_call>')
                    if first_tool_end != -1:
                        response = response[:first_tool_end + len('</tool_call>')]
                        tool_calls = self._parse_tool_calls(response)
                        has_tool_calls = bool(tool_calls)
                    has_final_answer = False
                    conflict_retries = 0

            # Log the LLM response
            if self.report_logger:
                self.report_logger.log_llm_response(
                    section_title=section.title,
                    section_index=section_index,
                    response=response,
                    iteration=iteration + 1,
                    has_tool_calls=has_tool_calls,
                    has_final_answer=has_final_answer
                )

            # ── Case 1: LLM output a Final Answer ──
            if has_final_answer:
                cleaned_response = ReportAgent._strip_fake_tool_results(response)
                # Not enough tool calls; reject and ask to keep calling tools
                if tool_calls_count < min_tool_calls:
                    messages.append({"role": "assistant", "content": cleaned_response})
                    unused_tools = all_tools - used_tools
                    unused_hint = localized(
                        f"（这些工具还未使用，推荐用一下他们: {', '.join(unused_tools)}）",
                        f" (These tools have not been used yet; consider using them: {', '.join(unused_tools)})",
                    ) if unused_tools else ""
                    messages.append({
                        "role": "user",
                        "content": localized(REACT_INSUFFICIENT_TOOLS_MSG, REACT_INSUFFICIENT_TOOLS_MSG_EN).format(
                            tool_calls_count=tool_calls_count,
                            min_tool_calls=min_tool_calls,
                            unused_hint=unused_hint,
                        ),
                    })
                    continue

                # Normal finish
                final_answer = cleaned_response.split("Final Answer:")[-1].strip()
                logger.info(t('report.sectionGenDone', title=section.title, count=tool_calls_count))

                if self.report_logger:
                    self.report_logger.log_section_content(
                        section_title=section.title,
                        section_index=section_index,
                        content=final_answer,
                        tool_calls_count=tool_calls_count
                    )
                return final_answer

            # ── Case 2: LLM tries to call a tool ──
            if has_tool_calls:
                # Tool quota exhausted → tell it explicitly and require a Final Answer
                if tool_calls_count >= self.MAX_TOOL_CALLS_PER_SECTION:
                    cleaned_response = ReportAgent._strip_fake_tool_results(response)
                    messages.append({"role": "assistant", "content": cleaned_response})
                    messages.append({
                        "role": "user",
                        "content": localized(REACT_TOOL_LIMIT_MSG, REACT_TOOL_LIMIT_MSG_EN).format(
                            tool_calls_count=tool_calls_count,
                            max_tool_calls=self.MAX_TOOL_CALLS_PER_SECTION,
                        ),
                    })
                    continue

                # Only execute the first tool call
                call = tool_calls[0]
                if len(tool_calls) > 1:
                    logger.info(t('report.multiToolOnlyFirst', total=len(tool_calls), toolName=call['name']))

                if self.report_logger:
                    self.report_logger.log_tool_call(
                        section_title=section.title,
                        section_index=section_index,
                        tool_name=call["name"],
                        parameters=call.get("parameters", {}),
                        iteration=iteration + 1
                    )

                result = self._execute_tool(
                    call["name"],
                    call.get("parameters", {}),
                    report_context=report_context
                )

                if self.report_logger:
                    self.report_logger.log_tool_result(
                        section_title=section.title,
                        section_index=section_index,
                        tool_name=call["name"],
                        result=result,
                        iteration=iteration + 1
                    )

                tool_calls_count += 1
                used_tools.add(call['name'])

                # Build hint for unused tools
                unused_tools = all_tools - used_tools
                unused_hint = ""
                if unused_tools and tool_calls_count < self.MAX_TOOL_CALLS_PER_SECTION:
                    unused_hint = localized(REACT_UNUSED_TOOLS_HINT, REACT_UNUSED_TOOLS_HINT_EN).format(
                        unused_list=localized("、", ", ").join(unused_tools)
                    )

                cleaned_response = ReportAgent._strip_fake_tool_results(response)
                messages.append({"role": "assistant", "content": cleaned_response})
                messages.append({
                    "role": "user",
                    "content": localized(REACT_OBSERVATION_TEMPLATE, REACT_OBSERVATION_TEMPLATE_EN).format(
                        tool_name=call["name"],
                        result=result,
                        tool_calls_count=tool_calls_count,
                        max_tool_calls=self.MAX_TOOL_CALLS_PER_SECTION,
                        used_tools_str=", ".join(used_tools),
                        unused_hint=unused_hint,
                    ),
                })
                continue

            # ── Case 3: neither a tool call nor a Final Answer ──
            cleaned_response = ReportAgent._strip_fake_tool_results(response)
            messages.append({"role": "assistant", "content": cleaned_response})

            if tool_calls_count < min_tool_calls:
                # Not enough tool calls; recommend unused tools
                unused_tools = all_tools - used_tools
                unused_hint = localized(
                    f"（这些工具还未使用，推荐用一下他们: {', '.join(unused_tools)}）",
                    f" (These tools have not been used yet; consider using them: {', '.join(unused_tools)})",
                ) if unused_tools else ""

                messages.append({
                    "role": "user",
                    "content": localized(REACT_INSUFFICIENT_TOOLS_MSG_ALT, REACT_INSUFFICIENT_TOOLS_MSG_ALT_EN).format(
                        tool_calls_count=tool_calls_count,
                        min_tool_calls=min_tool_calls,
                        unused_hint=unused_hint,
                    ),
                })
                continue

            # Enough tool calls; the LLM output content without the "Final Answer:" prefix
            # Use this content directly as the final answer instead of idling
            logger.info(t('report.sectionNoPrefix', title=section.title, count=tool_calls_count))
            final_answer = cleaned_response

            if self.report_logger:
                self.report_logger.log_section_content(
                    section_title=section.title,
                    section_index=section_index,
                    content=final_answer,
                    tool_calls_count=tool_calls_count
                )
            return final_answer
        
        # Max iterations reached; force content generation
        logger.warning(t('report.sectionMaxIter', title=section.title))
        messages.append({"role": "user", "content": localized(REACT_FORCE_FINAL_MSG, REACT_FORCE_FINAL_MSG_EN)})
        
        response = self.llm.chat(
            messages=messages,
            temperature=0.5,
            max_tokens=4096
        )

        # Check whether the LLM returned None during the forced wrap-up
        if response is None:
            logger.error(t('report.sectionForceFailed', title=section.title))
            final_answer = t('report.sectionGenFailedContent')
        elif "Final Answer:" in response:
            final_answer = response.split("Final Answer:")[-1].strip()
        else:
            final_answer = response
        
        # Log section content generation done
        if self.report_logger:
            self.report_logger.log_section_content(
                section_title=section.title,
                section_index=section_index,
                content=final_answer,
                tool_calls_count=tool_calls_count
            )
        
        return final_answer
    
    def generate_report(
        self, 
        progress_callback: Optional[Callable[[str, int, str], None]] = None,
        report_id: Optional[str] = None
    ) -> Report:
        """
        Generate the full report (section-by-section real-time output)
        
        Each section is saved to the folder as soon as it is generated, without waiting for the whole report.
        File structure:
        reports/{report_id}/
            meta.json       - Report metadata
            outline.json    - Report outline
            progress.json   - Generation progress
            section_01.md   - Section 1
            section_02.md   - Section 2
            ...
            full_report.md  - Full report
        
        Args:
            progress_callback: Progress callback function (stage, progress, message)
            report_id: Report ID (optional; auto-generated if not provided)
            
        Returns:
            Report: Full report
        """
        import uuid
        
        # Auto-generate report_id if not provided
        if not report_id:
            report_id = f"report_{uuid.uuid4().hex[:12]}"
        start_time = datetime.now()
        
        report = Report(
            report_id=report_id,
            simulation_id=self.simulation_id,
            graph_id=self.graph_id,
            simulation_requirement=self.simulation_requirement,
            status=ReportStatus.PENDING,
            created_at=datetime.now().isoformat()
        )
        
        # List of completed section titles (for progress tracking)
        completed_section_titles = []
        
        try:
            # Initialize: create the report folder and save the initial state
            ReportManager._ensure_report_folder(report_id)
            
            # Initialize the logger (structured log agent_log.jsonl)
            self.report_logger = ReportLogger(report_id)
            self.report_logger.log_start(
                simulation_id=self.simulation_id,
                graph_id=self.graph_id,
                simulation_requirement=self.simulation_requirement
            )
            
            # Initialize the console logger (console_log.txt)
            self.console_logger = ReportConsoleLogger(report_id)
            
            ReportManager.update_progress(
                report_id, "pending", 0, t('progress.initReport'),
                completed_sections=[]
            )
            ReportManager.save_report(report)
            
            # Phase 1: plan the outline
            report.status = ReportStatus.PLANNING
            ReportManager.update_progress(
                report_id, "planning", 5, t('progress.startPlanningOutline'),
                completed_sections=[]
            )
            
            # Log planning start
            self.report_logger.log_planning_start()
            
            if progress_callback:
                progress_callback("planning", 0, t('progress.startPlanningOutline'))
            
            outline = self.plan_outline(
                progress_callback=lambda stage, prog, msg: 
                    progress_callback(stage, prog // 5, msg) if progress_callback else None
            )
            report.outline = outline
            
            # Log planning completion
            self.report_logger.log_planning_complete(outline.to_dict())
            
            # Save outline to file
            ReportManager.save_outline(report_id, outline)
            ReportManager.update_progress(
                report_id, "planning", 15, t('progress.outlineDone', count=len(outline.sections)),
                completed_sections=[]
            )
            ReportManager.save_report(report)
            
            logger.info(t('report.outlineSavedToFile', reportId=report_id))
            
            # Phase 2: generate section by section (saving each section)
            report.status = ReportStatus.GENERATING
            
            total_sections = len(outline.sections)
            generated_sections = []  # Keep content for context
            
            for i, section in enumerate(outline.sections):
                section_num = i + 1
                base_progress = 20 + int((i / total_sections) * 70)
                
                # Update progress
                ReportManager.update_progress(
                    report_id, "generating", base_progress,
                    t('progress.generatingSection', title=section.title, current=section_num, total=total_sections),
                    current_section=section.title,
                    completed_sections=completed_section_titles
                )

                if progress_callback:
                    progress_callback(
                        "generating",
                        base_progress,
                        t('progress.generatingSection', title=section.title, current=section_num, total=total_sections)
                    )
                
                # Generate main section content
                section_content = self._generate_section_react(
                    section=section,
                    outline=outline,
                    previous_sections=generated_sections,
                    progress_callback=lambda stage, prog, msg:
                        progress_callback(
                            stage, 
                            base_progress + int(prog * 0.7 / total_sections),
                            msg
                        ) if progress_callback else None,
                    section_index=section_num
                )
                
                section.content = section_content
                generated_sections.append(f"## {section.title}\n\n{section_content}")

                # Save section
                ReportManager.save_section(report_id, section_num, section)
                completed_section_titles.append(section.title)

                # Log section completion
                full_section_content = f"## {section.title}\n\n{section_content}"

                if self.report_logger:
                    self.report_logger.log_section_full_complete(
                        section_title=section.title,
                        section_index=section_num,
                        full_content=full_section_content.strip()
                    )

                logger.info(t('report.sectionSaved', reportId=report_id, sectionNum=f"{section_num:02d}"))
                
                # Update progress
                ReportManager.update_progress(
                    report_id, "generating", 
                    base_progress + int(70 / total_sections),
                    t('progress.sectionDone', title=section.title),
                    current_section=None,
                    completed_sections=completed_section_titles
                )
            
            # Phase 3: assemble the full report
            if progress_callback:
                progress_callback("generating", 95, t('progress.assemblingReport'))
            
            ReportManager.update_progress(
                report_id, "generating", 95, t('progress.assemblingReport'),
                completed_sections=completed_section_titles
            )
            
            # Assemble the full report with ReportManager
            report.markdown_content = ReportManager.assemble_full_report(report_id, outline)
            report.status = ReportStatus.COMPLETED
            report.completed_at = datetime.now().isoformat()
            
            # Compute total elapsed time
            total_time_seconds = (datetime.now() - start_time).total_seconds()
            
            # Log report completion
            if self.report_logger:
                self.report_logger.log_report_complete(
                    total_sections=total_sections,
                    total_time_seconds=total_time_seconds
                )
            
            # Save final report
            ReportManager.save_report(report)
            ReportManager.update_progress(
                report_id, "completed", 100, t('progress.reportComplete'),
                completed_sections=completed_section_titles
            )
            
            if progress_callback:
                progress_callback("completed", 100, t('progress.reportComplete'))
            
            logger.info(t('report.reportGenDone', reportId=report_id))
            
            # Close the console logger
            if self.console_logger:
                self.console_logger.close()
                self.console_logger = None
            
            return report
            
        except Exception as e:
            logger.error(t('report.reportGenFailed', error=str(e)))
            report.status = ReportStatus.FAILED
            report.error = str(e)
            
            # Log the error
            if self.report_logger:
                self.report_logger.log_error(str(e), "failed")
            
            # Save failed state
            try:
                ReportManager.save_report(report)
                ReportManager.update_progress(
                    report_id, "failed", -1, t('progress.reportFailed', error=str(e)),
                    completed_sections=completed_section_titles
                )
            except Exception:
                pass  # Ignore errors from failed saves
            
            # Close the console logger
            if self.console_logger:
                self.console_logger.close()
                self.console_logger = None
            
            return report
    
    def chat(
        self, 
        message: str,
        chat_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Chat with the Report Agent
        
        During the chat the agent can autonomously call retrieval tools to answer questions
        
        Args:
            message: User message
            chat_history: Chat history
            
        Returns:
            {
                "response": "Agent reply",
                "tool_calls": [list of tools called],
                "sources": [information sources]
            }
        """
        logger.info(t('report.agentChat', message=message[:50]))
        
        chat_history = chat_history or []
        
        # Get already-generated report content
        report_content = ""
        try:
            report = ReportManager.get_report_by_simulation(self.simulation_id)
            if report and report.markdown_content:
                # Limit report length to avoid an overly long context
                report_content = report.markdown_content[:15000]
                if len(report.markdown_content) > 15000:
                    report_content += localized("\n\n... [报告内容已截断] ...", "\n\n... [Report content truncated] ...")
        except Exception as e:
            logger.warning(t('report.fetchReportFailed', error=e))
        
        system_prompt = localized(CHAT_SYSTEM_PROMPT_TEMPLATE, CHAT_SYSTEM_PROMPT_TEMPLATE_EN).format(
            simulation_requirement=self.simulation_requirement,
            report_content=report_content if report_content else localized("（暂无报告）", "(No report yet)"),
            tools_description=self._get_tools_description(),
        )
        system_prompt = f"{system_prompt}\n\n{get_language_instruction()}"

        # Build messages
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add chat history
        for h in chat_history[-10:]:  # Limit history length
            messages.append(h)
        
        # Add user message
        messages.append({
            "role": "user", 
            "content": message
        })
        
        # ReACT loop (simplified)
        tool_calls_made = []
        max_iterations = 2  # Fewer iteration rounds
        
        for iteration in range(max_iterations):
            response = self.llm.chat(
                messages=messages,
                temperature=0.5
            )
            
            # Parse tool calls
            tool_calls = self._parse_tool_calls(response)
            
            if not tool_calls:
                # No tool calls; return the response directly
                clean_response = re.sub(r'<tool_call>.*?</tool_call>', '', response, flags=re.DOTALL)
                clean_response = re.sub(r'\[TOOL_CALL\].*?\)', '', clean_response)
                clean_response = ReportAgent._strip_fake_tool_results(clean_response)
                
                return {
                    "response": clean_response.strip(),
                    "tool_calls": tool_calls_made,
                    "sources": [tc.get("parameters", {}).get("query", "") for tc in tool_calls_made]
                }
            
            # Execute tool calls (limited count)
            tool_results = []
            for call in tool_calls[:1]:  # At most 1 tool call per round
                if len(tool_calls_made) >= self.MAX_TOOL_CALLS_PER_CHAT:
                    break
                result = self._execute_tool(call["name"], call.get("parameters", {}))
                tool_results.append({
                    "tool": call["name"],
                    "result": result[:1500]  # Limit result length
                })
                tool_calls_made.append(call)
            
            # Add results to messages
            cleaned_response = ReportAgent._strip_fake_tool_results(response)
            messages.append({"role": "assistant", "content": cleaned_response})
            observation = "\n".join([
                localized(f"[{r['tool']}结果]\n{r['result']}", f"[{r['tool']} result]\n{r['result']}")
                for r in tool_results
            ])
            messages.append({
                "role": "user",
                "content": observation + localized(CHAT_OBSERVATION_SUFFIX, CHAT_OBSERVATION_SUFFIX_EN)
            })
        
        # Max iterations reached; get the final response
        final_response = self.llm.chat(
            messages=messages,
            temperature=0.5
        )
        
        # Clean up response
        clean_response = re.sub(r'<tool_call>.*?</tool_call>', '', final_response, flags=re.DOTALL)
        clean_response = re.sub(r'\[TOOL_CALL\].*?\)', '', clean_response)
        clean_response = ReportAgent._strip_fake_tool_results(clean_response)
        
        return {
            "response": clean_response.strip(),
            "tool_calls": tool_calls_made,
            "sources": [tc.get("parameters", {}).get("query", "") for tc in tool_calls_made]
        }


class ReportManager:
    """
    Report manager
    
    Responsible for persistent storage and retrieval of reports
    
    File structure (section-by-section output):
    reports/
      {report_id}/
        meta.json          - Report metadata and status
        outline.json       - Report outline
        progress.json      - Generation progress
        section_01.md      - Section 1
        section_02.md      - Section 2
        ...
        full_report.md     - Full report
    """
    
    # Report storage directory
    REPORTS_DIR = os.path.join(Config.UPLOAD_FOLDER, 'reports')
    
    @classmethod
    def _ensure_reports_dir(cls):
        """Ensure the reports root directory exists"""
        os.makedirs(cls.REPORTS_DIR, exist_ok=True)
    
    @classmethod
    def _get_report_folder(cls, report_id: str) -> str:
        """Get the report folder path"""
        return os.path.join(cls.REPORTS_DIR, report_id)
    
    @classmethod
    def _ensure_report_folder(cls, report_id: str) -> str:
        """Ensure the report folder exists and return its path"""
        folder = cls._get_report_folder(report_id)
        os.makedirs(folder, exist_ok=True)
        return folder
    
    @classmethod
    def _get_report_path(cls, report_id: str) -> str:
        """Get the report metadata file path"""
        return os.path.join(cls._get_report_folder(report_id), "meta.json")
    
    @classmethod
    def _get_report_markdown_path(cls, report_id: str) -> str:
        """Get the full report Markdown file path"""
        return os.path.join(cls._get_report_folder(report_id), "full_report.md")
    
    @classmethod
    def _get_outline_path(cls, report_id: str) -> str:
        """Get the outline file path"""
        return os.path.join(cls._get_report_folder(report_id), "outline.json")
    
    @classmethod
    def _get_progress_path(cls, report_id: str) -> str:
        """Get the progress file path"""
        return os.path.join(cls._get_report_folder(report_id), "progress.json")
    
    @classmethod
    def _get_section_path(cls, report_id: str, section_index: int) -> str:
        """Get the section Markdown file path"""
        return os.path.join(cls._get_report_folder(report_id), f"section_{section_index:02d}.md")
    
    @classmethod
    def _get_agent_log_path(cls, report_id: str) -> str:
        """Get the Agent log file path"""
        return os.path.join(cls._get_report_folder(report_id), "agent_log.jsonl")
    
    @classmethod
    def _get_console_log_path(cls, report_id: str) -> str:
        """Get the console log file path"""
        return os.path.join(cls._get_report_folder(report_id), "console_log.txt")
    
    @classmethod
    def get_console_log(cls, report_id: str, from_line: int = 0) -> Dict[str, Any]:
        """
        Get console log content
        
        These are the console output logs (INFO, WARNING, etc.) from report generation,
        distinct from the structured logs in agent_log.jsonl.
        
        Args:
            report_id: Report ID
            from_line: Line to start reading from (for incremental fetching; 0 means from the beginning)
            
        Returns:
            {
                "logs": [list of log lines],
                "total_lines": total line count,
                "from_line": starting line number,
                "has_more": whether there are more logs
            }
        """
        log_path = cls._get_console_log_path(report_id)
        
        if not os.path.exists(log_path):
            return {
                "logs": [],
                "total_lines": 0,
                "from_line": 0,
                "has_more": False
            }
        
        logs = []
        total_lines = 0
        
        with open(log_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                total_lines = i + 1
                if i >= from_line:
                    # Keep the original log line, stripping the trailing newline
                    logs.append(line.rstrip('\n\r'))
        
        return {
            "logs": logs,
            "total_lines": total_lines,
            "from_line": from_line,
            "has_more": False  # Read to the end
        }
    
    @classmethod
    def get_console_log_stream(cls, report_id: str) -> List[str]:
        """
        Get the complete console log (fetch everything at once)
        
        Args:
            report_id: Report ID
            
        Returns:
            List of log lines
        """
        result = cls.get_console_log(report_id, from_line=0)
        return result["logs"]
    
    @classmethod
    def get_agent_log(cls, report_id: str, from_line: int = 0) -> Dict[str, Any]:
        """
        Get Agent log content
        
        Args:
            report_id: Report ID
            from_line: Line to start reading from (for incremental fetching; 0 means from the beginning)
            
        Returns:
            {
                "logs": [list of log entries],
                "total_lines": total line count,
                "from_line": starting line number,
                "has_more": whether there are more logs
            }
        """
        log_path = cls._get_agent_log_path(report_id)
        
        if not os.path.exists(log_path):
            return {
                "logs": [],
                "total_lines": 0,
                "from_line": 0,
                "has_more": False
            }
        
        logs = []
        total_lines = 0
        
        with open(log_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                total_lines = i + 1
                if i >= from_line:
                    try:
                        log_entry = json.loads(line.strip())
                        logs.append(log_entry)
                    except json.JSONDecodeError:
                        # Skip lines that fail to parse
                        continue
        
        return {
            "logs": logs,
            "total_lines": total_lines,
            "from_line": from_line,
            "has_more": False  # Read to the end
        }
    
    @classmethod
    def get_agent_log_stream(cls, report_id: str) -> List[Dict[str, Any]]:
        """
        Get the complete Agent log (for fetching everything at once)
        
        Args:
            report_id: Report ID
            
        Returns:
            List of log entries
        """
        result = cls.get_agent_log(report_id, from_line=0)
        return result["logs"]
    
    @classmethod
    def save_outline(cls, report_id: str, outline: ReportOutline) -> None:
        """
        Save the report outline
        
        Called immediately after the planning phase completes
        """
        cls._ensure_report_folder(report_id)
        
        with open(cls._get_outline_path(report_id), 'w', encoding='utf-8') as f:
            json.dump(outline.to_dict(), f, ensure_ascii=False, indent=2)
        
        logger.info(t('report.outlineSaved', reportId=report_id))
    
    @classmethod
    def save_section(
        cls,
        report_id: str,
        section_index: int,
        section: ReportSection
    ) -> str:
        """
        Save a single section

        Called immediately after each section is generated, enabling section-by-section output

        Args:
            report_id: Report ID
            section_index: Section index (starting from 1)
            section: Section object

        Returns:
            Saved file path
        """
        cls._ensure_report_folder(report_id)

        # Build section Markdown content - clean up possible duplicate headings
        cleaned_content = cls._clean_section_content(section.content, section.title)
        md_content = f"## {section.title}\n\n"
        if cleaned_content:
            md_content += f"{cleaned_content}\n\n"

        # Save file
        file_suffix = f"section_{section_index:02d}.md"
        file_path = os.path.join(cls._get_report_folder(report_id), file_suffix)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        logger.info(t('report.sectionFileSaved', reportId=report_id, fileSuffix=file_suffix))
        return file_path
    
    @classmethod
    def _clean_section_content(cls, content: str, section_title: str) -> str:
        """
        Clean section content
        
        1. Remove Markdown heading lines at the start that duplicate the section title
        2. Convert all ### and lower-level headings to bold text
        
        Args:
            content: Original content
            section_title: Section title
            
        Returns:
            Cleaned content
        """
        import re
        
        if not content:
            return content
        
        content = content.strip()
        lines = content.split('\n')
        cleaned_lines = []
        skip_next_empty = False
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Check whether this is a Markdown heading line
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
            
            if heading_match:
                level = len(heading_match.group(1))
                title_text = heading_match.group(2).strip()
                
                # Check whether it duplicates the section title (skip duplicates within the first 5 lines)
                if i < 5:
                    if title_text == section_title or title_text.replace(' ', '') == section_title.replace(' ', ''):
                        skip_next_empty = True
                        continue
                
                # Convert headings of all levels (#, ##, ###, ####, etc.) to bold
                # since the section title is added by the system, the content should contain no headings
                cleaned_lines.append(f"**{title_text}**")
                cleaned_lines.append("")  # Add blank line
                continue
            
            # If the previous line was a skipped heading and the current line is empty, skip it too
            if skip_next_empty and stripped == '':
                skip_next_empty = False
                continue
            
            skip_next_empty = False
            cleaned_lines.append(line)
        
        # Remove leading blank lines
        while cleaned_lines and cleaned_lines[0].strip() == '':
            cleaned_lines.pop(0)
        
        # Remove leading horizontal rules
        while cleaned_lines and cleaned_lines[0].strip() in ['---', '***', '___']:
            cleaned_lines.pop(0)
            # Also remove blank lines after the horizontal rule
            while cleaned_lines and cleaned_lines[0].strip() == '':
                cleaned_lines.pop(0)
        
        return '\n'.join(cleaned_lines)
    
    @classmethod
    def update_progress(
        cls, 
        report_id: str, 
        status: str, 
        progress: int, 
        message: str,
        current_section: str = None,
        completed_sections: List[str] = None
    ) -> None:
        """
        Update report generation progress
        
        The frontend can read progress.json to get real-time progress
        """
        cls._ensure_report_folder(report_id)
        
        progress_data = {
            "status": status,
            "progress": progress,
            "message": message,
            "current_section": current_section,
            "completed_sections": completed_sections or [],
            "updated_at": datetime.now().isoformat()
        }
        
        with open(cls._get_progress_path(report_id), 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def get_progress(cls, report_id: str) -> Optional[Dict[str, Any]]:
        """Get report generation progress"""
        path = cls._get_progress_path(report_id)
        
        if not os.path.exists(path):
            return None
        
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @classmethod
    def get_generated_sections(cls, report_id: str) -> List[Dict[str, Any]]:
        """
        Get the list of generated sections
        
        Return info for all saved section files
        """
        folder = cls._get_report_folder(report_id)
        
        if not os.path.exists(folder):
            return []
        
        sections = []
        for filename in sorted(os.listdir(folder)):
            if filename.startswith('section_') and filename.endswith('.md'):
                file_path = os.path.join(folder, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Parse section index from the filename
                parts = filename.replace('.md', '').split('_')
                section_index = int(parts[1])

                sections.append({
                    "filename": filename,
                    "section_index": section_index,
                    "content": content
                })

        return sections
    
    @classmethod
    def assemble_full_report(cls, report_id: str, outline: ReportOutline) -> str:
        """
        Assemble the full report
        
        Assemble the full report from saved section files and clean up headings
        """
        folder = cls._get_report_folder(report_id)
        
        # Build report header
        md_content = f"# {outline.title}\n\n"
        md_content += f"> {outline.summary}\n\n"
        md_content += f"---\n\n"
        
        # Read all section files in order
        sections = cls.get_generated_sections(report_id)
        for section_info in sections:
            md_content += section_info["content"]
        
        # Post-process: clean up heading issues across the whole report
        md_content = cls._post_process_report(md_content, outline)
        
        # Save full report
        full_path = cls._get_report_markdown_path(report_id)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(t('report.fullReportAssembled', reportId=report_id))
        return md_content
    
    @classmethod
    def _post_process_report(cls, content: str, outline: ReportOutline) -> str:
        """
        Post-process report content
        
        1. Remove duplicate headings
        2. Keep the report main title (#) and section titles (##); remove other heading levels (###, ####, etc.)
        3. Clean up extra blank lines and horizontal rules
        
        Args:
            content: Original report content
            outline: Report outline
            
        Returns:
            Processed content
        """
        import re
        
        lines = content.split('\n')
        processed_lines = []
        prev_was_heading = False
        
        # Collect all section titles from the outline
        section_titles = set()
        for section in outline.sections:
            section_titles.add(section.title)
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Check whether this is a heading line
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
            
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                
                # Check for a duplicate heading (same heading text within 5 consecutive lines)
                is_duplicate = False
                for j in range(max(0, len(processed_lines) - 5), len(processed_lines)):
                    prev_line = processed_lines[j].strip()
                    prev_match = re.match(r'^(#{1,6})\s+(.+)$', prev_line)
                    if prev_match:
                        prev_title = prev_match.group(2).strip()
                        if prev_title == title:
                            is_duplicate = True
                            break
                
                if is_duplicate:
                    # Skip the duplicate heading and the blank lines after it
                    i += 1
                    while i < len(lines) and lines[i].strip() == '':
                        i += 1
                    continue
                
                # Heading level handling:
                # - # (level=1) keep only the report main title
                # - ## (level=2) keep section titles
                # - ### and below (level>=3) convert to bold text
                
                if level == 1:
                    if title == outline.title:
                        # Keep the report main title
                        processed_lines.append(line)
                        prev_was_heading = True
                    elif title in section_titles:
                        # Section title wrongly used #; fix to ##
                        processed_lines.append(f"## {title}")
                        prev_was_heading = True
                    else:
                        # Convert other level-1 headings to bold
                        processed_lines.append(f"**{title}**")
                        processed_lines.append("")
                        prev_was_heading = False
                elif level == 2:
                    if title in section_titles or title == outline.title:
                        # Keep section title
                        processed_lines.append(line)
                        prev_was_heading = True
                    else:
                        # Convert non-section level-2 headings to bold
                        processed_lines.append(f"**{title}**")
                        processed_lines.append("")
                        prev_was_heading = False
                else:
                    # Convert ### and lower-level headings to bold text
                    processed_lines.append(f"**{title}**")
                    processed_lines.append("")
                    prev_was_heading = False
                
                i += 1
                continue
            
            elif stripped == '---' and prev_was_heading:
                # Skip a horizontal rule immediately after a heading
                i += 1
                continue
            
            elif stripped == '' and prev_was_heading:
                # Keep only one blank line after a heading
                if processed_lines and processed_lines[-1].strip() != '':
                    processed_lines.append(line)
                prev_was_heading = False
            
            else:
                processed_lines.append(line)
                prev_was_heading = False
            
            i += 1
        
        # Collapse consecutive blank lines (keep at most 2)
        result_lines = []
        empty_count = 0
        for line in processed_lines:
            if line.strip() == '':
                empty_count += 1
                if empty_count <= 2:
                    result_lines.append(line)
            else:
                empty_count = 0
                result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    @classmethod
    def save_report(cls, report: Report) -> None:
        """Save report metadata and the full report"""
        cls._ensure_report_folder(report.report_id)
        
        # Save metadata JSON
        with open(cls._get_report_path(report.report_id), 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        
        # Save outline
        if report.outline:
            cls.save_outline(report.report_id, report.outline)
        
        # Save full Markdown report
        if report.markdown_content:
            with open(cls._get_report_markdown_path(report.report_id), 'w', encoding='utf-8') as f:
                f.write(report.markdown_content)
        
        logger.info(t('report.reportSaved', reportId=report.report_id))
    
    @classmethod
    def get_report(cls, report_id: str) -> Optional[Report]:
        """Get a report"""
        path = cls._get_report_path(report_id)
        
        if not os.path.exists(path):
            # Legacy format compatibility: check files stored directly under the reports directory
            old_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.json")
            if os.path.exists(old_path):
                path = old_path
            else:
                return None
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Rebuild the Report object
        outline = None
        if data.get('outline'):
            outline_data = data['outline']
            sections = []
            for s in outline_data.get('sections', []):
                sections.append(ReportSection(
                    title=s['title'],
                    content=s.get('content', '')
                ))
            outline = ReportOutline(
                title=outline_data['title'],
                summary=outline_data['summary'],
                sections=sections
            )
        
        # If markdown_content is empty, try reading from full_report.md
        markdown_content = data.get('markdown_content', '')
        if not markdown_content:
            full_report_path = cls._get_report_markdown_path(report_id)
            if os.path.exists(full_report_path):
                with open(full_report_path, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
        
        return Report(
            report_id=data['report_id'],
            simulation_id=data['simulation_id'],
            graph_id=data['graph_id'],
            simulation_requirement=data['simulation_requirement'],
            status=ReportStatus(data['status']),
            outline=outline,
            markdown_content=markdown_content,
            created_at=data.get('created_at', ''),
            completed_at=data.get('completed_at', ''),
            error=data.get('error')
        )
    
    @classmethod
    def get_report_by_simulation(cls, simulation_id: str) -> Optional[Report]:
        """Get a report by simulation ID"""
        cls._ensure_reports_dir()
        
        for item in os.listdir(cls.REPORTS_DIR):
            item_path = os.path.join(cls.REPORTS_DIR, item)
            # New format: folder
            if os.path.isdir(item_path):
                report = cls.get_report(item)
                if report and report.simulation_id == simulation_id:
                    return report
            # Legacy format compatibility: JSON file
            elif item.endswith('.json'):
                report_id = item[:-5]
                report = cls.get_report(report_id)
                if report and report.simulation_id == simulation_id:
                    return report
        
        return None
    
    @classmethod
    def list_reports(cls, simulation_id: Optional[str] = None, limit: int = 50) -> List[Report]:
        """List reports"""
        cls._ensure_reports_dir()
        
        reports = []
        for item in os.listdir(cls.REPORTS_DIR):
            item_path = os.path.join(cls.REPORTS_DIR, item)
            # New format: folder
            if os.path.isdir(item_path):
                report = cls.get_report(item)
                if report:
                    if simulation_id is None or report.simulation_id == simulation_id:
                        reports.append(report)
            # Legacy format compatibility: JSON file
            elif item.endswith('.json'):
                report_id = item[:-5]
                report = cls.get_report(report_id)
                if report:
                    if simulation_id is None or report.simulation_id == simulation_id:
                        reports.append(report)
        
        # Sort by creation time, newest first
        reports.sort(key=lambda r: r.created_at, reverse=True)
        
        return reports[:limit]
    
    @classmethod
    def delete_report(cls, report_id: str) -> bool:
        """Delete a report (the entire folder)"""
        import shutil
        
        folder_path = cls._get_report_folder(report_id)
        
        # New format: delete the entire folder
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            shutil.rmtree(folder_path)
            logger.info(t('report.reportFolderDeleted', reportId=report_id))
            return True
        
        # Legacy format compatibility: delete individual files
        deleted = False
        old_json_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.json")
        old_md_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.md")
        
        if os.path.exists(old_json_path):
            os.remove(old_json_path)
            deleted = True
        if os.path.exists(old_md_path):
            os.remove(old_md_path)
            deleted = True
        
        return deleted
