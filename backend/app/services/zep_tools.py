"""
Zep retrieval tools service
Wraps graph search, node reading, edge queries, and other tools for use by the Report Agent

Core retrieval tools (optimized):
1. InsightForge (deep insight retrieval) - the most powerful hybrid retrieval; auto-generates sub-questions and retrieves across multiple dimensions
2. PanoramaSearch (breadth search) - gets the full picture, including expired content
3. QuickSearch (simple search) - fast retrieval
"""

import time
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from zep_cloud import NotFoundError

from ..config import Config
from ..utils.logger import get_logger
from ..utils.llm_client import LLMClient
from ..utils.locale import get_locale, localized, t
from ..utils.zep_paging import fetch_all_nodes, fetch_all_edges
from ..utils.zep import (
    call_zep_read_with_retry,
    get_zep_client,
    normalize_zep_search_limit,
    normalize_zep_search_query,
)

logger = get_logger('mirofish.zep_tools')


@dataclass
class SearchResult:
    """Search result"""
    facts: List[str]
    edges: List[Dict[str, Any]]
    nodes: List[Dict[str, Any]]
    query: str
    total_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "facts": self.facts,
            "edges": self.edges,
            "nodes": self.nodes,
            "query": self.query,
            "total_count": self.total_count
        }
    
    def to_text(self) -> str:
        """Convert to text format for the LLM"""
        text_parts = [
            localized(f"搜索查询: {self.query}", f"Search query: {self.query}"),
            localized(f"找到 {self.total_count} 条相关信息", f"Found {self.total_count} relevant results"),
        ]
        
        if self.facts:
            text_parts.append(localized("\n### 相关事实:", "\n### Related facts:"))
            for i, fact in enumerate(self.facts, 1):
                text_parts.append(f"{i}. {fact}")
        
        return "\n".join(text_parts)


@dataclass
class NodeInfo:
    """Node information"""
    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "labels": self.labels,
            "summary": self.summary,
            "attributes": self.attributes
        }
    
    def to_text(self) -> str:
        """Convert to text format"""
        entity_type = next((l for l in self.labels if l not in ["Entity", "Node"]), localized("未知类型", "Unknown type"))
        return localized(
            f"实体: {self.name} (类型: {entity_type})\n摘要: {self.summary}",
            f"Entity: {self.name} (Type: {entity_type})\nSummary: {self.summary}"
        )


@dataclass
class EdgeInfo:
    """Edge information"""
    uuid: str
    name: str
    fact: str
    source_node_uuid: str
    target_node_uuid: str
    source_node_name: Optional[str] = None
    target_node_name: Optional[str] = None
    # Temporal information
    created_at: Optional[str] = None
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None
    expired_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "fact": self.fact,
            "source_node_uuid": self.source_node_uuid,
            "target_node_uuid": self.target_node_uuid,
            "source_node_name": self.source_node_name,
            "target_node_name": self.target_node_name,
            "created_at": self.created_at,
            "valid_at": self.valid_at,
            "invalid_at": self.invalid_at,
            "expired_at": self.expired_at
        }
    
    def to_text(self, include_temporal: bool = False) -> str:
        """Convert to text format"""
        source = self.source_node_name or self.source_node_uuid[:8]
        target = self.target_node_name or self.target_node_uuid[:8]
        base_text = localized(
            f"关系: {source} --[{self.name}]--> {target}\n事实: {self.fact}",
            f"Relation: {source} --[{self.name}]--> {target}\nFact: {self.fact}"
        )
        
        if include_temporal:
            valid_at = self.valid_at or localized("未知", "Unknown")
            invalid_at = self.invalid_at or localized("至今", "Present")
            base_text += localized(f"\n时效: {valid_at} - {invalid_at}", f"\nValidity: {valid_at} - {invalid_at}")
            if self.expired_at:
                base_text += localized(f" (已过期: {self.expired_at})", f" (Expired: {self.expired_at})")
        
        return base_text
    
    @property
    def is_expired(self) -> bool:
        """Whether it has expired"""
        return self.expired_at is not None
    
    @property
    def is_invalid(self) -> bool:
        """Whether it has been invalidated"""
        return self.invalid_at is not None


@dataclass
class InsightForgeResult:
    """
    Deep insight retrieval result (InsightForge)
    Contains retrieval results for multiple sub-questions plus a combined analysis
    """
    query: str
    simulation_requirement: str
    sub_queries: List[str]
    
    # Retrieval results per dimension
    semantic_facts: List[str] = field(default_factory=list)  # Semantic search results
    entity_insights: List[Dict[str, Any]] = field(default_factory=list)  # Entity insights
    relationship_chains: List[str] = field(default_factory=list)  # Relationship chains
    
    # Statistics
    total_facts: int = 0
    total_entities: int = 0
    total_relationships: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "simulation_requirement": self.simulation_requirement,
            "sub_queries": self.sub_queries,
            "semantic_facts": self.semantic_facts,
            "entity_insights": self.entity_insights,
            "relationship_chains": self.relationship_chains,
            "total_facts": self.total_facts,
            "total_entities": self.total_entities,
            "total_relationships": self.total_relationships
        }
    
    def to_text(self) -> str:
        """Convert to a detailed text format for the LLM"""
        text_parts = [
            localized(f"## 未来预测深度分析", f"## In-Depth Future Prediction Analysis"),
            localized(f"分析问题: {self.query}", f"Analysis question: {self.query}"),
            localized(f"预测场景: {self.simulation_requirement}", f"Prediction scenario: {self.simulation_requirement}"),
            localized(f"\n### 预测数据统计", f"\n### Prediction Data Statistics"),
            localized(f"- 相关预测事实: {self.total_facts}条", f"- Relevant prediction facts: {self.total_facts}"),
            localized(f"- 涉及实体: {self.total_entities}个", f"- Entities involved: {self.total_entities}"),
            localized(f"- 关系链: {self.total_relationships}条", f"- Relationship chains: {self.total_relationships}")
        ]
        
        # Sub-questions
        if self.sub_queries:
            text_parts.append(localized(f"\n### 分析的子问题", f"\n### Analyzed Sub-questions"))
            for i, sq in enumerate(self.sub_queries, 1):
                text_parts.append(f"{i}. {sq}")
        
        # Semantic search results
        if self.semantic_facts:
            text_parts.append(localized(
                f"\n### 【关键事实】(请在报告中引用这些原文)",
                f"\n### [Key Facts] (quote these verbatim in the report)"
            ))
            for i, fact in enumerate(self.semantic_facts, 1):
                text_parts.append(f"{i}. \"{fact}\"")
        
        # Entity insights
        if self.entity_insights:
            text_parts.append(localized(f"\n### 【核心实体】", f"\n### [Core Entities]"))
            for entity in self.entity_insights:
                if get_locale() == 'zh':
                    text_parts.append(f"- **{entity.get('name', '未知')}** ({entity.get('type', '实体')})")
                else:
                    text_parts.append(f"- **{entity.get('name', 'Unknown')}** ({entity.get('type', 'Entity')})")
                if entity.get('summary'):
                    text_parts.append(localized(
                        f"  摘要: \"{entity.get('summary')}\"",
                        f"  Summary: \"{entity.get('summary')}\""
                    ))
                if entity.get('related_facts'):
                    text_parts.append(localized(
                        f"  相关事实: {len(entity.get('related_facts', []))}条",
                        f"  Related facts: {len(entity.get('related_facts', []))}"
                    ))
        
        # Relationship chains
        if self.relationship_chains:
            text_parts.append(localized(f"\n### 【关系链】", f"\n### [Relationship Chains]"))
            for chain in self.relationship_chains:
                text_parts.append(f"- {chain}")
        
        return "\n".join(text_parts)


@dataclass
class PanoramaResult:
    """
    Breadth search result (Panorama)
    Contains all related information, including expired content
    """
    query: str
    
    # All nodes
    all_nodes: List[NodeInfo] = field(default_factory=list)
    # All edges (including expired ones)
    all_edges: List[EdgeInfo] = field(default_factory=list)
    # Currently valid facts
    active_facts: List[str] = field(default_factory=list)
    # Expired/invalidated facts (history)
    historical_facts: List[str] = field(default_factory=list)
    
    # Statistics
    total_nodes: int = 0
    total_edges: int = 0
    active_count: int = 0
    historical_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "all_nodes": [n.to_dict() for n in self.all_nodes],
            "all_edges": [e.to_dict() for e in self.all_edges],
            "active_facts": self.active_facts,
            "historical_facts": self.historical_facts,
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "active_count": self.active_count,
            "historical_count": self.historical_count
        }
    
    def to_text(self) -> str:
        """Convert to text format (full version, no truncation)"""
        text_parts = [
            localized(f"## 广度搜索结果（未来全景视图）", f"## Breadth Search Results (Future Panoramic View)"),
            localized(f"查询: {self.query}", f"Query: {self.query}"),
            localized(f"\n### 统计信息", f"\n### Statistics"),
            localized(f"- 总节点数: {self.total_nodes}", f"- Total nodes: {self.total_nodes}"),
            localized(f"- 总边数: {self.total_edges}", f"- Total edges: {self.total_edges}"),
            localized(f"- 当前有效事实: {self.active_count}条", f"- Currently valid facts: {self.active_count}"),
            localized(f"- 历史/过期事实: {self.historical_count}条", f"- Historical/expired facts: {self.historical_count}")
        ]
        
        # Currently valid facts (full output, no truncation)
        if self.active_facts:
            text_parts.append(localized(
                f"\n### 【当前有效事实】(模拟结果原文)",
                f"\n### [Currently Valid Facts] (original simulation output)"
            ))
            for i, fact in enumerate(self.active_facts, 1):
                text_parts.append(f"{i}. \"{fact}\"")
        
        # Historical/expired facts (full output, no truncation)
        if self.historical_facts:
            text_parts.append(localized(
                f"\n### 【历史/过期事实】(演变过程记录)",
                f"\n### [Historical/Expired Facts] (record of how events evolved)"
            ))
            for i, fact in enumerate(self.historical_facts, 1):
                text_parts.append(f"{i}. \"{fact}\"")
        
        # Key entities (full output, no truncation)
        if self.all_nodes:
            text_parts.append(localized(f"\n### 【涉及实体】", f"\n### [Entities Involved]"))
            for node in self.all_nodes:
                entity_type = next((l for l in node.labels if l not in ["Entity", "Node"]), localized("实体", "Entity"))
                text_parts.append(f"- **{node.name}** ({entity_type})")
        
        return "\n".join(text_parts)


@dataclass
class AgentInterview:
    """Interview result for a single agent"""
    agent_name: str
    agent_role: str  # Role type (e.g. student, teacher, media)
    agent_bio: str  # Bio
    question: str  # Interview question
    response: str  # Interview answer
    key_quotes: List[str] = field(default_factory=list)  # Key quotes
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "agent_role": self.agent_role,
            "agent_bio": self.agent_bio,
            "question": self.question,
            "response": self.response,
            "key_quotes": self.key_quotes
        }
    
    def to_text(self) -> str:
        text = f"**{self.agent_name}** ({self.agent_role})\n"
        # Show the full agent_bio without truncation
        text += localized(f"_简介: {self.agent_bio}_\n\n", f"_Bio: {self.agent_bio}_\n\n")
        text += f"**Q:** {self.question}\n\n"
        text += f"**A:** {self.response}\n"
        if self.key_quotes:
            text += localized("\n**关键引言:**\n", "\n**Key Quotes:**\n")
            for quote in self.key_quotes:
                # Strip various quotation marks
                clean_quote = quote.replace('\u201c', '').replace('\u201d', '').replace('"', '')
                clean_quote = clean_quote.replace('\u300c', '').replace('\u300d', '')
                clean_quote = clean_quote.strip()
                # Remove leading punctuation
                while clean_quote and clean_quote[0] in '，,；;：:、。！？\n\r\t ':
                    clean_quote = clean_quote[1:]
                # Filter out junk containing question numbers (Question 1-9)
                skip = False
                for d in '123456789':
                    if f'\u95ee\u9898{d}' in clean_quote or f'Question {d}' in clean_quote:
                        skip = True
                        break
                if skip:
                    continue
                # Truncate overly long content (at a full stop rather than a hard cut)
                if len(clean_quote) > 150:
                    dot_pos = clean_quote.find('\u3002', 80)
                    if dot_pos > 0:
                        clean_quote = clean_quote[:dot_pos + 1]
                    else:
                        clean_quote = clean_quote[:147] + "..."
                if clean_quote and len(clean_quote) >= 10:
                    text += f'> "{clean_quote}"\n'
        return text


@dataclass
class InterviewResult:
    """
    Interview result (Interview)
    Contains interview answers from multiple simulated agents
    """
    interview_topic: str  # Interview topic
    interview_questions: List[str]  # List of interview questions
    
    # Agents selected for interview
    selected_agents: List[Dict[str, Any]] = field(default_factory=list)
    # Each agent's interview answers
    interviews: List[AgentInterview] = field(default_factory=list)
    
    # Reasoning for agent selection
    selection_reasoning: str = ""
    # Consolidated interview summary
    summary: str = ""
    
    # Statistics
    total_agents: int = 0
    interviewed_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "interview_topic": self.interview_topic,
            "interview_questions": self.interview_questions,
            "selected_agents": self.selected_agents,
            "interviews": [i.to_dict() for i in self.interviews],
            "selection_reasoning": self.selection_reasoning,
            "summary": self.summary,
            "total_agents": self.total_agents,
            "interviewed_count": self.interviewed_count
        }
    
    def to_text(self) -> str:
        """Convert to a detailed text format for the LLM and for report citations"""
        text_parts = [
            localized("## 深度采访报告", "## In-Depth Interview Report"),
            localized(f"**采访主题:** {self.interview_topic}", f"**Interview topic:** {self.interview_topic}"),
            localized(
                f"**采访人数:** {self.interviewed_count} / {self.total_agents} 位模拟Agent",
                f"**Interviewees:** {self.interviewed_count} / {self.total_agents} simulated agents"
            ),
            localized("\n### 采访对象选择理由", "\n### Interviewee Selection Rationale"),
            self.selection_reasoning or localized("（自动选择）", "(Automatically selected)"),
            "\n---",
            localized("\n### 采访实录", "\n### Interview Transcripts"),
        ]

        if self.interviews:
            for i, interview in enumerate(self.interviews, 1):
                text_parts.append(localized(
                    f"\n#### 采访 #{i}: {interview.agent_name}",
                    f"\n#### Interview #{i}: {interview.agent_name}"
                ))
                text_parts.append(interview.to_text())
                text_parts.append("\n---")
        else:
            text_parts.append(localized("（无采访记录）\n\n---", "(No interview records)\n\n---"))

        text_parts.append(localized("\n### 采访摘要与核心观点", "\n### Interview Summary and Key Viewpoints"))
        text_parts.append(self.summary or localized("（无摘要）", "(No summary)"))

        return "\n".join(text_parts)


class ZepToolsService:
    """
    Zep retrieval tools service
    
    [Core retrieval tools - optimized]
    1. insight_forge - deep insight retrieval (most powerful; auto-generates sub-questions, multi-dimensional retrieval)
    2. panorama_search - breadth search (gets the full picture, including expired content)
    3. quick_search - simple search (fast retrieval)
    4. interview_agents - in-depth interviews (interviews simulated agents for multiple perspectives)
    
    [Basic tools]
    - search_graph - graph semantic search
    - get_all_nodes - get all graph nodes
    - get_all_edges - get all graph edges (with temporal info)
    - get_node_detail - get node details
    - get_node_edges - get edges related to a node
    - get_entities_by_type - get entities by type
    - get_entity_summary - get an entity's relationship summary
    """
    
    # Retry configuration
    MAX_RETRIES = 3
    RETRY_DELAY = 2.0
    
    def __init__(self, api_key: Optional[str] = None, llm_client: Optional[LLMClient] = None):
        self.api_key = api_key or Config.ZEP_API_KEY
        if not self.api_key:
            raise ValueError("ZEP_API_KEY is not configured")
        
        self.client = get_zep_client(self.api_key)
        # LLM client used by InsightForge to generate sub-questions
        self._llm_client = llm_client
        logger.info(t("console.zepToolsInitialized"))
    
    @property
    def llm(self) -> LLMClient:
        """Lazily initialize the LLM client"""
        if self._llm_client is None:
            self._llm_client = LLMClient()
        return self._llm_client
    
    def _call_with_retry(self, func, operation_name: str, max_retries: int = None):
        """Retry one safe read using typed Zep/HTTPX error classification."""

        return call_zep_read_with_retry(
            func,
            operation_name=operation_name,
            max_attempts=max_retries or self.MAX_RETRIES,
            initial_delay=self.RETRY_DELAY,
        )
    
    def search_graph(
        self, 
        graph_id: str, 
        query: str, 
        limit: int = 10,
        scope: str = "edges"
    ) -> SearchResult:
        """
        Graph semantic search
        
        Uses hybrid search (semantic + BM25) to find related information in the graph.
        If Zep Cloud's search API is unavailable, falls back to local keyword matching.
        
        Args:
            graph_id: Graph ID (Standalone Graph)
            query: Search query
            limit: Number of results to return
            scope: Search scope, "edges" or "nodes"
            
        Returns:
            SearchResult: Search result
        """
        logger.info(t("console.graphSearch", graphId=graph_id, query=query[:50]))
        
        zep_query = normalize_zep_search_query(query)
        zep_limit = normalize_zep_search_limit(limit)

        try:
            search_results = self._call_with_retry(
                func=lambda: self.client.graph.search(
                    graph_id=graph_id,
                    query=zep_query,
                    limit=zep_limit,
                    scope=scope,
                    reranker="cross_encoder"
                ),
                operation_name=t("console.graphSearchOp", graphId=graph_id)
            )
            
            facts = []
            edges = []
            nodes = []
            
            # Parse edge search results
            if hasattr(search_results, 'edges') and search_results.edges:
                for edge in search_results.edges:
                    if hasattr(edge, 'fact') and edge.fact:
                        facts.append(edge.fact)
                    edges.append({
                        "uuid": getattr(edge, 'uuid_', None) or getattr(edge, 'uuid', ''),
                        "name": getattr(edge, 'name', ''),
                        "fact": getattr(edge, 'fact', ''),
                        "source_node_uuid": getattr(edge, 'source_node_uuid', ''),
                        "target_node_uuid": getattr(edge, 'target_node_uuid', ''),
                    })
            
            # Parse node search results
            if hasattr(search_results, 'nodes') and search_results.nodes:
                for node in search_results.nodes:
                    nodes.append({
                        "uuid": getattr(node, 'uuid_', None) or getattr(node, 'uuid', ''),
                        "name": getattr(node, 'name', ''),
                        "labels": getattr(node, 'labels', []),
                        "summary": getattr(node, 'summary', ''),
                    })
                    # Node summaries also count as facts
                    if hasattr(node, 'summary') and node.summary:
                        facts.append(f"[{node.name}]: {node.summary}")
            
            logger.info(t("console.searchComplete", count=len(facts)))
            
            return SearchResult(
                facts=facts,
                edges=edges,
                nodes=nodes,
                query=query,
                total_count=len(facts)
            )
            
        except Exception as e:
            # Authentication, invalid input, missing graphs, and exhausted
            # transient failures must remain visible to the report workflow.
            logger.error(t("console.zepSearchApiFallback", error=str(e)))
            raise
    
    def _local_search(
        self, 
        graph_id: str, 
        query: str, 
        limit: int = 10,
        scope: str = "edges"
    ) -> SearchResult:
        """
        Local keyword-matching search (fallback for the Zep Search API)
        
        Fetches all edges/nodes, then performs keyword matching locally
        
        Args:
            graph_id: Graph ID
            query: Search query
            limit: Number of results to return
            scope: Search scope
            
        Returns:
            SearchResult: Search result
        """
        logger.info(t("console.usingLocalSearch", query=query[:30]))
        
        facts = []
        edges_result = []
        nodes_result = []
        
        # Extract query keywords (simple tokenization)
        query_lower = query.lower()
        keywords = [w.strip() for w in query_lower.replace(',', ' ').replace('，', ' ').split() if len(w.strip()) > 1]
        
        def match_score(text: str) -> int:
            """Compute the match score between text and the query"""
            if not text:
                return 0
            text_lower = text.lower()
            # Exact query match
            if query_lower in text_lower:
                return 100
            # Keyword match
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 10
            return score
        
        try:
            if scope in ["edges", "both"]:
                # Fetch all edges and match
                all_edges = self.get_all_edges(graph_id)
                scored_edges = []
                for edge in all_edges:
                    score = match_score(edge.fact) + match_score(edge.name)
                    if score > 0:
                        scored_edges.append((score, edge))
                
                # Sort by score
                scored_edges.sort(key=lambda x: x[0], reverse=True)
                
                for score, edge in scored_edges[:limit]:
                    if edge.fact:
                        facts.append(edge.fact)
                    edges_result.append({
                        "uuid": edge.uuid,
                        "name": edge.name,
                        "fact": edge.fact,
                        "source_node_uuid": edge.source_node_uuid,
                        "target_node_uuid": edge.target_node_uuid,
                    })
            
            if scope in ["nodes", "both"]:
                # Fetch all nodes and match
                all_nodes = self.get_all_nodes(graph_id)
                scored_nodes = []
                for node in all_nodes:
                    score = match_score(node.name) + match_score(node.summary)
                    if score > 0:
                        scored_nodes.append((score, node))
                
                scored_nodes.sort(key=lambda x: x[0], reverse=True)
                
                for score, node in scored_nodes[:limit]:
                    nodes_result.append({
                        "uuid": node.uuid,
                        "name": node.name,
                        "labels": node.labels,
                        "summary": node.summary,
                    })
                    if node.summary:
                        facts.append(f"[{node.name}]: {node.summary}")
            
            logger.info(t("console.localSearchComplete", count=len(facts)))
            
        except Exception as e:
            logger.error(t("console.localSearchFailed", error=str(e)))
        
        return SearchResult(
            facts=facts,
            edges=edges_result,
            nodes=nodes_result,
            query=query,
            total_count=len(facts)
        )
    
    def get_all_nodes(self, graph_id: str) -> List[NodeInfo]:
        """
        Get all nodes of the graph (paginated)

        Args:
            graph_id: Graph ID

        Returns:
            List of nodes
        """
        logger.info(t("console.fetchingAllNodes", graphId=graph_id))

        nodes = fetch_all_nodes(self.client, graph_id)

        result = []
        for node in nodes:
            node_uuid = getattr(node, 'uuid_', None) or getattr(node, 'uuid', None) or ""
            result.append(NodeInfo(
                uuid=str(node_uuid) if node_uuid else "",
                name=node.name or "",
                labels=node.labels or [],
                summary=node.summary or "",
                attributes=node.attributes or {}
            ))

        logger.info(t("console.fetchedNodes", count=len(result)))
        return result

    def get_all_edges(self, graph_id: str, include_temporal: bool = True) -> List[EdgeInfo]:
        """
        Get all edges of the graph (paginated, with temporal info)

        Args:
            graph_id: Graph ID
            include_temporal: Whether to include temporal info (default True)

        Returns:
            List of edges (with created_at, valid_at, invalid_at, expired_at)
        """
        logger.info(t("console.fetchingAllEdges", graphId=graph_id))

        edges = fetch_all_edges(self.client, graph_id)

        result = []
        for edge in edges:
            edge_uuid = getattr(edge, 'uuid_', None) or getattr(edge, 'uuid', None) or ""
            edge_info = EdgeInfo(
                uuid=str(edge_uuid) if edge_uuid else "",
                name=edge.name or "",
                fact=edge.fact or "",
                source_node_uuid=edge.source_node_uuid or "",
                target_node_uuid=edge.target_node_uuid or ""
            )

            # Add temporal info
            if include_temporal:
                edge_info.created_at = getattr(edge, 'created_at', None)
                edge_info.valid_at = getattr(edge, 'valid_at', None)
                edge_info.invalid_at = getattr(edge, 'invalid_at', None)
                edge_info.expired_at = getattr(edge, 'expired_at', None)

            result.append(edge_info)

        logger.info(t("console.fetchedEdges", count=len(result)))
        return result
    
    def get_node_detail(self, node_uuid: str) -> Optional[NodeInfo]:
        """
        Get detailed information for a single node
        
        Args:
            node_uuid: Node UUID
            
        Returns:
            Node info or None
        """
        logger.info(t("console.fetchingNodeDetail", uuid=node_uuid[:8]))
        
        try:
            node = self._call_with_retry(
                func=lambda: self.client.graph.node.get(uuid_=node_uuid),
                operation_name=t("console.fetchNodeDetailOp", uuid=node_uuid[:8])
            )
            
            if not node:
                return None
            
            return NodeInfo(
                uuid=getattr(node, 'uuid_', None) or getattr(node, 'uuid', ''),
                name=node.name or "",
                labels=node.labels or [],
                summary=node.summary or "",
                attributes=node.attributes or {}
            )
        except NotFoundError:
            return None
        except Exception as e:
            logger.error(t("console.fetchNodeDetailFailed", error=str(e)))
            raise
    
    def get_node_edges(self, graph_id: str, node_uuid: str) -> List[EdgeInfo]:
        """
        Get all edges related to a node
        
        Fetches all graph edges, then filters those related to the given node
        
        Args:
            graph_id: Graph ID
            node_uuid: Node UUID
            
        Returns:
            List of edges
        """
        logger.info(t("console.fetchingNodeEdges", uuid=node_uuid[:8]))
        
        try:
            # Fetch all graph edges, then filter
            all_edges = self.get_all_edges(graph_id)
            
            result = []
            for edge in all_edges:
                # Check whether the edge involves the given node (as source or target)
                if edge.source_node_uuid == node_uuid or edge.target_node_uuid == node_uuid:
                    result.append(edge)
            
            logger.info(t("console.foundNodeEdges", count=len(result)))
            return result
            
        except Exception as e:
            logger.error(t("console.fetchNodeEdgesFailed", error=str(e)))
            raise
    
    def get_entities_by_type(
        self, 
        graph_id: str, 
        entity_type: str
    ) -> List[NodeInfo]:
        """
        Get entities by type
        
        Args:
            graph_id: Graph ID
            entity_type: Entity type (e.g. Student, PublicFigure)
            
        Returns:
            List of entities matching the type
        """
        logger.info(t("console.fetchingEntitiesByType", type=entity_type))
        
        all_nodes = self.get_all_nodes(graph_id)
        
        filtered = []
        for node in all_nodes:
            # Check whether labels include the given type
            if entity_type in node.labels:
                filtered.append(node)
        
        logger.info(t("console.foundEntitiesByType", count=len(filtered), type=entity_type))
        return filtered
    
    def get_entity_summary(
        self, 
        graph_id: str, 
        entity_name: str
    ) -> Dict[str, Any]:
        """
        Get the relationship summary for the given entity
        
        Searches all information related to the entity and generates a summary
        
        Args:
            graph_id: Graph ID
            entity_name: Entity name
            
        Returns:
            Entity summary info
        """
        logger.info(t("console.fetchingEntitySummary", name=entity_name))
        
        # First search for information related to the entity
        search_result = self.search_graph(
            graph_id=graph_id,
            query=entity_name,
            limit=20
        )
        
        # Try to find the entity among all nodes
        all_nodes = self.get_all_nodes(graph_id)
        entity_node = None
        for node in all_nodes:
            if node.name.lower() == entity_name.lower():
                entity_node = node
                break
        
        related_edges = []
        if entity_node:
            # Pass the graph_id argument
            related_edges = self.get_node_edges(graph_id, entity_node.uuid)
        
        return {
            "entity_name": entity_name,
            "entity_info": entity_node.to_dict() if entity_node else None,
            "related_facts": search_result.facts,
            "related_edges": [e.to_dict() for e in related_edges],
            "total_relations": len(related_edges)
        }
    
    def get_graph_statistics(self, graph_id: str) -> Dict[str, Any]:
        """
        Get graph statistics
        
        Args:
            graph_id: Graph ID
            
        Returns:
            Statistics
        """
        logger.info(t("console.fetchingGraphStats", graphId=graph_id))
        
        nodes = self.get_all_nodes(graph_id)
        edges = self.get_all_edges(graph_id)
        
        # Count entity type distribution
        entity_types = {}
        for node in nodes:
            for label in node.labels:
                if label not in ["Entity", "Node"]:
                    entity_types[label] = entity_types.get(label, 0) + 1
        
        # Count relation type distribution
        relation_types = {}
        for edge in edges:
            relation_types[edge.name] = relation_types.get(edge.name, 0) + 1
        
        return {
            "graph_id": graph_id,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "entity_types": entity_types,
            "relation_types": relation_types
        }
    
    def get_simulation_context(
        self, 
        graph_id: str,
        simulation_requirement: str,
        limit: int = 30
    ) -> Dict[str, Any]:
        """
        Get simulation-related context
        
        Comprehensively searches all information related to the simulation requirement
        
        Args:
            graph_id: Graph ID
            simulation_requirement: Simulation requirement description
            limit: Limit on the number of items per category
            
        Returns:
            Simulation context info
        """
        logger.info(t("console.fetchingSimContext", requirement=simulation_requirement[:50]))
        
        # Search for information related to the simulation requirement
        search_result = self.search_graph(
            graph_id=graph_id,
            query=simulation_requirement,
            limit=limit
        )
        
        # Get graph statistics
        stats = self.get_graph_statistics(graph_id)
        
        # Get all entity nodes
        all_nodes = self.get_all_nodes(graph_id)
        
        # Filter entities with an actual type (not plain Entity nodes)
        entities = []
        for node in all_nodes:
            custom_labels = [l for l in node.labels if l not in ["Entity", "Node"]]
            if custom_labels:
                entities.append({
                    "name": node.name,
                    "type": custom_labels[0],
                    "summary": node.summary
                })
        
        return {
            "simulation_requirement": simulation_requirement,
            "related_facts": search_result.facts,
            "graph_statistics": stats,
            "entities": entities[:limit],  # Limit count
            "total_entities": len(entities)
        }
    
    # ========== Core retrieval tools (optimized) ==========
    
    def insight_forge(
        self,
        graph_id: str,
        query: str,
        simulation_requirement: str,
        report_context: str = "",
        max_sub_queries: int = 5
    ) -> InsightForgeResult:
        """
        [InsightForge - deep insight retrieval]
        
        The most powerful hybrid retrieval function; automatically decomposes the question and retrieves across multiple dimensions:
        1. Use the LLM to decompose the question into multiple sub-questions
        2. Run a semantic search for each sub-question
        3. Extract related entities and fetch their details
        4. Trace relationship chains
        5. Combine all results to produce deep insights
        
        Args:
            graph_id: Graph ID
            query: User question
            simulation_requirement: Simulation requirement description
            report_context: Report context (optional; for more precise sub-question generation)
            max_sub_queries: Maximum number of sub-questions
            
        Returns:
            InsightForgeResult: Deep insight retrieval result
        """
        logger.info(t("console.insightForgeStart", query=query[:50]))
        
        result = InsightForgeResult(
            query=query,
            simulation_requirement=simulation_requirement,
            sub_queries=[]
        )
        
        # Step 1: Generate sub-questions with the LLM
        sub_queries = self._generate_sub_queries(
            query=query,
            simulation_requirement=simulation_requirement,
            report_context=report_context,
            max_queries=max_sub_queries
        )
        result.sub_queries = sub_queries
        logger.info(t("console.generatedSubQueries", count=len(sub_queries)))
        
        # Step 2: Run a semantic search for each sub-question
        all_facts = []
        all_edges = []
        seen_facts = set()
        
        for sub_query in sub_queries:
            search_result = self.search_graph(
                graph_id=graph_id,
                query=sub_query,
                limit=15,
                scope="edges"
            )
            
            for fact in search_result.facts:
                if fact not in seen_facts:
                    all_facts.append(fact)
                    seen_facts.add(fact)
            
            all_edges.extend(search_result.edges)
        
        # Also search the original question
        main_search = self.search_graph(
            graph_id=graph_id,
            query=query,
            limit=20,
            scope="edges"
        )
        for fact in main_search.facts:
            if fact not in seen_facts:
                all_facts.append(fact)
                seen_facts.add(fact)
        
        result.semantic_facts = all_facts
        result.total_facts = len(all_facts)
        
        # Step 3: Extract related entity UUIDs from edges and fetch only those entities (not all nodes)
        entity_uuids = set()
        for edge_data in all_edges:
            if isinstance(edge_data, dict):
                source_uuid = edge_data.get('source_node_uuid', '')
                target_uuid = edge_data.get('target_node_uuid', '')
                if source_uuid:
                    entity_uuids.add(source_uuid)
                if target_uuid:
                    entity_uuids.add(target_uuid)
        
        # Fetch details for all related entities (no limit, full output)
        entity_insights = []
        node_map = {}  # Used later to build relationship chains
        
        for uuid in list(entity_uuids):  # Process all entities, no truncation
            if not uuid:
                continue
            try:
                # Fetch each related node individually
                node = self.get_node_detail(uuid)
                if node:
                    node_map[uuid] = node
                    entity_type = next((l for l in node.labels if l not in ["Entity", "Node"]), localized("实体", "Entity"))
                    
                    # Get all facts related to this entity (no truncation)
                    related_facts = [
                        f for f in all_facts 
                        if node.name.lower() in f.lower()
                    ]
                    
                    entity_insights.append({
                        "uuid": node.uuid,
                        "name": node.name,
                        "type": entity_type,
                        "summary": node.summary,
                        "related_facts": related_facts  # Full output, no truncation
                    })
            except Exception as e:
                logger.debug(f"Failed to fetch node {uuid}: {e}")
                continue
        
        result.entity_insights = entity_insights
        result.total_entities = len(entity_insights)
        
        # Step 4: Build all relationship chains (no limit)
        relationship_chains = []
        for edge_data in all_edges:  # Process all edges, no truncation
            if isinstance(edge_data, dict):
                source_uuid = edge_data.get('source_node_uuid', '')
                target_uuid = edge_data.get('target_node_uuid', '')
                relation_name = edge_data.get('name', '')
                
                source_name = node_map.get(source_uuid, NodeInfo('', '', [], '', {})).name or source_uuid[:8]
                target_name = node_map.get(target_uuid, NodeInfo('', '', [], '', {})).name or target_uuid[:8]
                
                chain = f"{source_name} --[{relation_name}]--> {target_name}"
                if chain not in relationship_chains:
                    relationship_chains.append(chain)
        
        result.relationship_chains = relationship_chains
        result.total_relationships = len(relationship_chains)
        
        logger.info(t("console.insightForgeComplete", facts=result.total_facts, entities=result.total_entities, relationships=result.total_relationships))
        return result
    
    def _generate_sub_queries(
        self,
        query: str,
        simulation_requirement: str,
        report_context: str = "",
        max_queries: int = 5
    ) -> List[str]:
        """
        Generate sub-questions with the LLM
        
        Decomposes a complex question into multiple independently retrievable sub-questions
        """
        system_prompt = localized("""你是一个专业的问题分析专家。你的任务是将一个复杂问题分解为多个可以在模拟世界中独立观察的子问题。

要求：
1. 每个子问题应该足够具体，可以在模拟世界中找到相关的Agent行为或事件
2. 子问题应该覆盖原问题的不同维度（如：谁、什么、为什么、怎么样、何时、何地）
3. 子问题应该与模拟场景相关
4. 返回JSON格式：{"sub_queries": ["子问题1", "子问题2", ...]}""", """You are a professional question-analysis expert. Your task is to decompose a complex question into multiple sub-questions that can each be observed independently in the simulated world.

Requirements:
1. Each sub-question should be specific enough that related agent behaviors or events can be found in the simulated world
2. Sub-questions should cover different dimensions of the original question (e.g. who, what, why, how, when, where)
3. Sub-questions should be relevant to the simulation scenario
4. Return JSON format: {"sub_queries": ["Sub-question 1", "Sub-question 2", ...]}""")

        if get_locale() == 'zh':
            user_prompt = f"""模拟需求背景：
{simulation_requirement}

{f"报告上下文：{report_context[:500]}" if report_context else ""}

请将以下问题分解为{max_queries}个子问题：
{query}

返回JSON格式的子问题列表。"""
        else:
            user_prompt = f"""Simulation requirement background:
{simulation_requirement}

{f"Report context: {report_context[:500]}" if report_context else ""}

Please decompose the following question into {max_queries} sub-questions:
{query}

Return the list of sub-questions in JSON format."""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            
            sub_queries = response.get("sub_queries", [])
            # Ensure it's a list of strings
            return [str(sq) for sq in sub_queries[:max_queries]]
            
        except Exception as e:
            logger.warning(t("console.generateSubQueriesFailed", error=str(e)))
            # Fallback: return variants based on the original question
            return [
                query,
                localized(f"{query} 的主要参与者", f"Main participants in {query}"),
                localized(f"{query} 的原因和影响", f"Causes and effects of {query}"),
                localized(f"{query} 的发展过程", f"How {query} developed")
            ][:max_queries]
    
    def panorama_search(
        self,
        graph_id: str,
        query: str,
        include_expired: bool = True,
        limit: int = 50
    ) -> PanoramaResult:
        """
        [PanoramaSearch - breadth search]
        
        Gets a full-picture view, including all related content and historical/expired information:
        1. Get all related nodes
        2. Get all edges (including expired/invalidated ones)
        3. Categorize currently valid vs. historical information
        
        This tool suits scenarios that need the full picture of an event or to trace how it evolved.
        
        Args:
            graph_id: Graph ID
            query: Search query (used for relevance ranking)
            include_expired: Whether to include expired content (default True)
            limit: Limit on the number of results returned
            
        Returns:
            PanoramaResult: Breadth search result
        """
        logger.info(t("console.panoramaSearchStart", query=query[:50]))
        
        result = PanoramaResult(query=query)
        
        # Get all nodes
        all_nodes = self.get_all_nodes(graph_id)
        node_map = {n.uuid: n for n in all_nodes}
        result.all_nodes = all_nodes
        result.total_nodes = len(all_nodes)
        
        # Get all edges (with temporal info)
        all_edges = self.get_all_edges(graph_id, include_temporal=True)
        result.all_edges = all_edges
        result.total_edges = len(all_edges)
        
        # Categorize facts
        active_facts = []
        historical_facts = []
        
        for edge in all_edges:
            if not edge.fact:
                continue
            
            # Add entity names to the fact
            source_name = node_map.get(edge.source_node_uuid, NodeInfo('', '', [], '', {})).name or edge.source_node_uuid[:8]
            target_name = node_map.get(edge.target_node_uuid, NodeInfo('', '', [], '', {})).name or edge.target_node_uuid[:8]
            
            # Determine whether expired/invalidated
            is_historical = edge.is_expired or edge.is_invalid
            
            if is_historical:
                # Historical/expired fact; add a time marker
                valid_at = edge.valid_at or localized("未知", "Unknown")
                invalid_at = edge.invalid_at or edge.expired_at or localized("未知", "Unknown")
                fact_with_time = f"[{valid_at} - {invalid_at}] {edge.fact}"
                historical_facts.append(fact_with_time)
            else:
                # Currently valid fact
                active_facts.append(edge.fact)
        
        # Rank by relevance to the query
        query_lower = query.lower()
        keywords = [w.strip() for w in query_lower.replace(',', ' ').replace('，', ' ').split() if len(w.strip()) > 1]
        
        def relevance_score(fact: str) -> int:
            fact_lower = fact.lower()
            score = 0
            if query_lower in fact_lower:
                score += 100
            for kw in keywords:
                if kw in fact_lower:
                    score += 10
            return score
        
        # Sort and limit count
        active_facts.sort(key=relevance_score, reverse=True)
        historical_facts.sort(key=relevance_score, reverse=True)
        
        result.active_facts = active_facts[:limit]
        result.historical_facts = historical_facts[:limit] if include_expired else []
        result.active_count = len(active_facts)
        result.historical_count = len(historical_facts)
        
        logger.info(t("console.panoramaSearchComplete", active=result.active_count, historical=result.historical_count))
        return result
    
    def quick_search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10
    ) -> SearchResult:
        """
        [QuickSearch - simple search]
        
        Fast, lightweight retrieval tool:
        1. Calls Zep semantic search directly
        2. Returns the most relevant results
        3. Suited to simple, direct retrieval needs
        
        Args:
            graph_id: Graph ID
            query: Search query
            limit: Number of results to return
            
        Returns:
            SearchResult: Search result
        """
        logger.info(t("console.quickSearchStart", query=query[:50]))
        
        # Call the existing search_graph method directly
        result = self.search_graph(
            graph_id=graph_id,
            query=query,
            limit=limit,
            scope="edges"
        )
        
        logger.info(t("console.quickSearchComplete", count=result.total_count))
        return result
    
    def interview_agents(
        self,
        simulation_id: str,
        interview_requirement: str,
        simulation_requirement: str = "",
        max_agents: int = 5,
        custom_questions: List[str] = None
    ) -> InterviewResult:
        """
        [InterviewAgents - in-depth interviews]
        
        Calls the real OASIS interview API to interview agents running in the simulation:
        1. Automatically reads profile files to learn about all simulated agents
        2. Uses the LLM to analyze the interview requirement and intelligently select the most relevant agents
        3. Uses the LLM to generate interview questions
        4. Calls the /api/simulation/interview/batch endpoint for real interviews (both platforms simultaneously)
        5. Combines all interview results into an interview report
        
        [Important] This feature requires the simulation environment to be running (OASIS environment not closed)
        
        [Use cases]
        - Understanding views on an event from different role perspectives
        - Collecting opinions and viewpoints from multiple parties
        - Getting real answers from simulated agents (not LLM-simulated)
        
        Args:
            simulation_id: Simulation ID (used to locate profile files and call the interview API)
            interview_requirement: Interview requirement description (unstructured, e.g. "understand students' views on the event")
            simulation_requirement: Simulation requirement background (optional)
            max_agents: Maximum number of agents to interview
            custom_questions: Custom interview questions (optional; auto-generated if not provided)
            
        Returns:
            InterviewResult: Interview result
        """
        from .simulation_runner import SimulationRunner
        
        logger.info(t("console.interviewAgentsStart", requirement=interview_requirement[:50]))
        
        result = InterviewResult(
            interview_topic=interview_requirement,
            interview_questions=custom_questions or []
        )
        
        # Step 1: Read profile files
        profiles = self._load_agent_profiles(simulation_id)
        
        if not profiles:
            logger.warning(t("console.profilesNotFound", simId=simulation_id))
            result.summary = localized("未找到可采访的Agent人设文件", "No agent profile files available for interview were found")
            return result
        
        result.total_agents = len(profiles)
        logger.info(t("console.loadedProfiles", count=len(profiles)))
        
        # Step 2: Use the LLM to select agents to interview (returns a list of agent_ids)
        selected_agents, selected_indices, selection_reasoning = self._select_agents_for_interview(
            profiles=profiles,
            interview_requirement=interview_requirement,
            simulation_requirement=simulation_requirement,
            max_agents=max_agents
        )
        
        result.selected_agents = selected_agents
        result.selection_reasoning = selection_reasoning
        logger.info(t("console.selectedAgentsForInterview", count=len(selected_agents), indices=selected_indices))
        
        # Step 3: Generate interview questions (if not provided)
        if not result.interview_questions:
            result.interview_questions = self._generate_interview_questions(
                interview_requirement=interview_requirement,
                simulation_requirement=simulation_requirement,
                selected_agents=selected_agents
            )
            logger.info(t("console.generatedInterviewQuestions", count=len(result.interview_questions)))
        
        # Merge the questions into a single interview prompt
        combined_prompt = "\n".join([f"{i+1}. {q}" for i, q in enumerate(result.interview_questions)])
        
        # Add an optimized prefix to constrain the agent's reply format
        INTERVIEW_PROMPT_PREFIX = localized(
            "你正在接受一次采访。请结合你的人设、所有的过往记忆与行动，"
            "以纯文本方式直接回答以下问题。\n"
            "回复要求：\n"
            "1. 直接用自然语言回答，不要调用任何工具\n"
            "2. 不要返回JSON格式或工具调用格式\n"
            "3. 不要使用Markdown标题（如#、##、###）\n"
            "4. 按问题编号逐一回答，每个回答以「问题X：」开头（X为问题编号）\n"
            "5. 每个问题的回答之间用空行分隔\n"
            "6. 回答要有实质内容，每个问题至少回答2-3句话\n\n",
            "You are being interviewed. Drawing on your persona and all of your past memories and actions, "
            "answer the following questions directly in plain text.\n"
            "Response requirements:\n"
            "1. Answer directly in natural language; do not call any tools\n"
            "2. Do not return JSON or tool-call format\n"
            "3. Do not use Markdown headings (such as #, ##, ###)\n"
            "4. Answer the questions one by one in numbered order, starting each answer with \"Question X:\" (X is the question number)\n"
            "5. Separate the answers to different questions with a blank line\n"
            "6. Give substantive answers, with at least 2-3 sentences per question\n\n"
        )
        optimized_prompt = f"{INTERVIEW_PROMPT_PREFIX}{combined_prompt}"
        
        # Step 4: Call the real interview API (no platform specified; defaults to both platforms simultaneously)
        try:
            # Build the batch interview list (no platform specified; dual-platform interview)
            interviews_request = []
            for agent_idx in selected_indices:
                interviews_request.append({
                    "agent_id": agent_idx,
                    "prompt": optimized_prompt  # Use the optimized prompt
                    # No platform specified; the API interviews on both twitter and reddit
                })
            
            logger.info(t("console.callingBatchInterviewApi", count=len(interviews_request)))
            
            # Call SimulationRunner's batch interview method (no platform; dual-platform interview)
            api_result = SimulationRunner.interview_agents_batch(
                simulation_id=simulation_id,
                interviews=interviews_request,
                platform=None,  # No platform specified; dual-platform interview
                timeout=180.0   # Dual-platform needs a longer timeout
            )
            
            logger.info(t("console.interviewApiReturned", count=api_result.get('interviews_count', 0), success=api_result.get('success')))
            
            # Check whether the API call succeeded
            if not api_result.get("success", False):
                error_msg = api_result.get("error", localized("未知错误", "Unknown error"))
                logger.warning(t("console.interviewApiReturnedFailure", error=error_msg))
                result.summary = localized(
                    f"采访API调用失败：{error_msg}。请检查OASIS模拟环境状态。",
                    f"Interview API call failed: {error_msg}. Please check the status of the OASIS simulation environment."
                )
                return result
            
            # Step 5: Parse the API result and build AgentInterview objects
            # Dual-platform mode return format: {"twitter_0": {...}, "reddit_0": {...}, "twitter_1": {...}, ...}
            api_data = api_result.get("result", {})
            results_dict = api_data.get("results", {}) if isinstance(api_data, dict) else {}
            
            for i, agent_idx in enumerate(selected_indices):
                agent = selected_agents[i]
                agent_name = agent.get("realname", agent.get("username", f"Agent_{agent_idx}"))
                agent_role = agent.get("profession", localized("未知", "Unknown"))
                agent_bio = agent.get("bio", "")
                
                # Get this agent's interview results on both platforms
                twitter_result = results_dict.get(f"twitter_{agent_idx}", {})
                reddit_result = results_dict.get(f"reddit_{agent_idx}", {})
                
                twitter_response = twitter_result.get("response", "")
                reddit_response = reddit_result.get("response", "")

                # Strip any tool-call JSON wrapper
                twitter_response = self._clean_tool_call_response(twitter_response)
                reddit_response = self._clean_tool_call_response(reddit_response)

                # Always output both platform markers
                twitter_text = twitter_response if twitter_response else localized("（该平台未获得回复）", "(No response from this platform)")
                reddit_text = reddit_response if reddit_response else localized("（该平台未获得回复）", "(No response from this platform)")
                response_text = localized(
                    f"【Twitter平台回答】\n{twitter_text}\n\n【Reddit平台回答】\n{reddit_text}",
                    f"[Twitter Platform Answer]\n{twitter_text}\n\n[Reddit Platform Answer]\n{reddit_text}"
                )

                # Extract key quotes (from both platforms' answers)
                key_quotes = self._extract_key_quotes(twitter_response, reddit_response)
                
                interview = AgentInterview(
                    agent_name=agent_name,
                    agent_role=agent_role,
                    agent_bio=agent_bio[:1000],  # Larger bio length limit
                    question=combined_prompt,
                    response=response_text,
                    key_quotes=key_quotes[:5]
                )
                result.interviews.append(interview)
            
            result.interviewed_count = len(result.interviews)
            
        except ValueError as e:
            # Simulation environment not running
            logger.warning(t("console.interviewApiCallFailed", error=e))
            result.summary = localized(
                f"采访失败：{str(e)}。模拟环境可能已关闭，请确保OASIS环境正在运行。",
                f"Interview failed: {str(e)}. The simulation environment may have been shut down; please make sure the OASIS environment is running."
            )
            return result
        except Exception as e:
            logger.error(t("console.interviewApiCallException", error=e))
            import traceback
            logger.error(traceback.format_exc())
            result.summary = localized(f"采访过程发生错误：{str(e)}", f"An error occurred during the interview: {str(e)}")
            return result
        
        # Step 6: Generate the interview summary
        if result.interviews:
            result.summary = self._generate_interview_summary(
                interviews=result.interviews,
                interview_requirement=interview_requirement
            )
        
        logger.info(t("console.interviewAgentsComplete", count=result.interviewed_count))
        return result
    
    @staticmethod
    def _extract_key_quotes(twitter_response: str, reddit_response: str) -> List[str]:
        """Extract key quotes from both platforms' interview answers"""
        import re
        english = get_locale() != 'zh'
        combined_responses = f"{twitter_response} {reddit_response}"

        # Clean the response text: remove markers, numbering, Markdown, and other noise
        clean_text = re.sub(r'#{1,6}\s+', '', combined_responses)
        clean_text = re.sub(r'\{[^}]*tool_name[^}]*\}', '', clean_text)
        clean_text = re.sub(r'[*_`|>~\-]{2,}', '', clean_text)
        clean_text = re.sub(r'问题\d+[：:]\s*', '', clean_text)
        if english:
            clean_text = re.sub(r'Question\s*\d+\s*[：:]\s*', '', clean_text, flags=re.IGNORECASE)
        clean_text = re.sub(r'【[^】]+】', '', clean_text)

        # Strategy 1 (primary): extract complete, substantive sentences
        sentences = re.split(localized(r'[。！？]', r'[。！？]|(?<=[.!?])\s+'), clean_text)
        meaningful = [
            s.strip() for s in sentences
            if 20 <= len(s.strip()) <= 150
            and not re.match(r'^[\s\W，,；;：:、]+', s.strip())
            and not s.strip().startswith(('{', '问题'))
            and not (english and re.match(r'Question\s*\d', s.strip(), re.IGNORECASE))
        ]
        meaningful.sort(key=len, reverse=True)
        key_quotes = [s + localized("。", "") for s in meaningful[:3]]

        # Strategy 2 (supplementary): long text inside properly paired Chinese quotation marks
        if not key_quotes:
            paired = re.findall(r'\u201c([^\u201c\u201d]{15,100})\u201d', clean_text)
            paired += re.findall(r'\u300c([^\u300c\u300d]{15,100})\u300d', clean_text)
            key_quotes = [q for q in paired if not re.match(r'^[，,；;：:、]', q)][:3]
        return key_quotes

    @staticmethod
    def _clean_tool_call_response(response: str) -> str:
        """Strip the JSON tool-call wrapper from an agent reply and extract the actual content"""
        if not response or not response.strip().startswith('{'):
            return response
        text = response.strip()
        if 'tool_name' not in text[:80]:
            return response
        import re as _re
        try:
            data = json.loads(text)
            if isinstance(data, dict) and 'arguments' in data:
                for key in ('content', 'text', 'body', 'message', 'reply'):
                    if key in data['arguments']:
                        return str(data['arguments'][key])
        except (json.JSONDecodeError, KeyError, TypeError):
            match = _re.search(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
            if match:
                return match.group(1).replace('\\n', '\n').replace('\\"', '"')
        return response

    def _load_agent_profiles(self, simulation_id: str) -> List[Dict[str, Any]]:
        """Load the simulation's agent profile files"""
        import os
        import csv
        
        # Build the profile file path
        sim_dir = os.path.join(
            os.path.dirname(__file__), 
            f'../../uploads/simulations/{simulation_id}'
        )
        
        profiles = []
        
        # Try the Reddit JSON format first
        reddit_profile_path = os.path.join(sim_dir, "reddit_profiles.json")
        if os.path.exists(reddit_profile_path):
            try:
                with open(reddit_profile_path, 'r', encoding='utf-8') as f:
                    profiles = json.load(f)
                logger.info(t("console.loadedRedditProfiles", count=len(profiles)))
                return profiles
            except Exception as e:
                logger.warning(t("console.readRedditProfilesFailed", error=e))
        
        # Try the Twitter CSV format
        twitter_profile_path = os.path.join(sim_dir, "twitter_profiles.csv")
        if os.path.exists(twitter_profile_path):
            try:
                with open(twitter_profile_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Convert CSV format to the unified format
                        profiles.append({
                            "realname": row.get("name", ""),
                            "username": row.get("username", ""),
                            "bio": row.get("description", ""),
                            "persona": row.get("user_char", ""),
                            "profession": localized("未知", "Unknown")
                        })
                logger.info(t("console.loadedTwitterProfiles", count=len(profiles)))
                return profiles
            except Exception as e:
                logger.warning(t("console.readTwitterProfilesFailed", error=e))
        
        return profiles
    
    def _select_agents_for_interview(
        self,
        profiles: List[Dict[str, Any]],
        interview_requirement: str,
        simulation_requirement: str,
        max_agents: int
    ) -> tuple:
        """
        Use the LLM to select agents to interview
        
        Returns:
            tuple: (selected_agents, selected_indices, reasoning)
                - selected_agents: Full info list of the selected agents
                - selected_indices: Index list of the selected agents (for API calls)
                - reasoning: Selection reasoning
        """
        
        # Build the agent summary list
        agent_summaries = []
        for i, profile in enumerate(profiles):
            summary = {
                "index": i,
                "name": profile.get("realname", profile.get("username", f"Agent_{i}")),
                "profession": profile.get("profession", localized("未知", "Unknown")),
                "bio": profile.get("bio", "")[:200],
                "interested_topics": profile.get("interested_topics", [])
            }
            agent_summaries.append(summary)
        
        system_prompt = localized("""你是一个专业的采访策划专家。你的任务是根据采访需求，从模拟Agent列表中选择最适合采访的对象。

选择标准：
1. Agent的身份/职业与采访主题相关
2. Agent可能持有独特或有价值的观点
3. 选择多样化的视角（如：支持方、反对方、中立方、专业人士等）
4. 优先选择与事件直接相关的角色

返回JSON格式：
{
    "selected_indices": [选中Agent的索引列表],
    "reasoning": "选择理由说明"
}""", """You are a professional interview-planning expert. Your task is to select, based on the interview requirement, the most suitable interviewees from the list of simulated agents.

Selection criteria:
1. The agent's identity/profession is relevant to the interview topic
2. The agent may hold unique or valuable viewpoints
3. Select a diverse range of perspectives (e.g. supporters, opponents, neutral parties, professionals, etc.)
4. Prioritize roles directly related to the event

Return JSON format:
{
    "selected_indices": [list of indices of the selected agents],
    "reasoning": "Explanation of the selection rationale"
}""")

        if get_locale() == 'zh':
            user_prompt = f"""采访需求：
{interview_requirement}

模拟背景：
{simulation_requirement if simulation_requirement else "未提供"}

可选择的Agent列表（共{len(agent_summaries)}个）：
{json.dumps(agent_summaries, ensure_ascii=False, indent=2)}

请选择最多{max_agents}个最适合采访的Agent，并说明选择理由。"""
        else:
            user_prompt = f"""Interview requirement:
{interview_requirement}

Simulation background:
{simulation_requirement if simulation_requirement else "Not provided"}

Available agents ({len(agent_summaries)} in total):
{json.dumps(agent_summaries, ensure_ascii=False, indent=2)}

Please select up to {max_agents} agents best suited for the interview, and explain your selection rationale."""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            
            selected_indices = response.get("selected_indices", [])[:max_agents]
            reasoning = response.get("reasoning", localized("基于相关性自动选择", "Automatically selected based on relevance"))
            
            # Get full info for the selected agents
            selected_agents = []
            valid_indices = []
            for idx in selected_indices:
                if 0 <= idx < len(profiles):
                    selected_agents.append(profiles[idx])
                    valid_indices.append(idx)
            
            return selected_agents, valid_indices, reasoning
            
        except Exception as e:
            logger.warning(t("console.llmSelectAgentFailed", error=e))
            # Fallback: select the first N
            selected = profiles[:max_agents]
            indices = list(range(min(max_agents, len(profiles))))
            return selected, indices, localized("使用默认选择策略", "Used the default selection strategy")
    
    def _generate_interview_questions(
        self,
        interview_requirement: str,
        simulation_requirement: str,
        selected_agents: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate interview questions with the LLM"""
        
        agent_roles = [a.get("profession", localized("未知", "Unknown")) for a in selected_agents]
        
        system_prompt = localized("""你是一个专业的记者/采访者。根据采访需求，生成3-5个深度采访问题。

问题要求：
1. 开放性问题，鼓励详细回答
2. 针对不同角色可能有不同答案
3. 涵盖事实、观点、感受等多个维度
4. 语言自然，像真实采访一样
5. 每个问题控制在50字以内，简洁明了
6. 直接提问，不要包含背景说明或前缀

返回JSON格式：{"questions": ["问题1", "问题2", ...]}""", """You are a professional journalist/interviewer. Based on the interview requirement, generate 3-5 in-depth interview questions.

Question requirements:
1. Open-ended questions that encourage detailed answers
2. Questions that different roles may answer differently
3. Cover multiple dimensions such as facts, opinions, and feelings
4. Natural language, like a real interview
5. Keep each question under 30 words, concise and clear
6. Ask directly; do not include background explanations or prefixes

Return JSON format: {"questions": ["Question 1", "Question 2", ...]}""")

        if get_locale() == 'zh':
            user_prompt = f"""采访需求：{interview_requirement}

模拟背景：{simulation_requirement if simulation_requirement else "未提供"}

采访对象角色：{', '.join(agent_roles)}

请生成3-5个采访问题。"""
        else:
            user_prompt = f"""Interview requirement: {interview_requirement}

Simulation background: {simulation_requirement if simulation_requirement else "Not provided"}

Interviewee roles: {', '.join(agent_roles)}

Please generate 3-5 interview questions."""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5
            )
            
            return response.get("questions", [localized(
                f"关于{interview_requirement}，您有什么看法？",
                f"What are your thoughts on {interview_requirement}?"
            )])
            
        except Exception as e:
            logger.warning(t("console.generateInterviewQuestionsFailed", error=e))
            return [
                localized(f"关于{interview_requirement}，您的观点是什么？", f"What is your view on {interview_requirement}?"),
                localized("这件事对您或您所代表的群体有什么影响？", "How does this affect you or the group you represent?"),
                localized("您认为应该如何解决或改进这个问题？", "How do you think this issue should be resolved or improved?")
            ]
    
    def _generate_interview_summary(
        self,
        interviews: List[AgentInterview],
        interview_requirement: str
    ) -> str:
        """Generate the interview summary"""
        
        if not interviews:
            return localized("未完成任何采访", "No interviews were completed")
        
        # Collect all interview content
        interview_texts = []
        for interview in interviews:
            interview_texts.append(localized(
                f"【{interview.agent_name}（{interview.agent_role}）】\n{interview.response[:500]}",
                f"[{interview.agent_name} ({interview.agent_role})]\n{interview.response[:500]}"
            ))
        
        quote_instruction = "引用受访者原话时使用中文引号「」" if get_locale() == 'zh' else 'Use quotation marks "" when quoting interviewees'
        system_prompt = localized(f"""你是一个专业的新闻编辑。请根据多位受访者的回答，生成一份采访摘要。

摘要要求：
1. 提炼各方主要观点
2. 指出观点的共识和分歧
3. 突出有价值的引言
4. 客观中立，不偏袒任何一方
5. 控制在1000字内

格式约束（必须遵守）：
- 使用纯文本段落，用空行分隔不同部分
- 不要使用Markdown标题（如#、##、###）
- 不要使用分割线（如---、***）
- {quote_instruction}
- 可以使用**加粗**标记关键词，但不要使用其他Markdown语法""", f"""You are a professional news editor. Based on the answers of multiple interviewees, produce an interview summary.

Summary requirements:
1. Distill the main viewpoints of each party
2. Point out where viewpoints agree and diverge
3. Highlight valuable quotes
4. Be objective and neutral, without favoring any side
5. Keep it within 500 words

Formatting constraints (must be followed):
- Use plain-text paragraphs, separating different sections with blank lines
- Do not use Markdown headings (such as #, ##, ###)
- Do not use horizontal rules (such as ---, ***)
- {quote_instruction}
- You may use **bold** to mark key terms, but do not use any other Markdown syntax""")

        user_prompt = localized(f"""采访主题：{interview_requirement}

采访内容：
{"".join(interview_texts)}

请生成采访摘要。""", f"""Interview topic: {interview_requirement}

Interview content:
{"".join(interview_texts)}

Please generate the interview summary.""")

        try:
            summary = self.llm.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            return summary
            
        except Exception as e:
            logger.warning(t("console.generateInterviewSummaryFailed", error=e))
            # Fallback: simple concatenation
            return localized(
                f"共采访了{len(interviews)}位受访者，包括：",
                f"Interviewed {len(interviews)} interviewees in total, including: "
            ) + localized("、", ", ").join([i.agent_name for i in interviews])
