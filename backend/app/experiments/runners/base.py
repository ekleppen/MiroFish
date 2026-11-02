"""
Experiment engine interface

Every engine reads guests from the experiment snapshot and emits rows in the
shared ResultRow schema, so population management, coding and summaries don't
depend on how the guests were simulated.
"""

from typing import Any, Callable, Dict, List, Optional, Protocol, Set

from ..schema import ENGINE_DIRECT, ENGINE_OASIS_SOCIAL, ExperimentSpec

ResultCallback = Callable[[Dict[str, Any]], None]
# (stage, current, total); engines name their own stages.
ProgressCallback = Callable[[str, int, int], None]


class EngineNotImplemented(NotImplementedError):
    """A known engine name that has no runner yet."""


class ExperimentRunner(Protocol):
    engine: str
    prompt_version: str
    reasoning_effort: str
    model_name: str

    def run(
        self,
        spec: ExperimentSpec,
        guests: List[Dict[str, Any]],
        on_result: ResultCallback,
        progress_callback: Optional[ProgressCallback] = None,
        completed_keys: Optional[Set[tuple]] = None,
        parallel: int = 5,
        existing_propensities: Optional[List[Dict[str, Any]]] = None,
        on_propensity: Optional[ResultCallback] = None,
        calibrator: Optional[Callable[[List[Dict[str, Any]]], Optional[float]]] = None,
    ) -> None:
        ...


def get_runner(engine: str, llm_client: Any = None,
               reasoning_effort: Optional[str] = None) -> ExperimentRunner:
    if engine == ENGINE_DIRECT:
        from .direct import REASONING_EFFORT, DirectMessageRunner
        return DirectMessageRunner(llm_client=llm_client,
                                   reasoning_effort=reasoning_effort or REASONING_EFFORT)
    if engine == ENGINE_OASIS_SOCIAL:
        raise EngineNotImplemented(
            "The oasis_social engine is planned as a separate experiment mode and isn't available yet")
    raise ValueError(f"Unknown engine: {engine}")
