"""
Logging configuration for interview agent.
Saves logs to project logs/ folder with rotation.
"""

import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
from logging.handlers import RotatingFileHandler


# Module-level logger cache
_loggers: dict = {}


def get_project_root() -> Path:
    """Get project root directory."""
    # Assume this file is in utils/, so parent.parent is project root
    return Path(__file__).parent.parent


def get_logs_dir() -> Path:
    """
    Get or create logs directory.

    Uses /tmp/logs/ on HuggingFace Spaces (read-only filesystem),
    otherwise uses project_root/logs for local development.

    Returns:
        Path: Logs directory path
    """
    # Detect HuggingFace Spaces environment
    if os.getenv("SPACE_ID"):
        # HF Spaces: use /tmp (only writable location)
        logs_dir = Path("/tmp/logs")
    else:
        # Local development: use project logs directory
        logs_dir = get_project_root() / "logs"

    # Create directory with fallback
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        # Fallback to /tmp if directory not writable
        logs_dir = Path("/tmp/logs")
        logs_dir.mkdir(parents=True, exist_ok=True)

    return logs_dir


def setup_logger(
    name: str,
    level: Optional[int] = None,
    log_to_console: bool = True,
    log_to_file: bool = True,
    session_id: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with console and file handlers.

    Args:
        name: Logger name (typically module name)
        level: Logging level (defaults to INFO or INTERVIEW_LOG_LEVEL env var)
        log_to_console: Whether to output to console
        log_to_file: Whether to output to file
        session_id: Optional session ID for session-specific log file

    Returns:
        Configured logger
    """
    # Resolve logging level from environment variable if not provided
    if level is None:
        import os
        env_level = os.getenv('INTERVIEW_LOG_LEVEL', 'INFO').upper()
        level = getattr(logging, env_level, logging.INFO)
    # Return cached logger if exists
    cache_key = f"{name}_{session_id}" if session_id else name
    if cache_key in _loggers:
        return _loggers[cache_key]
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Format
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        logs_dir = get_logs_dir()
        
        # Main log file (rotating)
        main_log = logs_dir / "interview_agent.log"
        file_handler = RotatingFileHandler(
            main_log,
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=5
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Session-specific log if session_id provided
        if session_id:
            session_log = logs_dir / f"session_{session_id}.log"
            session_handler = logging.FileHandler(session_log)
            session_handler.setLevel(level)
            session_handler.setFormatter(formatter)
            logger.addHandler(session_handler)
    
    _loggers[cache_key] = logger
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger or create a basic one.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    if name in _loggers:
        return _loggers[name]
    return setup_logger(name)


class InterviewLogger:
    """
    Structured logger for interview sessions.
    Provides convenience methods for common log patterns.
    """
    
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = setup_logger(
            f"interview.{self.session_id}",
            session_id=self.session_id
        )
        self.turn_count = 0
    
    def session_start(self, concept_name: str, schema_name: str) -> None:
        """Log session start."""
        self.logger.info("=" * 60)
        self.logger.info(f"SESSION START: {self.session_id}")
        self.logger.info(f"Concept: {concept_name}")
        self.logger.info(f"Schema: {schema_name}")
        self.logger.info("=" * 60)
    
    def session_end(self, summary: dict) -> None:
        """Log session end with summary."""
        self.logger.info("=" * 60)
        self.logger.info(f"SESSION END: {self.session_id}")
        self.logger.info(f"Total turns: {summary.get('turns', 0)}")
        self.logger.info(f"Nodes: {summary.get('graph', {}).get('nodes', 0)}")
        self.logger.info(f"Edges: {summary.get('graph', {}).get('edges', 0)}")
        self.logger.info(f"Coverage satisfied: {summary.get('coverage', {}).get('satisfied', False)}")
        self.logger.info(f"Completion reason: {summary.get('completion_reason', 'unknown')}")
        self.logger.info("=" * 60)
    
    def turn_start(self, turn_number: int) -> None:
        """Log turn start."""
        self.turn_count = turn_number
        self.logger.info(f"--- Turn {turn_number} ---")
    
    def question_generated(self, question: str, strategy: str) -> None:
        """Log generated question."""
        self.logger.info(f"[Q{self.turn_count}] Strategy: {strategy}")
        self.logger.info(f"[Q{self.turn_count}] Question: {question}")
    
    def response_received(self, response: str) -> None:
        """Log received response."""
        # Truncate long responses for log readability
        display = response[:200] + "..." if len(response) > 200 else response
        self.logger.info(f"[A{self.turn_count}] Response: {display}")
    
    def extraction_result(
        self, 
        nodes_added: int, 
        edges_added: int,
        is_extractable: bool
    ) -> None:
        """Log extraction result."""
        if is_extractable:
            self.logger.info(f"[Extract] +{nodes_added} nodes, +{edges_added} edges")
        else:
            self.logger.info("[Extract] Not extractable - clarification needed")
    
    def momentum_assessed(self, level: str, indicators: list) -> None:
        """Log momentum assessment."""
        self.logger.info(f"[Momentum] {level}: {indicators}")
    
    def strategy_selected(self, strategy_id: str, focus: str) -> None:
        """Log strategy selection."""
        self.logger.info(f"[Strategy] {strategy_id} -> {focus}")
    
    def coverage_update(self, gaps_remaining: int, satisfied: bool) -> None:
        """Log coverage state."""
        status = "SATISFIED" if satisfied else f"{gaps_remaining} gaps remaining"
        self.logger.info(f"[Coverage] {status}")
    
    def graph_state(self, isolated: int, ambiguous: int, terminals: int) -> None:
        """Log graph state."""
        self.logger.info(
            f"[Graph] isolated={isolated}, ambiguous={ambiguous}, terminals={terminals}"
        )
    
    def llm_call(self, task: str, provider: str, latency_ms: int, success: bool) -> None:
        """Log LLM call."""
        status = "OK" if success else "FAILED"
        self.logger.debug(f"[LLM] {task} via {provider}: {latency_ms}ms ({status})")
    
    def error(self, message: str, exc_info: bool = False) -> None:
        """Log error."""
        self.logger.error(message, exc_info=exc_info)
    
    def warning(self, message: str) -> None:
        """Log warning."""
        self.logger.warning(message)
    
    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)
    
    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)

    # Extended logging methods for post-interview analysis

    def strategy_reasoning(
        self,
        strategy_id: str,
        intent: str,
        focus_description: str,
        alternatives_considered: Optional[list] = None
    ) -> None:
        """Log why a strategy was selected."""
        self.logger.info(f"[Strategy Reasoning] Selected '{strategy_id}'")
        self.logger.info(f"  Intent: {intent}")
        if alternatives_considered:
            self.logger.info(f"  Alternatives considered: {alternatives_considered}")
        self.logger.info(f"  Focus: {focus_description}")

    def graph_evolution(
        self,
        turn: int,
        nodes_total: int,
        edges_total: int,
        nodes_added: int,
        edges_added: int,
        isolation_ratio: float,
        coverage_gaps: int
    ) -> None:
        """Log graph metrics at each turn."""
        self.logger.info(f"[Graph Evolution] Turn {turn}")
        self.logger.info(f"  Nodes: {nodes_total} (+{nodes_added}), Edges: {edges_total} (+{edges_added})")
        self.logger.info(f"  Isolation ratio: {isolation_ratio:.1%}, Coverage gaps: {coverage_gaps}")

    def extraction_details(
        self,
        nodes: list,
        edges: list,
        reactions: dict,
        element_mappings: dict
    ) -> None:
        """Log detailed extraction results."""
        self.logger.info(f"[Extraction Details]")
        if nodes:
            self.logger.info(f"  Nodes: {nodes}")
        if edges:
            edge_strs = [f"{s} --{r}--> {t}" for s, t, r in edges]
            self.logger.info(f"  Edges: {edge_strs}")
        if reactions:
            self.logger.info(f"  Reactions: {reactions}")
        if element_mappings:
            self.logger.info(f"  Element mappings: {element_mappings}")

    def session_summary_extended(
        self,
        total_turns: int,
        strategy_counts: dict,
        coverage_evolution: list,
        graph_growth: list,
        final_isolation_ratio: float,
        completion_reason: str
    ) -> None:
        """Log extended session summary at end."""
        self.logger.info("=" * 60)
        self.logger.info("[SESSION SUMMARY - EXTENDED]")
        self.logger.info(f"  Total turns: {total_turns}")
        self.logger.info(f"  Completion reason: {completion_reason}")
        self.logger.info("")
        self.logger.info("[Strategy Distribution]")
        for strategy, count in sorted(strategy_counts.items(), key=lambda x: -x[1]):
            pct = count / total_turns * 100 if total_turns > 0 else 0
            self.logger.info(f"  {strategy}: {count} ({pct:.0f}%)")
        self.logger.info("")
        self.logger.info("[Coverage Evolution]")
        self.logger.info(f"  Gaps per turn: {coverage_evolution}")
        self.logger.info("")
        self.logger.info("[Graph Growth]")
        if graph_growth:
            self.logger.info(f"  Final: {graph_growth[-1][0]} nodes, {graph_growth[-1][1]} edges")
            self.logger.info(f"  Growth by turn: {graph_growth}")
        self.logger.info(f"  Isolation ratio: {final_isolation_ratio:.1%}")
        self.logger.info("=" * 60)
