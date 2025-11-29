"""
Extended report generator for interview sessions.

Generates detailed Markdown reports with turn-by-turn breakdown,
richness scoring details, and LLM metadata.
"""

import logging
from datetime import datetime

from src.core.data_models import TurnLog

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate extended Markdown reports for interview sessions."""

    @staticmethod
    def generate_markdown_report(
        session_id: str,
        concept_description: str,
        turn_logs: list[TurnLog],
        final_graph_stats: dict,
    ) -> str:
        """
        Generate comprehensive Markdown report.

        Args:
            session_id: Session identifier
            concept_description: Concept being interviewed about
            turn_logs: List of turn logs from interview
            final_graph_stats: Final graph statistics

        Returns:
            Markdown formatted report string
        """
        lines = []

        # Header
        lines.extend(
            [
                "# Extended Interview Report",
                "",
                f"**Session ID:** `{session_id}`",
                f"**Concept:** {concept_description}",
                f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"**Total Turns:** {len(turn_logs)}",
                "",
                "---",
                "",
            ]
        )

        # Summary Statistics
        lines.extend(
            [
                "## Summary Statistics",
                "",
                f"- **Total Nodes Extracted:** {final_graph_stats.get('nodes', 0)}",
                f"- **Total Edges Extracted:** {final_graph_stats.get('edges', 0)}",
                f"- **Final Richness Score:** {final_graph_stats.get('richness', 0.0):.2f}",
                f"- **Schema Coverage:** {final_graph_stats.get('coverage', '0%')}",
                "",
                "---",
                "",
            ]
        )

        # Turn-by-turn breakdown
        lines.extend(
            [
                "## Turn-by-Turn Analysis",
                "",
            ]
        )

        for turn_log in turn_logs:
            lines.extend(ReportGenerator._format_turn(turn_log))
            lines.append("")

        # Metadata summary
        lines.extend(
            [
                "---",
                "",
                "## LLM Usage Summary",
                "",
            ]
        )

        total_tokens = sum(
            log.graph_delta.extraction_metadata.get("tokens_used", 0) for log in turn_logs
        )
        avg_latency = (
            sum(log.graph_delta.extraction_metadata.get("latency_ms", 0) for log in turn_logs)
            / len(turn_logs)
            if turn_logs
            else 0
        )

        lines.extend(
            [
                f"- **Total Tokens Used:** {total_tokens:,}",
                f"- **Average Latency:** {avg_latency:.0f}ms",
                f"- **Extraction Model:** {turn_logs[0].graph_delta.extraction_metadata.get('model_used', 'Unknown') if turn_logs else 'N/A'}",
                "",
            ]
        )

        return "\n".join(lines)

    @staticmethod
    def _format_turn(turn_log: TurnLog) -> list[str]:
        """Format a single turn log as Markdown."""
        lines = []

        # Turn header
        lines.extend(
            [
                f"### Turn {turn_log.turn_number}",
                "",
                f"**Timestamp:** {turn_log.timestamp.isoformat()}",
                "",
            ]
        )

        # Q&A
        lines.extend(
            [
                "**Participant Response:**",
                f"> {turn_log.participant_response}",
                "",
                "**Interviewer Question:**",
                f"> {turn_log.question_generated}",
                "",
            ]
        )

        # Extraction details
        lines.extend(
            [
                "#### Extraction Results",
                "",
            ]
        )

        # Nodes extracted
        nodes_added = turn_log.graph_delta.nodes_added
        if nodes_added:
            lines.extend(
                [
                    "**Nodes Added:**",
                    "",
                ]
            )
            for node in nodes_added:
                lines.append(f"- **{node.label}** ({node.type})")
                if node.source_quotes:
                    lines.append(f'  - Quote: "{node.source_quotes[0]}"')
            lines.append("")
        else:
            lines.extend(["*No nodes extracted this turn*", ""])

        # Edges extracted
        edges_added = turn_log.graph_delta.edges_added
        if edges_added:
            lines.extend(
                [
                    "**Edges Added:**",
                    "",
                ]
            )
            for edge in edges_added:
                lines.append(f"- {edge.source} → {edge.target} ({edge.type})")
                if edge.source_quote:
                    lines.append(f'  - Quote: "{edge.source_quote[:50]}..."')
            lines.append("")
        else:
            lines.extend(["*No edges extracted this turn*", ""])

        # Richness score breakdown
        lines.extend(
            [
                "#### Richness Score Breakdown",
                "",
                f"**Score Increase This Turn:** +{turn_log.graph_delta.richness_score:.2f}",
                "",
            ]
        )

        if nodes_added:
            lines.append("**Contribution by Node:**")
            lines.append("")
            for node in nodes_added:
                lines.append(f"- {node.label} ({node.type})")
            lines.append("")

        # Cumulative richness - handle None interview_state
        cumulative_richness = 0.0
        if turn_log.interview_state and hasattr(turn_log.interview_state, "cumulative_richness"):
            cumulative_richness = turn_log.interview_state.cumulative_richness

        lines.extend(
            [
                f"**Cumulative Richness:** {cumulative_richness:.2f}",
                "",
            ]
        )

        # LLM metadata
        metadata = turn_log.graph_delta.extraction_metadata
        lines.extend(
            [
                "#### LLM Metadata",
                "",
                f"- Model: `{metadata.get('model_used', 'Unknown')}`",
                f"- Tokens: {metadata.get('tokens_used', 0)}",
                f"- Latency: {metadata.get('latency_ms', 0)}ms",
            ]
        )

        # Validation issues
        if turn_log.errors:
            lines.extend(
                [
                    "",
                    "**Validation Errors:**",
                    "",
                ]
            )
            for error in turn_log.errors:
                lines.append(f"- ⚠️ {error}")

        if turn_log.warnings:
            lines.extend(
                [
                    "",
                    "**Validation Warnings:**",
                    "",
                ]
            )
            for warning in turn_log.warnings:
                lines.append(f"- ⚡ {warning}")

        lines.append("")
        lines.append("---")

        return lines
