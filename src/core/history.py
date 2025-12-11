"""
Conversation tracking.
Immutable record of interview turns.
"""

from typing import List, Tuple, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class Turn(BaseModel):
    """Single turn in the interview conversation."""
    
    turn_number: int = Field(description="Sequential turn number (1-indexed)")
    question: str = Field(description="Question asked by interviewer")
    response: str = Field(description="Respondent's answer")
    extracted_nodes: List[str] = Field(
        default_factory=list,
        description="Node IDs added this turn"
    )
    extracted_edges: List[Tuple[str, str]] = Field(
        default_factory=list,
        description="Edge pairs (source_id, target_id) added this turn"
    )
    strategy_used: str = Field(
        default="",
        description="Strategy ID that generated this question"
    )
    tactic_used: Optional[str] = Field(
        default=None,
        description="Tactic ID if reported by LLM"
    )
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict = Field(
        default_factory=dict,
        description="Additional turn data (momentum, focus node, etc.)"
    )


class History(BaseModel):
    """
    Complete interview history.
    Provides conversation context for LLM prompts.
    """
    
    turns: List[Turn] = Field(default_factory=list)
    
    def add_turn(self, turn: Turn) -> None:
        """Add a turn to history."""
        self.turns.append(turn)
    
    def get_recent(self, n: int) -> List[Turn]:
        """Get the n most recent turns."""
        return self.turns[-n:] if n > 0 else []

    def get_recent_questions(self, n: int = 6) -> List[str]:
        """
        Get n most recent questions for deduplication.

        Args:
            n: Number of recent questions to return

        Returns:
            List of question strings
        """
        return [t.question for t in self.turns[-n:]] if n > 0 else []

    def get_turn(self, turn_number: int) -> Optional[Turn]:
        """Get a specific turn by number (1-indexed)."""
        if 1 <= turn_number <= len(self.turns):
            return self.turns[turn_number - 1]
        return None
    
    @property
    def current_turn_number(self) -> int:
        """Get the current (last) turn number."""
        return len(self.turns)
    
    def format_for_prompt(self, n: int = 5) -> str:
        """
        Format recent conversation for LLM context.
        
        Args:
            n: Number of recent turns to include
            
        Returns:
            Formatted conversation string
        """
        recent = self.get_recent(n)
        
        if not recent:
            return "No conversation history yet."
        
        lines = []
        for turn in recent:
            lines.append(f"Q{turn.turn_number}: {turn.question}")
            lines.append(f"A{turn.turn_number}: {turn.response}")
            lines.append("")
        
        return "\n".join(lines).strip()
    
    def format_full_transcript(self) -> str:
        """
        Format complete conversation transcript.
        
        Returns:
            Full conversation as formatted string
        """
        return self.format_for_prompt(n=len(self.turns))
    
    def get_strategies_used(self) -> List[str]:
        """Get list of all strategies used in order."""
        return [turn.strategy_used for turn in self.turns if turn.strategy_used]
    
    def get_strategy_counts(self) -> dict:
        """Get count of each strategy used."""
        counts: dict = {}
        for strategy in self.get_strategies_used():
            counts[strategy] = counts.get(strategy, 0) + 1
        return counts
    
    def turns_since_strategy(self, strategy_id: str) -> int:
        """
        Count turns since a strategy was last used.
        
        Args:
            strategy_id: Strategy to check
            
        Returns:
            Number of turns since last use, or -1 if never used
        """
        for i, turn in enumerate(reversed(self.turns)):
            if turn.strategy_used == strategy_id:
                return i
        return -1
    
    def get_last_response(self) -> Optional[str]:
        """Get the most recent respondent response."""
        if self.turns:
            return self.turns[-1].response
        return None
    
    def get_all_extracted_nodes(self) -> List[str]:
        """Get all node IDs extracted across all turns."""
        nodes = []
        for turn in self.turns:
            nodes.extend(turn.extracted_nodes)
        return nodes
    
    def get_nodes_from_turn(self, turn_number: int) -> List[str]:
        """Get node IDs extracted in a specific turn."""
        turn = self.get_turn(turn_number)
        if turn:
            return turn.extracted_nodes
        return []
    
    def get_turn_for_node(self, node_id: str) -> Optional[int]:
        """Find which turn a node was extracted in."""
        for turn in self.turns:
            if node_id in turn.extracted_nodes:
                return turn.turn_number
        return None
    
    def to_dict(self) -> dict:
        """Export history as dictionary."""
        return {
            "turns": [turn.model_dump() for turn in self.turns],
            "total_turns": len(self.turns),
            "strategies_used": self.get_strategy_counts()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "History":
        """Reconstruct history from dictionary."""
        history = cls()
        for turn_data in data.get("turns", []):
            history.add_turn(Turn(**turn_data))
        return history
