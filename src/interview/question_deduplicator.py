"""
Question Deduplicator for detecting repetitive questions.

Detects repetitive questions using multiple similarity metrics:
- Word overlap (Jaccard similarity)
- Semantic similarity (heuristic-based: intent + focus)
"""

import logging
import re

logger = logging.getLogger(__name__)


class QuestionDeduplicator:
    """Detect and prevent repetitive questions."""

    # Stopwords to exclude from word overlap calculation
    STOPWORDS = {
        "a", "an", "the", "is", "are", "was", "were", "can", "could",
        "you", "me", "tell", "more", "about", "what", "how", "why",
        "when", "where", "do", "does", "did", "would", "should",
        "have", "has", "had", "will", "shall", "may", "might",
    }

    def __init__(
        self,
        word_overlap_threshold: float = 0.6,
        semantic_similarity_threshold: float = 0.75,
        history_window: int = 5,
    ):
        """
        Initialize question deduplicator.

        Args:
            word_overlap_threshold: Jaccard similarity threshold (0-1)
            semantic_similarity_threshold: Semantic similarity threshold (0-1)
            history_window: Number of recent questions to check
        """
        self.word_overlap_threshold = word_overlap_threshold
        self.semantic_similarity_threshold = semantic_similarity_threshold
        self.history_window = history_window

    def is_repetitive(
        self, new_question: str, question_history: list[str]
    ) -> tuple[bool, str, float]:
        """
        Check if question is too similar to recent questions.

        Args:
            new_question: Question to check
            question_history: List of previous questions

        Returns:
            Tuple of (is_repetitive, reason, similarity_score)
        """
        if not question_history:
            return False, "not_repetitive", 0.0

        # Check last N questions
        recent_questions = question_history[-self.history_window :]

        for past_q in recent_questions:
            # Check word overlap
            word_sim = self._calculate_word_overlap(new_question, past_q)
            if word_sim >= self.word_overlap_threshold:
                logger.warning(
                    f"Word overlap detected: {word_sim:.2f} >= {self.word_overlap_threshold}"
                )
                return True, "word_overlap", word_sim

            # Check semantic similarity
            semantic_sim = self._calculate_semantic_similarity(new_question, past_q)
            if semantic_sim >= self.semantic_similarity_threshold:
                logger.warning(
                    f"Semantic similarity detected: {semantic_sim:.2f} >= "
                    f"{self.semantic_similarity_threshold}"
                )
                return True, "semantic_similarity", semantic_sim

        return False, "not_repetitive", 0.0

    def _calculate_word_overlap(self, q1: str, q2: str) -> float:
        """
        Calculate Jaccard similarity of normalized words.

        Args:
            q1: First question
            q2: Second question

        Returns:
            Jaccard similarity score (0-1)
        """
        words1 = self._normalize_question(q1)
        words2 = self._normalize_question(q2)

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _normalize_question(self, question: str) -> set[str]:
        """
        Normalize question for comparison.

        Process:
        1. Lowercase
        2. Remove punctuation
        3. Remove stopwords
        4. Return set of words

        Args:
            question: Question text

        Returns:
            Set of normalized words
        """
        # Remove punctuation
        text = re.sub(r"[^\w\s]", "", question.lower())

        # Split and filter stopwords
        words = [w for w in text.split() if w not in self.STOPWORDS]

        return set(words)

    def _calculate_semantic_similarity(self, q1: str, q2: str) -> float:
        """
        Calculate semantic similarity using heuristic approach.

        Heuristic combines:
        1. Question intent (dig_deeper, connect, introduce)
        2. Focus concept (main topic)

        Same intent + same focus = highly similar
        Same focus, different intent = moderately similar
        Different focus = not similar

        Args:
            q1: First question
            q2: Second question

        Returns:
            Semantic similarity score (0-1)
        """
        intent1 = self._extract_intent(q1)
        intent2 = self._extract_intent(q2)

        focus1 = self._extract_focus_concept(q1)
        focus2 = self._extract_focus_concept(q2)

        # Same intent + same focus = highly similar
        if intent1 == intent2 and focus1 == focus2:
            return 0.9

        # Same focus, different intent = moderately similar
        if focus1 == focus2 and focus1 != "unknown":
            return 0.6

        # Same intent, different focus = slightly similar
        if intent1 == intent2 and intent1 != "other":
            return 0.3

        # Different focus = not similar
        return 0.2

    def _extract_intent(self, question: str) -> str:
        """
        Classify question intent.

        Categories:
        - dig_deeper: "tell me more", "elaborate", "explain"
        - connect: "relate", "connection", "how does X affect Y"
        - introduce: "thoughts on", "what about"
        - other: default

        Args:
            question: Question text

        Returns:
            Intent category
        """
        q_lower = question.lower()

        # Dig deeper patterns
        if any(
            phrase in q_lower
            for phrase in [
                "tell me more",
                "elaborate",
                "explain",
                "say more",
                "describe",
                "walk me through",
            ]
        ):
            return "dig_deeper"

        # Connect patterns
        elif any(
            phrase in q_lower
            for phrase in [
                "relate",
                "connection",
                "affect",
                "lead to",
                "influence",
                "connect",
                "link",
            ]
        ):
            return "connect"

        # Introduce patterns
        elif any(
            phrase in q_lower
            for phrase in ["thoughts on", "what about", "how about", "feel about"]
        ):
            return "introduce"

        else:
            return "other"

    def _extract_focus_concept(self, question: str) -> str:
        """
        Extract main concept being asked about.

        Strategy:
        1. Look for quoted terms (highest priority)
        2. Extract nouns after prepositions ("about X", "on X")
        3. Extract capitalized words
        4. Return "unknown" if nothing found

        Args:
            question: Question text

        Returns:
            Focus concept (normalized)
        """
        # Look for quoted terms
        quoted = re.findall(r'["\']([^"\']+)["\']', question)
        if quoted:
            return quoted[0].lower()

        # Look for specific patterns: "about X", "on X", "of X", "with X"
        patterns = [
            r"about\s+(\w+)",
            r"on\s+(\w+)",
            r"of\s+(\w+)",
            r"with\s+(\w+)",
            r"regarding\s+(\w+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, question.lower())
            if match:
                word = match.group(1)
                # Filter out stopwords
                if word not in self.STOPWORDS:
                    return word

        # Look for capitalized words (might be important concepts)
        capitalized = re.findall(r"\b[A-Z][a-z]+\b", question)
        if capitalized:
            return capitalized[0].lower()

        return "unknown"
