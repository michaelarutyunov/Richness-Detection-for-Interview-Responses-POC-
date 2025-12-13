"""
LLM-based extraction of nodes and edges from respondent text.
Includes extractability check and momentum assessment.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Tuple, Any, Set
from pydantic import BaseModel, Field

from core.graph import Graph, Node, Edge
from core.schema import Schema
from core.history import History
from core.state import CoverageState, Momentum
from utils.llm_manager import LLMManager, TaskType

logger = logging.getLogger(__name__)


# Function calling schema for structured extraction
# Using this ensures element_mapping is always provided
EXTRACTION_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_graph_delta",
        "description": "Extract nodes and edges from interview response with element mappings for coverage tracking",
        "parameters": {
            "type": "object",
            "required": ["nodes", "edges"],
            "properties": {
                "nodes": {
                    "type": "array",
                    "description": "List of extracted concept nodes",
                    "items": {
                        "type": "object",
                        "required": ["label", "node_type", "quote"],
                        "properties": {
                            "label": {
                                "type": "string",
                                "description": "Concise label for the concept"
                            },
                            "node_type": {
                                "type": "string",
                                "description": "Schema node type (e.g., attribute, consequence, value)"
                            },
                            "quote": {
                                "type": "string",
                                "description": "Exact quote from response supporting this node"
                            },
                            "is_ambiguous": {
                                "type": "boolean",
                                "description": "Whether this concept needs clarification",
                                "default": False
                            },
                            "element_mapping": {
                                "type": ["string", "null"],
                                "description": "Reference element ID this node relates to (e.g., 'insight', 'promise', 'rtb') or null if none"
                            },
                            "reaction": {
                                "type": ["string", "null"],
                                "enum": ["positive", "negative", "neutral", "skeptical", "curious", None],
                                "description": "Respondent's evaluative stance toward this concept, if expressed"
                            }
                        }
                    }
                },
                "edges": {
                    "type": "array",
                    "description": "List of relationships between nodes",
                    "items": {
                        "type": "object",
                        "required": ["source_label", "target_label", "relation_type", "quote"],
                        "properties": {
                            "source_label": {
                                "type": "string",
                                "description": "Label of source node"
                            },
                            "target_label": {
                                "type": "string",
                                "description": "Label of target node"
                            },
                            "relation_type": {
                                "type": "string",
                                "description": "Type of relationship (e.g., leads_to, enables, causes)"
                            },
                            "quote": {
                                "type": "string",
                                "description": "Quote supporting this relationship"
                            }
                        }
                    }
                }
            }
        }
    }
}


class ExtractedNodeData(BaseModel):
    """Raw extracted node data from LLM."""
    label: str = Field(description="Concept label")
    node_type: str = Field(description="Schema node type")
    quote: str = Field(default="", description="Supporting quote from response")
    is_ambiguous: bool = Field(default=False)


class ExtractedEdgeData(BaseModel):
    """Raw extracted edge data from LLM."""
    source_label: str = Field(description="Source node label")
    target_label: str = Field(description="Target node label")
    relation_type: str = Field(description="Edge type")
    quote: str = Field(default="", description="Supporting quote")


class ExtractionResult(BaseModel):
    """Result of extraction process."""
    nodes: List[Node] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)
    node_element_mappings: Dict[str, str] = Field(
        default_factory=dict,
        description="node_id -> element_id for coverage tracking"
    )
    element_reactions: Dict[str, str] = Field(
        default_factory=dict,
        description="element_id -> reaction for coverage tracking"
    )
    is_extractable: bool = Field(default=True)
    ambiguity_reason: Optional[str] = Field(default=None)
    raw_response: str = Field(default="", description="Raw LLM response for debugging")
    llm_response: Optional[Any] = Field(default=None, description="LLMResponse object for token tracking")


class Extractor:
    """
    Extracts graph structure from respondent text.
    Uses LLM for natural language understanding.
    """

    def __init__(
        self,
        schema: Schema,
        coverage_state: CoverageState,
        llm_manager: LLMManager,
        config: Optional[Dict] = None
    ):
        self.schema = schema
        self.coverage_state = coverage_state
        self.llm = llm_manager

        # Semantic deduplication configuration
        self.config = config or {}
        dedup_config = self.config.get("extraction", {}).get("semantic_deduplication", {})

        self.dedup_method = dedup_config.get("method", "hybrid")
        self.jaccard_threshold = dedup_config.get("jaccard_threshold", 0.75)
        self.embeddings_enabled = dedup_config.get("embeddings_enabled", True)
        self.embeddings_threshold = dedup_config.get("embeddings_threshold", 0.80)

        # Embedding model and cache (lazy loaded)
        self._embedding_model = None
        self._embedding_cache: Dict[str, Any] = {}

        # Load embedding model eagerly if enabled
        if self.embeddings_enabled:
            self._load_embedding_model()

        logger.info(
            f"[Extraction] Semantic deduplication initialized: method={self.dedup_method}, "
            f"jaccard_threshold={self.jaccard_threshold}, embeddings_enabled={self.embeddings_enabled}, "
            f"embeddings_threshold={self.embeddings_threshold}"
        )

    # =========================================================================
    # SEMANTIC DEDUPLICATION METHODS (Phase 2A + 2B)
    # =========================================================================

    def _lemmatize_phrase(self, phrase: str) -> Set[str]:
        """
        Lemmatize a phrase by removing common suffixes and expanding synonyms.

        Phase 2A: Enhanced Jaccard similarity component.

        Args:
            phrase: Input phrase (e.g., "proper foam", "froths well")

        Returns:
            Set of lemmatized words + synonyms
        """
        # Synonym pairs (bidirectional)
        SYNONYMS = {
            "foam": {"foam", "froth"},
            "froth": {"foam", "froth"},
            "thick": {"thick", "heavy"},
            "heavy": {"thick", "heavy"},
            "thin": {"thin", "watery"},
            "watery": {"thin", "watery"},
            "creamy": {"creamy", "smooth"},
            "smooth": {"creamy", "smooth"},
        }

        # Suffix removal patterns (ordered by length to avoid greedy matches)
        SUFFIXES = ["ing", "est", "ed", "er", "ly", "s"]

        # Words that should NOT be lemmatized (common false positives)
        STOP_LEMMATIZE = {"proper", "after", "under", "over", "super", "inner", "outer"}

        words = phrase.lower().split()
        lemmatized = set()

        for word in words:
            # Skip lemmatization for certain words
            if word in STOP_LEMMATIZE:
                base_word = word
            else:
                base_word = word

                # Try removing suffixes (only if results in reasonable stem)
                for suffix in SUFFIXES:
                    if word.endswith(suffix) and len(word) > len(suffix) + 2:
                        potential_base = word[: -len(suffix)]
                        # Only use if not in stop list
                        if potential_base not in STOP_LEMMATIZE:
                            base_word = potential_base
                            break

            # Add base word
            lemmatized.add(base_word)

            # Add synonyms if present
            if base_word in SYNONYMS:
                lemmatized.update(SYNONYMS[base_word])

        return lemmatized

    def _jaccard_similarity_with_lemmas(self, label1: str, label2: str) -> float:
        """
        Compute Jaccard similarity between two labels using lemmatization.

        Phase 2A: Enhanced Jaccard similarity.

        Args:
            label1: First label
            label2: Second label

        Returns:
            Similarity score (0-1)
        """
        lemmas1 = self._lemmatize_phrase(label1)
        lemmas2 = self._lemmatize_phrase(label2)

        if not lemmas1 or not lemmas2:
            return 0.0

        intersection = lemmas1 & lemmas2
        union = lemmas1 | lemmas2

        similarity = len(intersection) / len(union) if union else 0.0

        logger.debug(
            f"[Jaccard] '{label1}' vs '{label2}': "
            f"lemmas1={lemmas1}, lemmas2={lemmas2}, "
            f"intersection={intersection}, union={union}, "
            f"similarity={similarity:.3f}"
        )

        return similarity

    def _load_embedding_model(self) -> None:
        """
        Load sentence-transformers model for semantic similarity.

        Phase 2B: Semantic embeddings.
        Loads 'all-MiniLM-L6-v2' model (~100MB, fast inference).
        """
        if self._embedding_model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer

            logger.info("[Extraction] Loading sentence-transformers model 'all-MiniLM-L6-v2'...")
            self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("[Extraction] Embedding model loaded successfully")
        except ImportError:
            logger.warning(
                "[Extraction] sentence-transformers not installed. "
                "Semantic embeddings disabled. Install with: pip install sentence-transformers"
            )
            self.embeddings_enabled = False
        except Exception as e:
            logger.error(f"[Extraction] Failed to load embedding model: {e}")
            self.embeddings_enabled = False

    def _get_cached_embedding(self, label: str) -> Optional[Any]:
        """
        Get or compute embedding for a label with caching.

        Phase 2B: Embeddings with cache optimization.

        Args:
            label: Node label

        Returns:
            Embedding vector (numpy array) or None if model unavailable
        """
        if not self.embeddings_enabled or self._embedding_model is None:
            return None

        # Check cache
        if label in self._embedding_cache:
            return self._embedding_cache[label]

        # Compute embedding
        try:
            embedding = self._embedding_model.encode(label, convert_to_numpy=True)
            self._embedding_cache[label] = embedding
            logger.debug(f"[Embedding] Cached embedding for '{label}' (shape={embedding.shape})")
            return embedding
        except Exception as e:
            logger.error(f"[Embedding] Failed to compute embedding for '{label}': {e}")
            return None

    def _compute_semantic_similarity(self, label1: str, label2: str) -> float:
        """
        Compute semantic similarity using embeddings.

        Phase 2B: Semantic embeddings.

        Args:
            label1: First label
            label2: Second label

        Returns:
            Cosine similarity (0-1)
        """
        emb1 = self._get_cached_embedding(label1)
        emb2 = self._get_cached_embedding(label2)

        if emb1 is None or emb2 is None:
            return 0.0

        try:
            import numpy as np

            # Cosine similarity
            similarity = float(
                np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            )

            logger.debug(
                f"[Semantic] '{label1}' vs '{label2}': similarity={similarity:.3f}"
            )

            return similarity
        except Exception as e:
            logger.error(f"[Semantic] Failed to compute similarity: {e}")
            return 0.0

    def _find_similar_node(
        self,
        label: str,
        graph: Graph,
        node_type: str
    ) -> Optional[Node]:
        """
        Find existing similar node using hybrid deduplication strategy.

        Phase 2A + 2B: Hybrid semantic deduplication.

        Search strategy (fast-to-slow):
        1. Exact match (O(1) via label index)
        2. Enhanced Jaccard with lemmas (O(n) but fast)
        3. Semantic embeddings (O(n) but slower, if enabled)

        Args:
            label: New node label
            graph: Current graph
            node_type: Node type (must match for deduplication)

        Returns:
            Matching node or None
        """
        # Fast path 1: Exact match
        existing = graph.get_node_by_label(label)
        if existing and existing.node_type == node_type:
            logger.debug(f"[Dedup] Exact match: '{label}' -> {existing.id}")
            return existing

        # Fast path 2: Enhanced Jaccard (Phase 2A)
        for node in graph.nodes.values():
            if node.node_type != node_type:
                continue

            jaccard_sim = self._jaccard_similarity_with_lemmas(label, node.label)
            if jaccard_sim >= self.jaccard_threshold:
                logger.info(
                    f"[Dedup] Jaccard match: '{label}' ~= '{node.label}' "
                    f"(similarity={jaccard_sim:.3f}, threshold={self.jaccard_threshold}, "
                    f"type={node_type})"
                )
                return node

        # Slow path: Semantic embeddings (Phase 2B)
        if self.embeddings_enabled:
            for node in graph.nodes.values():
                if node.node_type != node_type:
                    continue

                semantic_sim = self._compute_semantic_similarity(label, node.label)
                if semantic_sim >= self.embeddings_threshold:
                    logger.info(
                        f"[Dedup] Semantic match: '{label}' ~= '{node.label}' "
                        f"(similarity={semantic_sim:.3f}, threshold={self.embeddings_threshold}, "
                        f"type={node_type})"
                    )
                    return node

        logger.debug(f"[Dedup] No match found for '{label}' (type={node_type})")
        return None

    # =========================================================================
    # EXTRACTION METHODS
    # =========================================================================

    def assess_extractability(
        self,
        response: str,
        history: History
    ) -> Tuple[bool, Optional[str], Optional[Any]]:
        """
        Quick check: can we extract from this response?

        Returns:
            Tuple of (is_extractable, reason_if_not, llm_response)
        """
        system_prompt = """You are assessing whether a respondent's answer contains extractable concepts.

An answer is NOT extractable if:
- It's too vague to identify specific concepts (e.g., "I don't know, it just feels right")
- It's a simple yes/no without elaboration
- It's completely off-topic and doesn't relate to any discussed concepts
- It contains only filler words or hedging without any substance
- It's a literal word-for-word repetition of the previous response

An answer IS extractable if:
- It mentions specific things, features, or outcomes
- It expresses opinions, feelings, or evaluations about something concrete
- It describes experiences, examples, or comparisons
- It SUMMARIZES or RECAPS previous points (summaries reveal prioritization!)
- It connects or synthesizes multiple concepts together
- Even short answers can be extractable if they contain a clear concept
- Responses that restate points with new framing or emphasis ARE extractable

IMPORTANT: A summary or recap of previous discussion points IS extractable.
Summaries reveal how the respondent organizes, prioritizes, or frames concepts.
Only reject if the response is literally meaningless or completely off-topic.

Respond with JSON:
{
  "extractable": true/false,
  "reason": "explanation if not extractable, null if extractable"
}"""

        user_prompt = f"""Recent conversation:
        {history.format_for_prompt(n=3)}

        Latest response to assess:
        "{response}"

        Is this response extractable?"""

        llm_response = self.llm.complete(
            task=TaskType.EXTRACTABILITY_CHECK,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1
        )
        
        if not llm_response.success:
            # Default to extractable if LLM fails
            return True, None, llm_response

        try:
            result = self._parse_json(llm_response.content)
            extractable = result.get("extractable", True)
            reason = result.get("reason") if not extractable else None
            return extractable, reason, llm_response
        except Exception:
            return True, None, llm_response
    
    def extract(
        self,
        response: str,
        graph: Graph,
        history: History
    ) -> ExtractionResult:
        """
        Full extraction: nodes, edges, element mappings.
        Uses function calling for structured output with required element_mapping.

        Args:
            response: Respondent's text
            graph: Current graph (for merging with existing nodes)
            history: Conversation history

        Returns:
            ExtractionResult with nodes, edges, and mappings
        """
        # Build extraction prompt
        system_prompt = self._build_extraction_system_prompt()
        user_prompt = self._build_extraction_user_prompt(response, graph, history)

        # Use function calling for structured extraction
        llm_response = self.llm.complete(
            task=TaskType.GRAPH_EXTRACTION,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.3,
            tools=[EXTRACTION_TOOL]  # Use function calling schema
        )

        if not llm_response.success:
            return ExtractionResult(
                is_extractable=False,
                ambiguity_reason=f"Extraction failed: {llm_response.error}",
                raw_response=llm_response.content,
                llm_response=llm_response
            )

        # Parse extraction result (content is JSON from function call)
        result = self._parse_extraction_result(llm_response.content, graph)
        result.llm_response = llm_response
        return result
    
    def assess_momentum(
        self,
        response: str,
        history: History
    ) -> Momentum:
        """
        Assess respondent engagement level.
        
        Args:
            response: Latest response
            history: Conversation history
            
        Returns:
            Momentum assessment
        """
        system_prompt = """You are assessing respondent engagement in a qualitative interview.

        HIGH momentum indicators:
        - Long, elaborated responses with detail
        - Unprompted examples, stories, or concrete scenarios
        - Emotional language or emphasis
        - Self-initiated connections ("and that's also why...")
        - Enthusiasm, energy in the response

        NEUTRAL momentum indicators (this is the EXPECTED baseline):
        - Coherent answers with some detail
        - No example/story but not avoiding the topic
        - Mild emotional tone
        - Standard conversational engagement
        - May include THINKING-ALOUD hedging ("I guess", "to be honest") if accompanied by elaboration

        LOW momentum indicators:
        - Short AND unelaborated responses ("yeah", "dunno")
        - Repetition of previous content without new insight
        - Signs of withdrawal or disinterest
        - Deflection or topic avoidance
        - Fatigue signals (sighing, "I dunno… it's whatever")
        - Hedging ONLY WHEN the entire response stays vague

        CRITICAL BALANCING RULES:
        - If the response contains a mix of signals (e.g., hedging but also elaboration), classify as NEUTRAL
        - Long elaboration outweighs hedging
        - Emotional depth outweighs uncertainty
        - A concrete example outweighs brevity
        - Do NOT classify as "low" if the respondent expresses uncertainty while still giving a meaningful explanation
        - Hedging during reasoning/thinking is NORMAL in exploratory interviews - not low engagement
        - Only assign LOW when the pattern of disengagement holds across the response

        Respond with JSON:
        {
        "level": "high" | "neutral" | "low",
        "indicators": ["list", "of", "observed", "signals"]
        }"""

        # Include recent history for context (increased from 3 to 5 for better trend detection)
        recent_responses = []
        for turn in history.get_recent(5):
            recent_responses.append(f"- {turn.response[:100]}...")
        
        user_prompt = f"""Recent responses for context:
        {chr(10).join(recent_responses)}

        Latest response to assess:
        "{response}"

        What is the engagement level?"""

        llm_response = self.llm.complete(
            task=TaskType.MOMENTUM_ASSESSMENT,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.2
        )
        
        if not llm_response.success:
            momentum = Momentum.default()
            momentum.llm_response = llm_response
            return momentum

        try:
            result = self._parse_json(llm_response.content)
            momentum = Momentum(
                level=result.get("level", "neutral"),
                indicators=result.get("indicators", []),
                llm_response=llm_response
            )
            return momentum
        except Exception:
            momentum = Momentum.default()
            momentum.llm_response = llm_response
            return momentum
    
    def _build_extraction_system_prompt(self) -> str:
        """Build system prompt for extraction with type-aware sentiment guidance."""
        schema_prompt = self.schema.get_extraction_prompt()

        # Add reference elements with clear mapping instructions and types
        elements_section = ""
        sentiment_guidance = ""
        if self.coverage_state.reference_elements:
            elements_section = "\n\n## CRITICAL: Reference Elements for Coverage Tracking\n"
            elements_section += "You MUST map each extracted node to one of these reference elements when relevant.\n"
            elements_section += "Valid element_mapping values are:\n"

            # Build element list with types
            type_examples = {"problem": [], "solution": [], "evidence": []}
            for elem_id, elem in self.coverage_state.reference_elements.items():
                content_preview = elem.content[:150] + "..." if len(elem.content) > 150 else elem.content
                elem_type = elem.element_type if elem.element_type != "unknown" else "general"
                elements_section += f"- **\"{elem_id}\"** (type: {elem_type}): {content_preview}\n"

                # Group by type category for sentiment guidance
                if elem.element_type in ["problem", "tension", "frustration", "unmet_need"]:
                    type_examples["problem"].append(elem_id)
                elif elem.element_type in ["solution", "benefit", "outcome"]:
                    type_examples["solution"].append(elem_id)
                elif elem.element_type in ["evidence", "feature", "mechanism", "proof"]:
                    type_examples["evidence"].append(elem_id)

            elements_section += "- **null**: Use null ONLY if the node doesn't relate to any reference element above\n"

            # Add sentiment attribution guidance based on types
            sentiment_guidance = "\n\n## Sentiment Attribution Rules\n"
            sentiment_guidance += "When the respondent expresses sentiment, correctly identify the TARGET element:\n\n"

            if type_examples["problem"]:
                sentiment_guidance += f"**Problem-type elements ({', '.join(type_examples['problem'])}):**\n"
                sentiment_guidance += "- Negative reactions often indicate AGREEMENT with the problem (\"that's so annoying\" = validates the problem exists)\n"
                sentiment_guidance += "- Positive reactions may indicate the problem doesn't resonate with them\n\n"

            if type_examples["solution"]:
                sentiment_guidance += f"**Solution-type elements ({', '.join(type_examples['solution'])}):**\n"
                sentiment_guidance += "- Positive reactions indicate appeal of the solution\n"
                sentiment_guidance += "- Negative reactions indicate skepticism about the solution\n\n"

            if type_examples["evidence"]:
                sentiment_guidance += f"**Evidence-type elements ({', '.join(type_examples['evidence'])}):**\n"
                sentiment_guidance += "- Focus on whether the evidence is CONVINCING, not whether they like/dislike it\n"
                sentiment_guidance += "- \"That makes sense\" = evidence is credible; \"I don't buy it\" = evidence lacks credibility\n\n"

            sentiment_guidance += "**CRITICAL**: Do NOT conflate reactions to different elements. A negative reaction to a 'problem' element is different from rejecting a 'solution' element.\n"

        return f"""You are extracting concepts and relationships from interview responses.

{schema_prompt}
{elements_section}
{sentiment_guidance}

## CRITICAL: Edge Extraction Priority

Edges are AS IMPORTANT as nodes. For every concept, ask: "Why?" or "What does this lead to?"

**Extract edges when respondent uses:**
- Causal language: "because", "leads to", "results in", "causes"
- Explanations: "X is important because it gives me Y"
- Connections: "X is related to Y", "X enables Y"
- Sequences: "first X happens, then Y"

**Minimum expectation:** 1 edge per 2 nodes
If you extract 4 nodes, extract at least 2 edges connecting them.

**Edge validation checklist:**
1. Both nodes exist or are being created in this extraction
2. Relationship is explicitly stated or strongly implied
3. Relation type follows schema adjacency rules

## Extraction Rules

1. **Extract ATOMIC concepts (avoid over-fragmentation)**
   - Each node = ONE concept that could vary independently
   - Break compounds ONLY if respondent treats them separately:
     * "thick and creamy" → ONE node if treated as single quality
     * "thick and creamy" → TWO nodes if discussed separately

   Test: Would respondent have mentioned one without the other?

   Examples:
   - "smooth creamy texture" → ONE node (single gestalt quality)
   - "smooth texture, plus it's also creamy" → TWO nodes (distinct attributes)

2. **Only extract what is explicitly stated or clearly implied**
   - Every node must have quote support
   - Don't infer relationships that aren't expressed

3. **Use existing nodes when possible**
   - If a concept matches an existing node, reference it by label
   - Only create new nodes for genuinely new concepts

4. **Ambiguity Detection (Mark Liberally)**

Set **is_ambiguous: true** when:
1. **Type uncertainty**: "smooth" (texture attribute? ease-of-use functional?)
2. **Scope ambiguity**: "it works well" (what is "it"? what aspect "works"?)
3. **Implicit reference**: "that", "the other one" without clear antecedent
4. **Polysemy**: "security" (physical? data? financial?)
5. **Vague intensifiers**: "very good", "quite nice" without specifics

**Default when uncertain: FLAG IT**
It's better to flag 10 nodes and clarify 2 than miss 1 critical ambiguity.

5. **Map to reference elements (when clear)**
   - Provide element_mapping ONLY if the node clearly discusses that element
   - When uncertain, use null (don't force mappings)
   - Concept extraction takes priority over element mapping

## Reaction Detection
For each node, assess if the respondent expressed an evaluative stance toward that specific concept:
- "positive": Approval, enthusiasm, agreement ("that sounds great", "I love that")
- "negative": Disapproval, rejection ("I don't like that", "that's annoying")
- "neutral": Acknowledgment without evaluation ("I see", "okay")
- "skeptical": Doubt or uncertainty about the concept ("not sure about that")
- "curious": Interest wanting to know more ("tell me more", "how does that work?")
- null: No evaluative stance expressed

NOTE: A single response may contain multiple reactions to different concepts.
Extract each node's reaction independently based on what the respondent said about THAT specific concept.

Call the extract_graph_delta function with the extracted nodes and edges."""

    def _build_extraction_user_prompt(
        self,
        response: str,
        graph: Graph,
        history: History
    ) -> str:
        """Build user prompt for extraction."""
        # Existing nodes for reference
        existing_nodes = []
        for node in graph.nodes.values():
            existing_nodes.append(f"- {node.label} ({node.node_type})")
        
        existing_section = ""
        if existing_nodes:
            existing_section = f"""
Existing nodes in graph (reference these if concepts match):
{chr(10).join(existing_nodes[:20])}  # Limit to avoid token overflow
"""
        
        return f"""## Conversation Context
{history.format_for_prompt(n=3)}

{existing_section}

## Response to Extract
"{response}"

Extract nodes and edges from this response."""

    def _parse_extraction_result(
        self,
        llm_content: str,
        graph: Graph
    ) -> ExtractionResult:
        """Parse LLM extraction response into ExtractionResult."""
        try:
            data = self._parse_json(llm_content)
        except Exception as e:
            return ExtractionResult(
                is_extractable=False,
                ambiguity_reason=f"Failed to parse extraction: {e}",
                raw_response=llm_content
            )

        # Reaction intensity order for conflict resolution
        REACTION_INTENSITY = {"positive": 5, "negative": 5, "skeptical": 3, "curious": 2, "neutral": 1}
        element_reactions = {}

        nodes = []
        edges = []
        node_element_mappings = {}
        label_to_id = {}  # Track label -> node_id for edge creation
        
        # Process nodes with semantic deduplication
        for node_data in data.get("nodes", []):
            label = node_data.get("label", "").strip()
            if not label:
                continue

            node_type = node_data.get("node_type")
            if not node_type:
                logger.warning(f"[Extraction] Node '{label}' missing node_type, skipping")
                continue

            # Check if similar node already exists (semantic deduplication)
            existing = self._find_similar_node(label, graph, node_type)
            if existing:
                label_to_id[label] = existing.id
                logger.debug(
                    f"[Extraction] Reusing existing node: '{label}' -> {existing.id} ('{existing.label}')"
                )
                continue

            # Create new node
            node = Node(
                label=label,
                node_type=node_type,
                is_ambiguous=node_data.get("is_ambiguous", False),
                metadata={
                    "quote": node_data.get("quote", ""),
                }
            )
            nodes.append(node)
            label_to_id[label] = node.id
            logger.debug(f"[Extraction] Created new node: '{label}' ({node_type}) -> {node.id}")

            # Track element mapping
            element_mapping = node_data.get("element_mapping")
            if element_mapping and element_mapping in self.coverage_state.reference_elements:
                node_element_mappings[node.id] = element_mapping

                # Track reaction for coverage
                reaction = node_data.get("reaction")
                if reaction:
                    existing = element_reactions.get(element_mapping)
                    if not existing or REACTION_INTENSITY.get(reaction, 0) > REACTION_INTENSITY.get(existing, 0):
                        element_reactions[element_mapping] = reaction
        
        # Process edges
        for edge_data in data.get("edges", []):
            source_label = edge_data.get("source_label", "").strip()
            target_label = edge_data.get("target_label", "").strip()
            
            # Look up node IDs
            source_id = label_to_id.get(source_label)
            target_id = label_to_id.get(target_label)
            
            # Also check existing graph
            if not source_id:
                existing = graph.get_node_by_label(source_label)
                if existing:
                    source_id = existing.id
            if not target_id:
                existing = graph.get_node_by_label(target_label)
                if existing:
                    target_id = existing.id
            
            if not source_id or not target_id:
                continue  # Skip edges with missing nodes

            # Use schema-agnostic default edge type if LLM doesn't specify
            default_edge_type = self.schema.get_default_edge_type()

            edge = Edge(
                source_id=source_id,
                target_id=target_id,
                relation_type=edge_data.get("relation_type", default_edge_type),
                metadata={
                    "quote": edge_data.get("quote", ""),
                }
            )
            edges.append(edge)
        
        # Log extraction details
        if nodes:
            logger.info(f"[Extraction] Nodes: {[n.label for n in nodes]}")
        if edges:
            edge_desc = [(graph.get_node(e.source_id).label if graph.get_node(e.source_id) else e.source_id,
                         e.relation_type,
                         graph.get_node(e.target_id).label if graph.get_node(e.target_id) else e.target_id)
                        for e in edges]
            logger.info(f"[Extraction] Edges: {edge_desc}")
        if node_element_mappings:
            logger.info(f"[Extraction] Element mappings: {node_element_mappings}")

        # Validate extraction result quality
        validation_warnings = self.schema.validate_extraction_result(
            nodes=nodes,
            edges=edges,
            strict=False  # Log warnings, don't fail
        )

        if validation_warnings:
            for warning in validation_warnings:
                logger.warning(f"[Extraction Validation] {warning}")

        return ExtractionResult(
            nodes=nodes,
            edges=edges,
            node_element_mappings=node_element_mappings,
            element_reactions=element_reactions,
            is_extractable=True,
            raw_response=llm_content
        )
    
    def _parse_json(self, content: str) -> Dict:
        """Parse JSON from LLM response, handling markdown code blocks."""
        # Remove markdown code blocks if present
        content = content.strip()
        if content.startswith("```"):
            # Remove opening ```json or ```
            content = re.sub(r'^```(?:json)?\s*\n?', '', content)
            # Remove closing ```
            content = re.sub(r'\n?```\s*$', '', content)
        
        return json.loads(content)
