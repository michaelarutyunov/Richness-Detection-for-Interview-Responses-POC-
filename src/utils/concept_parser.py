"""
Concept parsing utilities.
Parses concept files and extracts elements (insight, promise, RTB).
"""

import re
from pathlib import Path
from typing import Dict, Optional, Any
from pydantic import BaseModel, Field

from utils.llm_manager import LLMManager, TaskType
from utils.logger import get_logger

logger = get_logger(__name__)


class ConceptElements(BaseModel):
    """Parsed concept elements with type classification."""
    insight: str = Field(default="", description="The consumer insight")
    insight_type: str = Field(
        default="problem",
        description="Insight type: problem, tension, frustration, unmet_need"
    )
    promise: str = Field(default="", description="The product promise/benefit")
    promise_type: str = Field(
        default="solution",
        description="Promise type: solution, benefit, outcome"
    )
    rtb: str = Field(default="", description="Reason to believe")
    rtb_type: str = Field(
        default="evidence",
        description="RTB type: evidence, feature, mechanism, proof"
    )
    full_text: str = Field(default="", description="Complete concept text")


class ParsedConcept(BaseModel):
    """Complete parsed concept with metadata."""
    name: str = Field(description="Concept name")
    description: str = Field(description="Full concept description")
    elements: ConceptElements = Field(default_factory=ConceptElements)
    source_path: Optional[str] = Field(default=None)
    
    def get_element_config(self) -> Dict[str, Any]:
        """
        Convert to element_config format for CoverageState.

        Returns:
            Dictionary suitable for CoverageState.initialize()
        """
        config = {}

        if self.elements.insight:
            config["insight"] = {
                "content": self.elements.insight,
                "element_type": self.elements.insight_type,
                "requirements": {
                    "mention": True,
                    "reaction": True,
                    "comprehension": False,
                    "connections_to": []
                }
            }

        if self.elements.promise:
            config["promise"] = {
                "content": self.elements.promise,
                "element_type": self.elements.promise_type,
                "requirements": {
                    "mention": True,
                    "reaction": True,
                    "comprehension": False,
                    "connections_to": ["insight"] if self.elements.insight else []
                }
            }

        if self.elements.rtb:
            config["rtb"] = {
                "content": self.elements.rtb,
                "element_type": self.elements.rtb_type,
                "requirements": {
                    "mention": True,
                    "reaction": True,
                    "comprehension": True,  # RTB often needs comprehension check
                    "connections_to": ["promise"] if self.elements.promise else []
                }
            }

        return config


class ConceptParser:
    """
    Parses concept files and extracts structured elements.
    Uses LLM for semantic parsing of insight/promise/RTB.
    """
    
    def __init__(self, llm_manager: Optional[LLMManager] = None):
        self.llm = llm_manager
    
    def parse_file(self, path: str) -> ParsedConcept:
        """
        Parse a concept from a markdown file.
        
        Args:
            path: Path to concept file
            
        Returns:
            ParsedConcept with name, description, and elements
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Concept file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract name and description from markdown format
        name = self._extract_name(content)
        description = self._extract_description(content)
        
        logger.info(f"Parsed concept file: {path.name}")
        logger.debug(f"Concept name: {name}")
        
        # Parse elements
        if self.llm:
            elements = self._parse_elements_with_llm(description)
        else:
            elements = self._parse_elements_heuristic(description)
        
        return ParsedConcept(
            name=name,
            description=description,
            elements=elements,
            source_path=str(path)
        )
    
    def parse_text(self, text: str, name: str = "Unnamed Concept") -> ParsedConcept:
        """
        Parse a concept from raw text.
        
        Args:
            text: Concept text
            name: Concept name
            
        Returns:
            ParsedConcept with elements
        """
        if self.llm:
            elements = self._parse_elements_with_llm(text)
        else:
            elements = self._parse_elements_heuristic(text)
        
        return ParsedConcept(
            name=name,
            description=text,
            elements=elements
        )
    
    def _extract_name(self, content: str) -> str:
        """Extract concept name from markdown."""
        # Look for # Name followed by quoted text
        match = re.search(r'#\s*Name\s*\n+"([^"]+)"', content, re.IGNORECASE)
        if match:
            return match.group(1)
        
        # Fallback: first # header
        match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if match:
            return match.group(1).strip()
        
        return "Unnamed Concept"
    
    def _extract_description(self, content: str) -> str:
        """Extract concept description from markdown."""
        # Look for # Description followed by quoted text
        match = re.search(
            r'#\s*Description\s*\n+"([^"]+)"', 
            content, 
            re.IGNORECASE | re.DOTALL
        )
        if match:
            return match.group(1).strip()
        
        # Fallback: everything after first paragraph
        lines = content.strip().split('\n')
        description_lines = []
        in_description = False
        
        for line in lines:
            if line.startswith('#'):
                if 'description' in line.lower():
                    in_description = True
                continue
            if in_description or not line.startswith('#'):
                # Skip the name section
                if '"' in line and not description_lines:
                    # This might be the name, check if short
                    text = line.strip().strip('"')
                    if len(text) < 50:
                        continue
                description_lines.append(line)
        
        return '\n'.join(description_lines).strip().strip('"')
    
    def _parse_elements_with_llm(self, description: str) -> ConceptElements:
        """Use LLM to parse concept elements with type classification."""
        system_prompt = """You are analyzing a product concept to identify its core components and classify their types.

A concept typically has three parts:
1. **Insight**: The consumer problem, tension, or unmet need being addressed.
   Types: "problem" (explicit issue), "tension" (conflicting desires), "frustration" (pain point), "unmet_need" (gap/desire)

2. **Promise**: The benefit or solution being offered.
   Types: "solution" (direct fix), "benefit" (outcome/advantage), "outcome" (end result)

3. **Reason to Believe (RTB)**: Evidence that makes the promise credible.
   Types: "evidence" (proof/data), "feature" (product capability), "mechanism" (how it works), "proof" (testimonial/study)

Extract these elements AND classify each one's type. If an element is not clearly present, leave it empty.

Respond with JSON:
{
  "insight": "The insight text...",
  "insight_type": "problem",
  "promise": "The promise text...",
  "promise_type": "solution",
  "rtb": "The reason to believe text...",
  "rtb_type": "evidence"
}"""

        user_prompt = f"""Parse this concept into insight, promise, and RTB. Classify each element's type:

---
{description}
---"""

        response = self.llm.complete(
            task=TaskType.GRAPH_EXTRACTION,  # Reuse extraction task settings
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.2
        )

        if not response.success:
            logger.warning(f"LLM parsing failed: {response.error}")
            return self._parse_elements_heuristic(description)

        try:
            import json
            # Clean markdown code blocks
            content = response.content.strip()
            if content.startswith("```"):
                content = re.sub(r'^```(?:json)?\s*\n?', '', content)
                content = re.sub(r'\n?```\s*$', '', content)

            data = json.loads(content)

            elements = ConceptElements(
                insight=data.get("insight", ""),
                insight_type=data.get("insight_type", "problem"),
                promise=data.get("promise", ""),
                promise_type=data.get("promise_type", "solution"),
                rtb=data.get("rtb", ""),
                rtb_type=data.get("rtb_type", "evidence"),
                full_text=description
            )

            logger.info("Concept elements parsed via LLM")
            logger.debug(f"Insight ({elements.insight_type}): {elements.insight[:50]}..." if elements.insight else "Insight: (empty)")
            logger.debug(f"Promise ({elements.promise_type}): {elements.promise[:50]}..." if elements.promise else "Promise: (empty)")
            logger.debug(f"RTB ({elements.rtb_type}): {elements.rtb[:50]}..." if elements.rtb else "RTB: (empty)")

            return elements

        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return self._parse_elements_heuristic(description)
    
    def _parse_elements_heuristic(self, description: str) -> ConceptElements:
        """
        Heuristic parsing of concept elements.
        Used when LLM is not available.
        """
        sentences = self._split_sentences(description)
        
        insight = ""
        promise = ""
        rtb = ""
        
        # Simple heuristic: 
        # - First 1-2 sentences with problem indicators = insight
        # - Sentences with benefit language = promise  
        # - Sentences with proof/evidence = RTB
        
        insight_indicators = [
            "but", "however", "frustrat", "problem", "struggle", 
            "difficult", "pain", "annoying", "wish", "want"
        ]
        promise_indicators = [
            "now", "finally", "introducing", "different", 
            "solution", "helps", "makes", "gives you", "so you can"
        ]
        rtb_indicators = [
            "because", "proven", "clinical", "made with", "contains",
            "technology", "ingredient", "patent", "research", "studies"
        ]
        
        insight_sentences = []
        promise_sentences = []
        rtb_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Check for RTB first (most specific)
            if any(ind in sentence_lower for ind in rtb_indicators):
                rtb_sentences.append(sentence)
            # Then promise
            elif any(ind in sentence_lower for ind in promise_indicators):
                promise_sentences.append(sentence)
            # Then insight
            elif any(ind in sentence_lower for ind in insight_indicators):
                insight_sentences.append(sentence)
            # Default: first sentences likely insight, later sentences likely promise
            elif not insight_sentences and len(promise_sentences) == 0:
                insight_sentences.append(sentence)
            elif not promise_sentences:
                promise_sentences.append(sentence)
        
        insight = " ".join(insight_sentences)
        promise = " ".join(promise_sentences)
        rtb = " ".join(rtb_sentences)
        
        logger.info("Concept elements parsed via heuristics")
        
        return ConceptElements(
            insight=insight,
            promise=promise,
            rtb=rtb,
            full_text=description
        )
    
    def _split_sentences(self, text: str) -> list:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


def load_concept(
    path: str, 
    llm_manager: Optional[LLMManager] = None
) -> ParsedConcept:
    """
    Convenience function to load and parse a concept file.
    
    Args:
        path: Path to concept file
        llm_manager: Optional LLM manager for semantic parsing
        
    Returns:
        ParsedConcept
    """
    parser = ConceptParser(llm_manager)
    return parser.parse_file(path)


def list_concepts(concepts_dir: str) -> list:
    """
    List all concept files in a directory.
    
    Args:
        concepts_dir: Path to concepts directory
        
    Returns:
        List of concept file paths
    """
    path = Path(concepts_dir)
    if not path.exists():
        return []
    
    return sorted([
        str(f) for f in path.glob("*.md")
    ])
