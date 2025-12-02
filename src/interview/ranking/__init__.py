"""
Opportunity ranking module.

Exports:
- OpportunityRanker: Main public API
- QuestionStrategy: Enum for question strategies
- RankedOpportunity: Data class for ranked opportunities
"""

from src.interview.ranking.opportunity_ranker import OpportunityRanker
from src.interview.ranking.ranking_engine import QuestionStrategy, RankedOpportunity

__all__ = ["OpportunityRanker", "QuestionStrategy", "RankedOpportunity"]
