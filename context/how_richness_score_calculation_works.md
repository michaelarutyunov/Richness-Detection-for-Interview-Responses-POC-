# How the Richness Score Calculation Works

The richness score is a fundamental metric in the AI Interview System that quantifies the depth and quality of insights extracted from a participant's responses. It serves as both a measure of interview progress and a termination criterion. Let me break down how this sophisticated scoring system works.

## The Philosophy Behind Richness

Imagine you're having a conversation about a new coffee subscription service. Saying "it's convenient" adds some value to our understanding, but saying "it saves me precious time during hectic mornings so I can focus on being present with my family rather than rushing to the store" reveals much deeper insights about your values, lifestyle, and emotional connections.

The richness score captures this difference - it rewards the system for extracting not just surface-level features, but the meaningful connections and deeper implications that reveal why people really care about products.

## The Core Calculation Formula

At its heart, richness calculation follows a simple but powerful formula:

**`Total Richness = Sum of all node weights + Sum of all edge boosts`**

But the magic lies in how these weights and boosts are determined and applied.

### Node Richness Weights: The Foundation

Each concept (node) in your knowledge graph has a predefined "richness weight" based on its type. Looking at the schema configuration:

```yaml
# Node Types and their Richness Weights
- attribute: 0.5          # Concrete features (price, taste, size)
- functional_consequence: 1.0   # Practical outcomes (saves time, reduces mess)
- psychosocial_consequence: 1.5 # Personal meanings (feel competent, look good)
- value: 2.0              # Core life goals (security, belonging, family wellbeing)
```

This hierarchy reflects a fundamental insight from consumer psychology: concrete product features are easy to mention but don't reveal much about motivation, while values and emotional connections reveal the deep "why" behind consumer behavior.

**Example in Practice:**
- "It's affordable" (attribute) → 0.5 richness points
- "It saves me money" (functional consequence) → 1.0 richness points  
- "It gives me peace of mind about finances" (psychosocial consequence) → 1.5 richness points
- "It helps me provide security for my family" (value) → 2.0 richness points

### Edge Richness Boosts: The Connection Bonus

When the system discovers that two concepts are connected (like "fresh roasting" → "better taste"), it adds "richness boosts" for these relationships:

```yaml
# Edge Types and their Richness Boosts
- leads_to: 1.0     # A causes or enables B
- blocks: 2.0       # A prevents or hinders B (very valuable insight!)
```

**Why Connections Matter:** A participant who can articulate how product features lead to personal benefits reveals a sophisticated mental model. Even more valuable is understanding what prevents them from achieving their goals - these "blockages" often represent the most important insights for product development.

**Example:** "The premium price blocks me from feeling financially secure" reveals a tension between quality desires and financial constraints that's worth 2.0 richness points.

## The Calculation in Action: A Real Example

Let's trace through the actual sample interview to see how richness accumulated:

**Initial Concept:** "A premium coffee subscription service that delivers freshly roasted beans from local roasters every month."

**Turn 1 - Participant Response:** "I like that it is locally roasted. it mean the beans will be fresh and the coffee will be very aromatic."

**Extracted Concepts:**
- "locally_roasted" (attribute, 0.5 points)
- "fresh" (attribute, 0.5 points) 
- "aromatic" (attribute, 0.5 points)

**Richness Added:** 1.5 points (3 attributes × 0.5)

**Turn 2+ - More Responses Reveal:**
- "weekly_delivery" (functional consequence, 1.0 points)
- "stale_beans" (attribute, 0.5 points)
- "aroma_loss" (attribute, 0.5 points)
- Connection: "weekly_delivery" → prevents → "stale_beans" (edge boost, 1.0 points)

**Final Result:** After 11 turns, the interview achieved 8.0 total richness points from 13 nodes and 1 edge.

## Different Types of Richness Calculations

The system actually calculates richness at multiple levels:

### 1. Graph Delta Richness (Per Response)
When processing each participant response, the system calculates how much richness that specific response contributed:

```python
def _calculate_richness(self, nodes, edges, existing_graph):
    richness = 0.0
    
    # New nodes discovered
    for node in nodes:
        weight = existing_graph.schema.get_richness_weight(node.type)
        richness += weight
        
    # New connections discovered  
    for edge in edges:
        boost = existing_graph.schema.get_richness_boost(edge.type)
        richness += boost
        
    return richness
```

This helps the system understand which responses were most insightful and can guide future questioning strategy.

### 2. Cumulative Graph Richness (Total Interview)
This is the running total of all richness across the entire interview, calculated exactly the same way but across all nodes and edges in the complete knowledge graph.

### 3. Interview State Richness (Progress Tracking)
Used to determine interview phases and termination:

```python
richness = self.graph.calculate_richness()
if coverage < 0.3:
    phase = InterviewPhase.COVERAGE
elif richness < self.min_richness * 0.7:
    phase = InterviewPhase.DEPTH
elif coverage < 0.8:
    phase = InterviewPhase.CONNECTION
else:
    phase = InterviewPhase.WRAP_UP
```

## Richness Thresholds and Decision Making

The system uses richness scores to make strategic decisions:

**Termination Criteria:**
- `min_richness = 5.0` (configurable) - Stop when total richness reaches this threshold
- This prevents overly long interviews once sufficient insight depth is achieved

**Opportunity Ranking:**
- Nodes from more "valuable" types (values > psychosocial > functional > attributes) get higher priority
- The system uses richness weights when determining what to explore next

**Phase Transitions:**
- Low richness → Stay in coverage mode (explore broadly)
- Medium richness → Move to depth mode (go deeper on interesting areas)
- High richness → Focus on connections between concepts

## The Intelligence Behind the Scoring

What makes this richness calculation sophisticated is that it's not just counting concepts - it's encoding psychological insights about consumer behavior:

1. **Abstraction Hierarchy:** Higher-level concepts (values) are weighted more because they reveal deeper motivations
2. **Connection Value:** Understanding relationships between concepts is rewarded because it reveals mental model structure
3. **Barrier Recognition:** Negative connections (blocks) are weighted highest because they reveal pain points and obstacles

## Limitations and Considerations

**Potential Issues:**
- The system might over-value abstract concepts and under-value concrete insights
- Fixed weights don't account for context - sometimes a concrete feature might be more important than a vague value statement
- No mechanism to adjust weights based on product category or interview context

**Quality vs. Quantity:** Richness rewards having diverse concept types but doesn't directly measure insight quality - a profound statement about convenience might be more valuable than three vague value statements.

## Real-World Impact

In practice, the richness score serves as a reliable proxy for interview depth. The sample interview reached 8.0 richness points, which exceeded the 5.0 minimum threshold, indicating that the conversation moved beyond surface-level features into meaningful consequences and personal values.

The score helps researchers quickly understand whether an interview successfully uncovered the "why" behind consumer preferences, not just the "what" - transforming raw conversation into measurable insight depth.