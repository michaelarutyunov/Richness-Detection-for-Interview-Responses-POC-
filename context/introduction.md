# AI Interview System - Complete Beginner's Guide

## What This System Does

Imagine you're a product researcher conducting interviews to understand what customers value about a product. Instead of a human interviewer, this AI system automatically generates questions to explore the participant's thoughts deeply and systematically.

**Goal:** Extract a rich "mental model" of how the participant thinks about the product - what features they value, why they value them, and how different aspects connect together.

---

## Core Concepts Explained

### 1. The Knowledge Graph

**What it is:** A web of interconnected ideas extracted from the participant's responses.

**Analogy:** Think of it like a mind map that grows as the interview progresses.

**Components:**
- **Nodes** = Individual concepts the participant mentions
  - Example: "fast_delivery", "fresh_beans", "saves_time"

- **Edges** = Connections between concepts showing relationships
  - Example: "fast_delivery ’ saves_time" (delivery leads to saving time)

**Visual example:**
```
freshly_roasted_beans ’ aromatic_coffee ’ feel_energized ’ productive_morning
         “                                      “
    fresh_taste                          pleasant_sensation
```

### 2. Node Types (The "Schema")

Nodes are categorized into 4 types forming a hierarchy called a **"Means-End Chain"** - moving from concrete features to abstract values:

1. **Attributes** (bottom layer)
   - Physical features or characteristics
   - Example: "locally roasted", "premium beans", "weekly delivery"

2. **Functional Consequences** (middle layer)
   - Direct outcomes or effects
   - Example: "saves_time", "tastes_better", "stays_fresh"

3. **Psychosocial Consequences** (upper layer)
   - Emotional or social impacts
   - Example: "feel_confident", "impresses_guests", "peace_of_mind"

4. **Values** (top layer)
   - Core personal values
   - Example: "self_actualization", "financial_security", "family_wellbeing"

**Why this matters:** The system tries to climb this ladder - starting with concrete features and eventually reaching deeper values.

### 3. Edge Types (Relationship Types)

Connections between nodes can be:
- **leads_to**: One thing causes another (A ’ B)
- **enables**: One thing makes another possible
- **correlates_with**: Two things happen together
- **exemplifies**: One thing is an example of another
- **means**: One thing is the way to achieve another
- **blocks**: One thing prevents another

### 4. Richness Score

**What it is:** A numerical measure of how much understanding we've extracted.

**How it's calculated:**
- Each node type has a weight:
  - Attribute: 0.5 points
  - Functional Consequence: 2.0 points
  - Psychosocial Consequence: 2.5 points
  - Value: 3.0 points

- Each edge adds a boost:
  - leads_to: +0.5 points
  - enables: +0.3 points
  - correlates_with: +0.5 points
  - exemplifies: +0.3 points

- **Total Richness** = Sum of all node weights + Sum of all edge boosts

**Example:**
```
Interview with 5 attributes (2.5 pts) + 3 functional consequences (6.0 pts)
+ 4 edges (2.0 pts) = 10.5 richness score
```

**Target:** Interviews typically aim for 25-30 richness before wrapping up.

---

## The Interview Flow - Step by Step

### Turn-by-Turn Process

Each interview turn follows this sequence:

```
1. Participant responds to question
2. System extracts nodes and edges from response
3. System calculates richness increase
4. System determines current interview phase
5. System ranks all possible next questions
6. System selects best question
7. System generates the actual question text
8. Question presented to participant
9. Repeat
```

Let's break down each step:

---

## Step 1: Interview Phases

**What they are:** Different stages of the interview, each with a different goal.

### Phase 1: COVERAGE (Early Interview)
**Goal:** Explore broadly, touch many different topics
**When:** Coverage < 30% (few node types explored)
**Behavior:** Ask about new, unexplored areas
**Example:** "What do you like about the delivery?" ’ "How about the taste?"

### Phase 2: DEPTH (Middle Interview)
**Goal:** Go deeper into topics already mentioned
**When:** Coverage e 30% but richness < 70% of target
**Behavior:** Ask follow-up questions about existing nodes
**Example:** "You mentioned it saves time. How exactly does that help you?"

### Phase 3: CONNECTION (Later Interview)
**Goal:** Link different topics together
**When:** Coverage e 30% and richness e 70% of target
**Behavior:** Ask how concepts relate
**Example:** "How does the fresh taste connect to your morning routine?"

### Phase 4: WRAP_UP (Final Interview)
**Goal:** Finish gracefully
**When:** Richness target reached or max turns hit
**Behavior:** "Is there anything else you'd like to share?"

---

## Step 2: Opportunity Ranking

**What is an "opportunity"?** A potential next question we could ask about a specific node.

### The 5 Scoring Dimensions

For every node in the graph, we calculate 5 scores (each from 0.0 to 1.0):

#### 1. Coverage Score
**Measures:** How underexplored is this node's type?
**Formula:** 1 / (number of this type + 1)
**Example:**
- Only 1 attribute explored so far ’ coverage_score = 1/2 = 0.50
- Already 5 attributes explored ’ coverage_score = 1/6 = 0.17
**Higher when:** This node type hasn't been explored much yet

#### 2. Depth Score
**Measures:** How shallow is this node? (Does it need more elaboration?)
**Formula:** 1 / (number of outgoing edges + 1)
**Example:**
- Node has 0 children ’ depth_score = 1/1 = 1.0 (very shallow!)
- Node has 3 children ’ depth_score = 1/4 = 0.25 (already deep)
**Higher when:** Node hasn't been elaborated on much

#### 3. Recency Score
**Measures:** How long since we visited this node?
**Formula:** Decays exponentially with visits: 1/2^(visit_count)
**Example:**
- Never visited ’ recency_score = 1.0
- Visited once ’ recency_score = 0.5
- Visited twice ’ recency_score = 0.25
**Higher when:** Node hasn't been asked about recently

#### 4. Focus Score
**Measures:** How related is this to what we just talked about?
**Values:**
- Child of recent topic ’ 1.0 (very related)
- Parent of recent topic ’ 0.8 (somewhat related)
- Unrelated ’ 0.3 (keep some coherence)
**Higher when:** Node is close to recent conversation

#### 5. Diversity Score
**Measures:** How far is this from recent topics? (for variety)
**Formula:** Distance in graph / 3 (capped at 1.0)
**Example:**
- 1 edge away ’ diversity_score = 0.33
- 3+ edges away ’ diversity_score = 1.0
**Higher when:** Node is far from recent focus (adds variety)

### Phase-Adaptive Weights

Different phases prioritize different scores by multiplying them by different weights:

**COVERAGE Phase weights:**
- Coverage × **4.0** (strongly favor new types!)
- Depth × 1.0
- Recency × 1.5
- Focus × 1.0
- Diversity × 1.0

**DEPTH Phase weights:**
- Coverage × 2.0
- Depth × **2.5** (favor shallow nodes!)
- Recency × 2.0
- Focus × 1.5
- Diversity × 1.0

**CONNECTION Phase weights:**
- Coverage × 1.0
- Depth × 2.0
- Recency × 1.5
- Focus × **2.5** (stay coherent!)
- Diversity × 1.0

### Final Priority Calculation

**Priority Score** = (coverage_score × coverage_weight)
                   + (depth_score × depth_weight)
                   + (recency_score × recency_weight)
                   + (focus_score × focus_weight)
                   + (diversity_score × diversity_weight)

All opportunities are ranked by this priority score (highest first).

---

## Step 3: Question Strategies

Each opportunity is assigned a **strategy** based on the node's state:

### INTRODUCE_TOPIC
**When:** Node has never been discussed before (visit_count = 0)
**Purpose:** Bring up a new topic
**Example:** "Let's talk about delivery. How do you feel about it?"

### DIG_DEEPER
**When:** Node visited before BUT has few/no children (shallow)
**Purpose:** Get more details about something already mentioned
**Example:** "You mentioned fast delivery. Can you tell me more about that?"

### CONNECT_CONCEPTS
**When:** Node visited before AND has multiple children (well-explored)
**Purpose:** Link this concept to others
**Example:** "How does fast delivery relate to your morning routine?"

---

## Step 4: Opportunity Selection

We don't always pick the #1 ranked opportunity! We use **epsilon-greedy exploration**:

**What this means:** Sometimes we deliberately pick a random option instead of the best one, to add variety.

**Exploration rates by phase:**
- COVERAGE: 30% random (pick from top 5)
- DEPTH: 20% random
- CONNECTION: 10% random
- Otherwise: 0% random (always pick best)

**After selection:**
1. **Visit the node:** Increment its visit_count, record turn number
2. **Update focus:** Add to a memory of recent topics (last 5)

---

## Step 5: Question Text Generation

Now we have a strategy and a target node. How do we create the actual question text?

### Method 1: LLM Generation (Primary)

**What's an LLM?** Large Language Model (like ChatGPT) - an AI that generates text.

**Inputs to the LLM:**
- The strategy (dig_deeper / connect_concepts / introduce_topic)
- The target node label and type
- Recent conversation history (last 6 turns)
- The participant's last response

**Output:** Natural, conversational question text

**Example:**
```
Input:
- Strategy: dig_deeper
- Node: "aromatic_coffee" (functional_consequence)
- Last response: "The coffee smells amazing in the morning"

Output: "When you walk into the kitchen after brewing, what does
         that aroma feel like for you?"
```

### Method 2: Template Fallback (Secondary)

If the LLM fails or is unavailable, use pre-written templates:

**DIG_DEEPER templates:**
- "You mentioned {node}. Can you tell me more about that?"
- "That's interesting. What makes {node} important to you?"
- "Help me understand {node} better."

**CONNECT_CONCEPTS templates:**
- "How do {node_a} and {node_b} relate in your mind?"
- "What's the connection between {node_a} and {node_b}?"

**INTRODUCE_TOPIC templates:**
- "Let's explore {topic}. What are your thoughts on that?"
- "I'd like to hear about {topic}. What stands out to you?"

### Deduplication Check

Before finalizing, the system checks if this question is too similar to recent questions:

**Two similarity checks:**
1. **Word overlap:** Do they share 60%+ of the same words?
2. **Semantic similarity:** Do they mean the same thing?

**If repetitive:**
- Try generating again (max 3 attempts)
- On final attempt, add variety phrases like "Building on that..." or "Following up..."

---

## Complete Example: One Turn

Let's walk through a full turn:

### Setup
- Turn 3 of interview
- Participant just said: "The coffee makes me feel energized"
- Graph now has: 5 nodes, 3 edges
- Richness: 8.5 / 25.0 target

### Step 1: Extract from Response
```
New nodes:
- "feel_energized" (psychosocial_consequence) = +2.5 pts

New edges:
- aromatic_coffee ’ feel_energized (leads_to) = +0.5 pts

Richness increase: +3.0 pts
New total: 11.5
```

### Step 2: Determine Phase
```
Coverage: 3/4 node types = 75% (> 30%)
Richness: 11.5 / 25 = 46% (< 70%)
’ DEPTH phase
```

### Step 3: Rank Opportunities

Using DEPTH weights (coverage=2.0, depth=2.5, recency=2.0, focus=1.5):

```
Node: "feel_energized"
- coverage_score: 0.33 (3 psychosocial nodes exist)
- depth_score: 1.0 (no children yet)
- recency_score: 1.0 (just created, never visited)
- focus_score: 1.0 (just mentioned)
- diversity_score: 0.3
’ Priority: (0.33×2.0)+(1.0×2.5)+(1.0×2.0)+(1.0×1.5)+(0.3×1.0) = 6.96

Node: "aromatic_coffee"
- coverage_score: 0.25 (4 functional nodes exist)
- depth_score: 0.5 (has 1 child)
- recency_score: 0.5 (visited once)
- focus_score: 0.8 (parent of recent topic)
- diversity_score: 0.33
’ Priority: (0.25×2.0)+(0.5×2.5)+(0.5×2.0)+(0.8×1.5)+(0.33×1.0) = 4.78

Ranked list:
1. feel_energized (6.96)  Best!
2. aromatic_coffee (4.78)
3. ...
```

### Step 4: Select
```
Epsilon-greedy with 20% exploration (DEPTH phase)
Random number: 0.15 (< 0.20) ’ Pick randomly from top 5

Selected: aromatic_coffee (picked from top 5 randomly for variety)
```

### Step 5: Visit & Update
```
aromatic_coffee.visit_count: 1 ’ 2
aromatic_coffee.last_visit_turn: 1 ’ 3
focus_stack: [..., "feel_energized", "aromatic_coffee"]
```

### Step 6: Determine Strategy
```
visit_count = 2 (> 0)
out_degree = 1 (< 1? No! e 1)
’ Strategy: CONNECT_CONCEPTS
```

### Step 7: Generate Question
```
Input to LLM:
- Strategy: connect_concepts
- Target: "aromatic_coffee"
- Recent context: "The coffee makes me feel energized"

LLM generates:
"You mentioned feeling energizedhow does that aromatic quality
 play into that energy boost for you?"

Deduplication check: Not similar to last 5 questions 

Final question: "You mentioned feeling energizedhow does that
                 aromatic quality play into that energy boost for you?"
```

### Step 8: Present to Participant

Question logged with metadata:
- Strategy: connect_concepts
- Method: llm
- Generation time: 1.2s
- Turn: 3

---

## Key Terminology Glossary

| Term | Definition |
|------|------------|
| **Node** | A concept extracted from participant's response |
| **Edge** | A relationship connecting two nodes |
| **Graph** | The complete web of nodes and edges |
| **Schema** | The 4-level hierarchy of node types |
| **Richness** | Numerical measure of interview depth/breadth |
| **Coverage** | Percentage of schema types explored |
| **Opportunity** | A potential next question about a specific node |
| **Strategy** | The approach for a question (introduce/dig/connect) |
| **Phase** | Current interview stage (coverage/depth/connection/wrap-up) |
| **Visit count** | How many times a node has been questioned |
| **Out-degree** | Number of outgoing edges from a node (children) |
| **Priority score** | Combined ranking score for an opportunity |
| **Epsilon-greedy** | Strategy that occasionally picks random options |
| **LLM** | Large Language Model (AI text generator) |
| **Template** | Pre-written question format with blanks to fill |
| **Deduplication** | Checking if a question is too similar to recent ones |

---

## Why This Design?

**The Challenge:** Simple interviews either:
- Ask random questions (incoherent)
- Follow a script (rigid, misses unique insights)
- Go too shallow (surface-level only)
- Go too narrow (miss important areas)

**This System's Solution:**
1. **Graph structure** captures rich mental models
2. **Multiple scoring dimensions** balance breadth vs depth
3. **Phase progression** ensures systematic exploration
4. **Adaptive weights** shift priorities as interview progresses
5. **Strategy system** varies question types naturally
6. **LLM generation** creates natural conversation flow
7. **Deduplication** avoids repetitive questions

**Result:** Interviews that feel natural but systematically extract deep, comprehensive understanding.
