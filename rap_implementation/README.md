## **2. RAP_MCTS_Story_Planning.md**
```markdown
# RAP-MCTS for Story Planning

## Overview
Implementation of "Reasoning with Language Model is Planning with World Model" (RAP) from EMNLP 2023, applied to automated story planning instead of the paper's original Blocksworld task.

## Paper Reference
**Title**: Reasoning with Language Model is Planning with World Model  
**Conference**: EMNLP 2023  
**Authors**: Hao et al.  
**Original Task**: Blocksworld planning, Math reasoning  
**Our Task**: Story narrative planning

## Key Innovations from RAP

### 1. World Model
- LLM predicts state transitions deterministically
- No randomness in state evolution (fixed from original)

### 2. Dual Reward System
- **r1**: Action likelihood (narrative coherence)
- **r2**: Task-specific heuristic (goal achievement)
- Combined: `reward = r1^α × r2^(1-α)`

### 3. No Random Rollouts
- Uses LLM evaluation instead of Monte Carlo simulation
- Direct state evaluation at leaf nodes

## Implementation Components

### Story Domain
- **States**: Characters, locations, plot points, tension levels
- **Actions**: Dialogue, movement, conflict, revelations
- **Goals**: Target plot points, relationships, locations

### Classes
- `StoryState`: Narrative situation representation
- `StoryAction`: Narrative events
- `StoryGoal`: Target story conditions
- `LLMWorldModel`: Simulated LLM for state transitions
- `RAP_MCTS`: Main algorithm with dual rewards

### Improvements Over Original
1. Deterministic action generation (removed randomness)
2. Goal-directed action prioritization
3. Enhanced reward function for narrative goals
4. Cycle detection in path extraction

## Test Scenarios

### 1. Hero's Journey
- Goal: Reach castle, complete revelation and conflict
- Success Rate: 0% (location goal challenging)

### 2. Mystery Resolution
- Goal: Two revelations, tense relationship
- Success Rate: 100% (consistently solved)

### 3. Romance Arc
- Goal: Conflict with rival, friendly relationships
- Success Rate: 0% (relationship goals partially met)

## Results

### RAP-MCTS Performance
- Overall Success: 33.3% (1/3 scenarios)
- Average Path Length: 2-3 steps
- Nodes Explored: 50-100 per search

### Comparison with Baseline
- Greedy Baseline: 100% success rate
- RAP-MCTS: Better exploration but lower success
- Trade-off: Exploration vs exploitation

## Key Findings

### Successes
- Successfully implements RAP's dual reward system
- Demonstrates structured exploration via MCTS
- Solves Mystery Resolution consistently

### Challenges
- Location goals often missed
- Bidirectional relationships not established
- UCB exploration may distract from goals

## Usage
```python
world_model = LLMWorldModel()
rap_mcts = RAP_MCTS(
    world_model=world_model,
    iterations=100,
    alpha=0.5
)
best_plan = rap_mcts.search(initial_state, goal)