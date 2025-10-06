# MCTS with Large Language Models: Implementing RAP for Story Planning

## Project Overview

This project implements Monte Carlo Tree Search (MCTS) integrated with Large Language Models (LLMs) based on the RAP (Reasoning via Planning) framework. The implementation demonstrates the application of advanced AI planning techniques to automated story narrative generation, a novel domain distinct from the paper's original applications.

## Implementation Components

### 1. MCTS-UCT Foundation (`MCTS_UCT.ipynb`)
- Complete implementation of MCTS with UCB1 (Upper Confidence Bound) selection
- Proper four-phase algorithm: Selection, Expansion, Simulation, Backpropagation
- Demonstrated on Tic-Tac-Toe with performance analysis
- UCB1 formula: `exploitation + c * sqrt(ln(parent_visits)/visits)`

### 2. RAP-MCTS with LLM Integration (`RAP_MCTS.ipynb`)
- Implementation of the dual reward system from RAP paper
- LLM integration with hybrid mode for narrative coherence assessment
- Application to story planning domain
- Tree search with goal-directed planning

## Selected Paper

**Title:** Reasoning with Language Model is Planning with World Model  
**Authors:** Shibo Hao, Yi Gu, Haodi Ma, et al.  
**Conference:** EMNLP 2023 (Conference on Empirical Methods in Natural Language Processing)  
**Original Domains:** Blocksworld planning, Mathematical reasoning (GSM8k), Logical reasoning  

## Novel Application: Story Narrative Planning

Instead of the paper's original domains, this implementation applies RAP to **automated story planning**:

### Domain Specification
```python
StoryState:
  - characters: Dict[name -> attributes]
  - location: current scene location
  - plot_points: completed narrative events
  - tension_level: story tension (0-1)

StoryAction:
  - action_type: dialogue|move|conflict|reveal
  - actor: character performing action
  - target: interaction target
  - content: action description

StoryGoal:
  - required_plot_points: narrative milestones
  - target_relationships: character dynamics
  - target_location: destination scene
```

## Key Innovation: Dual Reward System

The RAP framework's core innovation is combining LLM assessment with task-specific heuristics:

### r₁: LLM Coherence Score
- Assesses narrative coherence of actions
- Considers character consistency
- Evaluates plot progression
- Implemented via hybrid LLM model

### r₂: Goal Achievement Heuristic
- Measures progress toward story goals
- Rewards plot point completion
- Evaluates relationship development
- Scores location achievements

### Combined Reward
```
reward = r₁^α × r₂^(1-α)
```
Where α balances narrative coherence vs goal achievement (default: 0.6)

## Implementation Architecture

### World Model (`ActualLLMWorldModel`)
Three operational modes:
1. **OpenAI Mode**: Direct GPT-3.5-turbo integration
2. **HuggingFace Mode**: Free API access to open models  
3. **Hybrid Mode**: Heuristic-based LLM simulation (used for experiments)

Key methods:
- `get_valid_actions()`: Generates narratively appropriate actions
- `calculate_action_likelihood()`: Computes r₁ coherence score
- `predict_next_state()`: Models story state transitions

### RAP-MCTS Algorithm
```python
for iteration in range(iterations):
    node = selection()      # UCB1 tree traversal
    node = expansion(node)   # Add new action via LLM
    reward = evaluation(node) # r₁ × r₂ calculation
    backpropagation(node, reward)
```

## Experimental Results

### Test Scenarios

| Scenario | Goal Components | Success | Path Length |
|----------|-----------------|---------|-------------|
| Hero's Journey | • 2 plot points<br>• 2 relationships<br>• Location: castle | ✗ | 2 steps |
| Mystery Resolution | • 2 revelations<br>• 1 tense relationship | ✓ | 3 steps |

### Performance Metrics

| Metric | Value |
|--------|-------|
| Success Rate | 50% (1/2 scenarios) |
| Average Path Length | 2.5 steps |
| Average Search Time | <0.01s |
| Model Type | Hybrid LLM |

### Generated Story Plans

**Mystery Resolution (Successful):**
1. Suspect1 reveal: makes important discovery
2. Detective reveal: makes important discovery  
3. Detective dialogue with Suspect2: confronts

Result: Achieved both revelations and tense relationship

**Hero's Journey (Partial Success):**
1. Hero reveal: makes important discovery
2. Hero conflict with Villain: confronts

Result: Completed plot points but missed location/relationship goals

## Comparison with RAP Paper

| Aspect | RAP Paper | This Implementation |
|--------|-----------|-------------------|
| **Domain** | Blocksworld, Math, Logic | Story Planning |
| **State Space** | Block configurations | Character dynamics & plot |
| **Action Space** | Stack/Unstack operations | Narrative actions |
| **LLM Usage** | LLaMA-33B, GPT-4 | Hybrid simulation |
| **Evaluation** | Task completion | Narrative coherence + goals |

## Technical Contributions

1. **Novel Domain Application**: First application of RAP to creative narrative generation
2. **Hybrid LLM Integration**: Cost-effective approach maintaining algorithmic integrity
3. **Multi-objective Planning**: Balances narrative quality with goal achievement
4. **Scalable Architecture**: Supports multiple LLM backends

## Challenges and Insights

### Challenges
- Story planning has larger action space than Blocksworld
- Multiple simultaneous goals increase complexity
- Narrative coherence harder to quantify than logical correctness

### Insights
- Dual reward system effectively balances creativity with goal-directed planning
- MCTS exploration helps discover non-obvious narrative paths
- Hybrid mode sufficient for demonstrating algorithmic concepts

## Running the Implementation

```python
# Initialize world model (hybrid mode - no API required)
world_model = ActualLLMWorldModel(model_type="hybrid")

# Configure RAP-MCTS
rap_mcts = RAP_MCTS(
    world_model=world_model,
    iterations=30,
    alpha=0.6  # Balance coherence vs goals
)

# Run planning
best_path = rap_mcts.search(initial_state, goal)
```

## Future Work

- Fine-tune LLMs specifically for narrative coherence
- Expand to multi-character perspective planning
- Incorporate reader engagement metrics
- Scale to novel-length narrative planning

## Conclusion

This implementation successfully demonstrates the RAP framework's applicability beyond its original domains. By applying MCTS with LLM integration to story planning, we show that the dual reward system can effectively balance creative generation with structured goal achievement. The 50% success rate, while lower than deterministic planning domains, reflects the increased complexity of narrative planning where multiple soft constraints must be satisfied simultaneously.

## References

Hao, S., Gu, Y., Ma, H., Hong, J. J., Wang, Z., Wang, D. Z., & Hu, Z. (2023). Reasoning with Language Model is Planning with World Model. *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP)*.

