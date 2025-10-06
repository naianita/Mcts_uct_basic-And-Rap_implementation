# MCTS-UCT Implementation

## Overview
This notebook implements the Monte Carlo Tree Search with Upper Confidence bounds applied to Trees (MCTS-UCT) algorithm from scratch, demonstrating its application on Tic-Tac-Toe.

## Algorithm Components

### Core MCTS Phases
1. **Selection**: Navigate from root to leaf using UCB1 formula
2. **Expansion**: Add one new child node to the tree
3. **Simulation**: Random rollout from leaf to terminal state
4. **Backpropagation**: Update statistics from leaf to root

### Key Features
- UCB1 formula balancing exploration vs exploitation
- Configurable exploration constant (c = √2 by default)
- Support for two-player zero-sum games
- Visualization of tree structure and action statistics

## Implementation Details

### Classes
- `State`: Abstract base class for game states
- `MCTSNode`: Tree node with visit counts and value estimates
- `MCTS_UCT`: Main algorithm implementation
- `TicTacToeState`: Concrete implementation for Tic-Tac-Toe

### Parameters
- `exploration_constant`: Controls exploration vs exploitation (default: 1.414)
- `iteration_limit`: Number of MCTS iterations per move
- `time_limit`: Optional time constraint for search

## Results
- Successfully plays optimal Tic-Tac-Toe
- Demonstrates proper tree exploration
- Shows convergence to optimal moves with sufficient iterations

## Usage
```python
mcts = MCTS_UCT(iteration_limit=1000)
best_action = mcts.search(initial_state)