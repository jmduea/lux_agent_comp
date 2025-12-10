# Lux AI Season 3 Challenge - Deep Space Relic Explorer

![Python](https://img.shields.io/badge/python-3.10+-blue)
![JAX](https://img.shields.io/badge/JAX-enabled-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![Competition](https://img.shields.io/badge/Kaggle-Lux%20AI%20S3-20BEFF)

A sophisticated multi-agent reinforcement learning system for the [Kaggle Lux AI Season 3 Challenge](https://www.kaggle.com/competitions/lux-ai-season-3), featuring custom pathfinding algorithms, Deep Q-Networks, and strategic exploration-exploitation balance for competitive gameplay in a dynamic space environment.

## 🚀 Project Overview

This project implements intelligent agents capable of competing in the Lux AI Season 3 challenge, where two teams control units in deep space to explore ancient relics, harvest energy, and score points across best-of-5 match sequences. The challenge features:

- **Dynamic Environments**: Procedurally generated 24x24 maps with asteroids, nebulae, and energy nodes
- **Fog of War**: Limited vision based on unit positions and environmental factors
- **Strategic Depth**: Balance between exploration in early matches and exploitation in later matches
- **Randomized Mechanics**: Game parameters vary between matches, requiring adaptive strategies

## ✨ Key Features

### Advanced Agent Implementations
- **DQN-Based Agent**: Deep Q-Network implementation using JAX for high-performance training
- **Q-Learning Agent**: Traditional reinforcement learning with experience replay and exploration strategies
- **Relic Bot Base**: Specialized pathfinding and relic discovery agent
- **Defensive Sapper**: Strategic agent with defensive positioning and energy sapping capabilities

### Technical Highlights
- **JAX Integration**: High-performance numerical computing with GPU acceleration support
- **Custom Space Mapping**: Efficient fog-of-war tracking and terrain analysis using JAX arrays
- **Pathfinding System**: A* pathfinding with dynamic obstacle avoidance
- **Modular Architecture**: Clean separation between agent logic, environment interaction, and utilities
- **Comprehensive Testing**: Unit tests for core functionality with pytest
- **Replay System**: Episode recording for post-match analysis and debugging

## 🏗️ Technical Architecture

```
lux_agent_comp/
├── core/                    # Core agent framework
│   ├── base_agent.py       # Abstract base agent with DQN support
│   ├── space.py            # Space representation and fog-of-war tracking
│   ├── pathfinding.py      # A* pathfinding implementation
│   ├── node.py             # Node structure for pathfinding
│   └── debug.py            # Visualization tools
├── lux/                    # Lux AI environment integration
│   ├── kit.py              # Lux AI kit utilities
│   └── utils.py            # Helper functions
├── agent.py                # Main agent implementation
├── dqn_agent.py           # Deep Q-Network agent
├── main.py                # Kaggle submission entry point
├── evaluate_agents.py     # Agent evaluation and testing
├── test_env.py            # Environment testing utilities
├── profiling.py           # Performance profiling
├── saved_agents/          # Version-controlled agent snapshots
│   ├── q_learning_agent/
│   ├── relic_bot_base/
│   └── relic_bot_defensive_sapper/
└── tests/                 # Unit and integration tests
```

## 🛠️ Technologies Used

| Category | Technologies |
|----------|-------------|
| **Core ML/RL** | JAX, Flax, Stable-Baselines3, Ray RLlib, OpenRL |
| **Environment** | Gymnasium, PettingZoo, luxai-s3 |
| **Visualization** | Matplotlib, Seaborn, Plotly, TensorBoard, Weights & Biases |
| **Utilities** | NumPy, Pygame (rendering), Beautiful Soup |
| **Testing** | pytest, pytest-cov |
| **Package Management** | uv, pip |

## 📦 Installation

### Prerequisites
- Python 3.10 or higher
- (Optional) CUDA-capable GPU for JAX acceleration

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/jmduea/lux_agent_comp.git
   cd lux_agent_comp
   ```

2. **Install dependencies using uv** (recommended)
   ```bash
   # Install uv if you don't have it
   pip install uv
   
   # Install project dependencies
   uv sync
   ```

   Or using pip:
   ```bash
   pip install -e .
   ```

3. **Install development dependencies** (for testing)
   ```bash
   uv sync --group dev
   ```

## 🎮 Usage

### Running the Main Agent

```bash
python main.py
```

This agent is designed to work with the Kaggle submission system and reads input from stdin following the Lux AI protocol.

### Evaluating Agents

Compare different agent implementations:

```bash
python evaluate_agents.py
```

This script runs multiple games between agents and saves replays to the `replays/` directory for analysis.

### Testing the Environment

```bash
python test_env.py
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=core --cov=lux

# Run specific test files
pytest tests/core/test_space.py
```

### Training a DQN Agent

```bash
python dqn_agent.py
```

Monitor training progress using TensorBoard or Weights & Biases integration.

### Creating a Kaggle Submission

```bash
python create_submission.py
```

This packages your agent and dependencies into a submission-ready format for Kaggle.

## 🎯 Agent Implementations

### 1. Main Agent (`agent.py`)
The primary competition agent featuring:
- Relic exploration with unknown tile targeting
- Energy-aware movement decisions
- Dynamic action selection based on game state
- JAX-accelerated state processing

### 2. DQN Agent (`dqn_agent.py`)
Deep reinforcement learning agent with:
- Flax-based neural network architecture
- Experience replay buffer
- Target network for stable training
- Epsilon-greedy exploration

### 3. Saved Agent Variants
- **Q-Learning Agent**: Traditional Q-learning with custom reward shaping
- **Relic Bot Base**: Specialized relic discovery and collection strategy
- **Defensive Sapper**: Focuses on defensive positioning and opponent disruption

## 📊 Development Workflow

1. **Development**: Implement and test agents locally
2. **Evaluation**: Run agents against baseline implementations
3. **Profiling**: Identify performance bottlenecks using `profiling.py`
4. **Visualization**: Analyze game replays and agent behavior
5. **Iteration**: Refine strategy based on evaluation results
6. **Submission**: Package and submit to Kaggle

## 🎓 Key Learnings

- **Exploration vs Exploitation**: Early match exploration is crucial for mapping relic positions and energy distributions
- **JAX Performance**: Using JAX for state representation provides significant performance improvements
- **Pathfinding**: Custom A* implementation handles dynamic obstacles (asteroids, nebulae) efficiently
- **Vision Management**: Proper fog-of-war tracking is essential for strategic decision-making
- **Modular Design**: Separating concerns allows for rapid agent iteration and testing

## 🔮 Future Improvements

- [ ] Implement Monte Carlo Tree Search (MCTS) for strategic planning
- [ ] Add multi-agent coordination strategies
- [ ] Enhance opponent prediction and modeling
- [ ] Optimize energy harvesting algorithms
- [ ] Implement adaptive strategy selection based on game parameters
- [ ] Add comprehensive reward shaping for RL training
- [ ] Integrate transformer-based models for sequence prediction

## 🤝 Contributing

This is a personal competition project, but feedback and suggestions are welcome! Feel free to open issues for discussion.

## 📄 License

This project is open source and available for educational and portfolio purposes.

## 🙏 Acknowledgments

- [Lux AI Challenge Team](https://github.com/Lux-AI-Challenge) for creating an engaging competition
- Kaggle community for strategies and insights
- JAX and Flax teams for excellent ML frameworks

## 📚 Resources

- [Lux AI Season 3 Specifications](https://github.com/Lux-AI-Challenge/Lux-Design-S3)
- [Competition Page](https://www.kaggle.com/competitions/lux-ai-season-3)
- [Lux AI Documentation](https://github.com/Lux-AI-Challenge/Lux-Design-S3/tree/main/kits)
- [Lux AI Discord Community](https://discord.gg/aWJt3UAcgn)

---

**Author**: Jon Duea ([jdueadev@gmail.com](mailto:jdueadev@gmail.com))

**Repository**: [github.com/jmduea/lux_agent_comp](https://github.com/jmduea/lux_agent_comp)
