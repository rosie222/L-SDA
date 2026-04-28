# L-SDA: LLM-Guided MCTS for Autonomous Driving

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)

**L-SDA** is an intelligent decision-making framework for autonomous driving that combines strategic reasoning from Large Language Models (LLMs) with the exploration efficiency of Monte Carlo Tree Search (MCTS). It is designed for highway scenarios where the ego vehicle must make safe and efficient decisions under dense, interactive traffic.

## ✨ Key Features

- 🤖 **LLM-Guided Decision Making** — High-level priors and explanations for action selection
- 🌳 **MCTS Integration** — Balance exploration and exploitation with tree search
- ⏱ **TTC Safety Mechanism** — Time-To-Collision based safety switching
- 💭 **Reflection Module** — Learn from past mistakes and update memories
- 🧠 **Memory System** — ChromaDB-based vector storage for experience retrieval
- 🎯 **Critical Vehicle Detection** — Focus on most relevant surrounding vehicles
- 📊 **Visualization** — Interactive trajectory and decision inspection

## 📁 Project Structure

```
L-SDA/
├── lsda/                          # Core L-SDA framework
│   ├── driver_agent/              # Agent, reflection, memory
│   ├── llm_mcts/                  # LLM-MCTS prompts & policies
│   └── scenario/                  # Environment wrappers & plotting
├── scripts/                       # Entry-point scripts
│   ├── run_lsda.py               # Main simulation runner
│   ├── run_experiments.py         # Batch experiments/ablations
│   ├── visualize_results.py       # Episode visualization
│   ├── manage_memories.py         # Memory utilities
│   └── ...
├── rl-agents-master/              # Tree search components (MCTS)
├── config.yaml                    # Main configuration file
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Requirements

- **Python**: 3.10+ (tested with 3.10, 3.11)
- **macOS/Linux/Windows** with standard development tools

> **Note**: Python 3.14 is **not** currently compatible due to legacy pinned dependencies (e.g., `numpy==1.24.3`). For best compatibility, use Python 3.10 or 3.11.

### 2. Installation

```bash
# Clone the repository
git clone <repo-url>
cd L-SDA

# Create virtual environment (example with Python 3.11)
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install pip tools
python -m pip install --upgrade pip setuptools wheel

# Install project dependencies
python -m pip install -r requirements.txt
```

### 3. Configuration

Edit `config.yaml` to set:

```yaml
# OpenAI API Configuration
OPENAI_KEY: your-openai-api-key

# Highway Environment
episodes_num: 20              # Number of episodes to run
simulation_duration: 30       # Duration in seconds
vehicles_density: 3           # Traffic density (1-5)

# L-SDA Settings
reflection_module: False      # Enable reflection learning
few_shot_num: 5              # Few-shot examples (0 for zero-shot)
enable_ttc_safety: True      # TTC safety mechanism
ttc_threshold: 3.0           # TTC threshold (seconds)

# MCTS
mcts_budget: 100             # Simulations per decision
enable_llm_prior_guidance: True  # Use LLM for priors
```

### 4. Running L-SDA

```bash
# Activate virtual environment if needed
source .venv/bin/activate

# Run the main simulation
python scripts/run_lsda.py
```

This will:
1. Create a timestamped run directory under `results/`
2. Initialize highway environment with configured parameters
3. Run L-SDA for specified episodes
4. Save logs, databases, videos, and statistics

**Example output:**
```
🚀 LSDA starts running (Episodes: 20)
📁 Created new run directory: results/run_20260323_174540
------------------------------------------------------------
```

## 📋 Common Commands

```bash
# Run batch experiments/ablations
python scripts/run_experiments.py

# Summarize experiment results
python scripts/summary_experiment.py

# Visualize recorded episodes
python scripts/visualize_results.py -r results/highway_0.db

# View/manage memories
python scripts/view_memory.py
python scripts/manage_memories.py

# Manage vector database
python scripts/manage_vector_db.py
python scripts/compact_database.py
```

## 📊 Output Structure

Each run creates an isolated directory:

```
results/
└── run_20260323_174540/
    ├── output.txt                 # Complete log
    ├── statistics.json            # Results data
    ├── statistics_report.txt      # Human-readable report
    ├── highway_0.db              # Episode database
    ├── highway_0-episode-0.mp4   # Episode video
    └── ...
```

## 🔧 Troubleshooting

**Can't find `lsda` module:**
- Ensure you're running from project root with activated venv
- Check that `sys.path` includes the project root

**ModuleNotFoundError for `numba`, `tensorboard`, etc:**
- Install the missing package manually:
  ```bash
  python -m pip install numba
  python -m pip install tensorboard torch  # Optional for TensorBoard
  ```

**Python version incompatibility:**
- Use Python 3.10 or 3.11, NOT 3.14+
- Create a new venv with correct Python version:
  ```bash
  python3.11 -m venv .venv
  ```

## 📖 Core Components

### LLM Agent (`lsda/driver_agent/`)
- Decision-making with LLM priors
- Reflection mechanism for learning
- Memory-augmented decision processes

### MCTS Engine (`rl-agents-master/`)
- Monte Carlo Tree Search implementation
- Action value estimation and UCB selection
- Tree visualization (optional)

### Environment (`lsda/scenario/`)
- Highway-env wrapper
- State/action abstractions
- Trajectory recording and plotting

## 🎯 Example Workflow

1. **Configure** `config.yaml` with your parameters
2. **Run simulation** with `python scripts/run_lsda.py`
3. **Analyze results** from `results/` directory
4. **Visualize** episodes with `python scripts/visualize_results.py`
5. **Iterate** on reflection module or MCTS settings

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions are welcome! Please follow the existing code structure and comment your changes.

## 📚 Citation

If you use L-SDA in research, please cite:

```bibtex
@article{lsda2026,
  title   = {L-SDA: LLM-Guided Monte Carlo Tree Search for Decision-Making in Autonomous Driving},
  author  = {Yanxi Luo, Yunxiao Shan},
  journal = {...},
  year    = {2026}
}
```

## 📞 Support

For bugs, questions, or suggestions, open an issue on GitHub.

---

**Python Version:** 3.10–3.11 (tested)  
**Status:** Stable
