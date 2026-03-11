# L-SDA: Large Language Model Guided Monte Carlo Tree Search for Decision-Making in Autonomous Driving

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)

**L-SDA** is an intelligent decision-making framework for autonomous driving that combines the strategic reasoning capabilities of Large Language Models (LLMs) with the exploration efficiency of Monte Carlo Tree Search (MCTS).  
It is designed for highway scenarios where the ego vehicle must make safe and efficient decisions under dense, interactive traffic.

---

## ✨ Key Features

- 🤖 **LLM-Guided Decision Making**  
  Uses an LLM to provide high-level priors and explanations for action selection.

- 🌳 **MCTS Integration**  
  Combines LLM guidance with MCTS to balance exploration and exploitation.

- ⏱ **TTC Safety Mechanism**  
  A Time-To-Collision (TTC) based safety layer that switches to conservative LLM decisions when collision risk is high.

- 💭 **Reflection Module**  
  A reflection mechanism that learns from past mistakes and updates memories for future episodes.

- 🧠 **Memory System**  
  ChromaDB-based vector storage for efficient retrieval of relevant past experiences.

- 🎯 **Critical Vehicle Detection**  
  Identifies and focuses on the most relevant surrounding vehicles for decision-making.

- 📊 **Comprehensive Visualization**  
  An interactive visualization tool to inspect trajectories, decisions, and safety events.

---

## 📁 Project Structure

A recommended high-level structure for this repository is:

- `lsda/` – Core implementation of the L-SDA framework  
  - `driver_agent/` – L-SDA agent, reflection, memory interfaces  
  - `llm_mcts/` – LLM-MCTS prompts, policies, and utilities (`llm_prompts`, `llm_mcts_utils`, `llm_mcts_policies`)  
  - `scenario/` – Environment wrappers, replay, plotting, DB bridge

- `scripts/` – Entry-point scripts (recommended destination for top-level Python files)  
  - `run_lsda.py` – Main entry point to run L-SDA simulations  
  - `run_experiments.py` – Batch experiments / ablations  
  - `summary_experiment.py` – Summarize experiment results  
  - `visualize_results.py` – Visualization of recorded episodes  
  - `ttc_safety_mechanism.py` – TTC-related analysis and utilities  
  - `manage_memories.py`, `view_memory.py`, `manage_vector_db.py`, `compact_database.py` – Memory and vector DB management tools

- `third_party/` – Optional directory for external libraries  
  - `rl-agents/` – RL Agents codebase used for tree search components (moved from `rl-agents-master/`)

- `configs/` – Configuration files (optional, e.g. `config.yaml`)

- `docs/` – Additional documentation (optional)  
  - `DIRECTORY_STRUCTURE.md` – Detailed description of output directory structure  
  - `TTC_SAFETY_GUIDE.md` – Additional details on TTC safety mechanism

- `requirements.txt` – Python dependencies  
- `README.md` – This file  
- `LICENSE` – License file (MIT)

> Note: The exact layout may differ in your local copy. The structure above is the recommended organization for a clean public release.

---

## 🚀 Getting Started

### 1. Requirements

- **Python**: 3.8 or higher

Install all dependencies using:

```bash
pip install -r requirements.txt
```

**Main Dependencies** (subset):

- `gymnasium==0.28.1` – OpenAI Gym interface  
- `highway-env==1.8.2` – Highway driving simulation environment  
- `langchain==0.0.335` – LLM integration framework  
- `openai==0.28.1` – OpenAI API client  
- `chromadb==0.3.29` – Vector database for experience memory  
- `gradio==3.36.0` – Web UI for visualization  
- `matplotlib`, `numpy`, `PyYAML` – Scientific computing and configuration

---

## 2. Configuration ⚙️

The main configuration file is `config.yaml` (you may keep it at the repository root or under `configs/`).

### LLM Configuration

Set your OpenAI API credentials via environment variables or directly in the config file (do **not** commit real keys to GitHub):

```yaml
# OpenAI API Configuration
OPENAI_KEY: your-openai-api-key
OPENAI_CHAT_MODEL: 'gpt-3.5-turbo-0125'
OPENAI_API_BASE: 'https://api.openai.com/v1'
```

### Core Settings

```yaml
# L-SDA Agent Settings
reflection_module: False        # Enable/disable reflection learning
few_shot_num: 5                 # Number of few-shot examples (0 for zero-shot)
memory_path: 'memories/20_mem'  # Path to memory storage

# Highway Environment Settings
simulation_duration: 30         # Simulation duration in seconds
episodes_num: 3                 # Number of episodes to run
vehicles_density: 3             # Traffic density
lanes_count: 6                  # Number of highway lanes
vehicle_count: 15               # Total number of vehicles

# LLM-MCTS Integration
enable_llm_prior_guidance: True           # Use LLM for action priors
use_llm_critical_vehicle_guidance: False  # Use LLM for vehicle filtering
llm_critical_vehicle_max_count: 5         # Max critical vehicles to consider

# MCTS Configuration
mcts_budget: 100               # Number of MCTS simulations per decision
step_deterministic: True       # Deterministic decisions
display_tree: False            # Visualize MCTS tree during execution
```

---

## 3. Running L-SDA

Run the main simulation:

```bash
python run_lsda.py
```

This will:

1. Create a **timestamped run directory** under `results/`  
2. Initialize the highway environment with the configured parameters  
3. Run the L-SDA decision-making process for the specified number of episodes  
4. Save all outputs (logs, databases, videos, statistics) into the run directory

### Example console output

```text
🚀 L-SDA is starting (Episodes: 20)
📁 Run directory: results/run_20251127_143052_ep20_lane5_density3
📝 Output file: results/run_20251127_143052_ep20_lane5_density3/output.txt
💡 All logs are saved to output.txt; nothing is printed to the console during the run.
------------------------------------------------------------
```

---

## 4. Output Directory Structure

Each run creates an isolated directory under `results/`:

```text
results/
└── run_20251127_143052_ep20_lane5_density3/  ← Timestamped run directory
    ├── output.txt                            ← Complete output log file
    ├── statistics.json                       ← Statistics data
    ├── statistics_report.txt                 ← Human-readable report
    ├── highway_0.db                          ← Episode 0 database
    ├── highway_0-episode-0.mp4               ← Episode 0 video
    ├── highway_1.db                          ← Episode 1 database
    ├── highway_1-episode-0.mp4               ← Episode 1 video
    └── tb_logs/                              ← TensorBoard logs (optional)
```

**Benefits:**

- ✅ Each run is isolated – no file overwrites  
- ✅ Easy to compare different runs  
- ✅ Complete history is preserved  
- ✅ Directory name encodes configuration information

For more detailed documentation, see `DIRECTORY_STRUCTURE.md` (optional).

### Viewing Output

```bash
# Monitor output in real time
tail -f results/run_*/output.txt

# View a specific run
cat results/run_20251127_143052_ep20_lane5_density3/output.txt
```

---

## 5. Reflection Module

The reflection module allows the agent to learn from past mistakes and improve over time.

1. Set `reflection_module: True` in `config.yaml`  
2. Run simulations – experiences will be automatically collected  
3. New memory items will be written into the memory storage (e.g. `memories/...`)  
4. Future runs will retrieve and use these memories via the vector store

---

## 6. TTC Safety Mechanism

The Time-To-Collision (TTC) safety mechanism provides an additional safety layer by switching to LLM decisions when collision risk is high.

### Configuration (in `config.yaml`)

```yaml
enable_ttc_safety: True    # Enable TTC safety mechanism
ttc_threshold: 3.0         # TTC threshold (seconds)
ttc_verbose: True          # Print detailed TTC information
```

### How It Works

- Compute TTC for all nearby vehicles:  
  \[
  \text{TTC} = \frac{\|P_{\text{ego}} - P_{\text{target}}\|}{v_{\text{ego}} - v_{\text{target}}}
  \]
- When TTC < threshold: use the **LLM** decision (safety-critical mode)  
- When TTC ≥ threshold: use the **MCTS** decision (normal operation)

The final applied action is:

\[
a^{\text{Applied}} = (1-I) \cdot a^{\text{MCTS}} + I \cdot a^{\text{LLM}},
\]

where \( I = 1 \) if TTC < threshold, and \( I = 0 \) otherwise.

For more details, see `TTC_SAFETY_GUIDE.md` (optional).

---

## 7. Visualization

We provide an interactive visualization tool to inspect recorded episodes:

```bash
python visualize_results.py -r results/highway_1.db
```

You can adapt this command to point to any `.db` file generated in the `results/` directory.

---

## 8. Reproducibility & Environment

- Use the provided `requirements.txt` to create a virtual environment.  
- We recommend excluding local virtual environments (`.venv/`) and large result directories (`results/`, `memories/`, `tb_logs/`) from version control.  
- If you need fully reproducible experiments, consider pinning package versions as done in `requirements.txt` and sharing a minimal example configuration.

---

## 9. Citation

If you use L-SDA in your research, please consider citing:

```bibtex
@article{lsda2025,
  title   = {L-SDA: Large Language Model Guided Monte Carlo Tree Search for Decision-Making in Autonomous Driving},
  author  = {...},
  journal = {...},
  year    = {2025}
}
```

(Replace with the actual citation when available.)

---

## 10. License

This project is released under the **MIT License**.  
See the `LICENSE` file for details.

