# CartPole Agents

This project implements random and DQN agents for the OpenAI Gymnasium CartPole-v1 environment, with a modular structure for future research and development.

## Structure

- `agents/`: Agent implementations (Random, DQN, etc.)
- `core/`: Core RL utilities (replay buffer, trainer)
- `configs/`: Config files for hyperparameters
- `experiments/`: Scripts to run experiments
- `utils/`: Logging and utilities
- `tests/`: Unit tests

## Getting Started

1. Install dependencies:
```
pip install -r requirements.txt
```

## Extending

Add new agents to `agents/`.

# Usage

Run any agent with: 
```
python -m cartpole.experiments.run_experiment --agent <agent_name> --config <config_path> [--env <env_id>]
```

Examples:
```sh
python -m cartpole.experiments.run_experiment --agent dqn --config configs/dqn_config.yaml
```
```sh
python -m cartpole.experiments.run_experiment --agent reinforce --config configs/reinforce_config.yaml
```
```sh
python -m cartpole.experiments.run_experiment --agent random --config configs/dqn_config.yaml
```

## Video Recording

To record videos of the agent at the beginning, middle, and end of training, use:

```sh
python -m cartpole.experiments.run_experiment --agent dqn --config configs/dqn_config.yaml --video_episodes 0,249,499 --show_plot
```

- Videos will be saved in the `videos/` directory by default.
- You can specify any episodes to record (e.g., `--video_episodes 0,99,199,299,399,499`).
- Use negative indices for counting from the end (e.g., `--video_episodes 0,-1` for first and last).
