# CartPole Agents

This project implements random, DQN, and REINFORCE agents for the OpenAI Gymnasium CartPole-v1 environment.

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

> **Note:** You do not need to install this project as a package or use `setup.py` for local development. All experiments can be run directly from the command line.

## Extending

Add new agents to `agents/`.

# Usage

Run any agent with: 
```
python experiments/run_experiment.py --agent <agent_name> --config <config_path> [--env <env_id>]
```

Examples:
```sh
python experiments/run_experiment.py --agent dqn --config configs/dqn_config.yaml
```
```sh
python experiments/run_experiment.py --agent reinforce --config configs/reinforce_config.yaml
```
```sh
python experiments/run_experiment.py --agent random --config configs/dqn_config.yaml
```

## Video Recording

To record videos of the agent at the beginning, middle, and end of training, use:

```sh
python experiments/run_experiment.py --agent dqn --config configs/dqn_config.yaml --record_videos --video_episodes 0,249,499 --show_plot
```

- Videos will be saved in the `videos/` directory by default.
- `--record_videos` must be included to record videos.
- You can specify any episodes to record (e.g., `--video_episodes 0,99,199,299,399,499`).
- Use negative indices for counting from the end (e.g., `--video_episodes 0,-1` for first and last).
