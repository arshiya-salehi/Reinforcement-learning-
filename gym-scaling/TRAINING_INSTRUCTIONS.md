# Training Instructions

This document provides instructions for training the RL agents and running evaluations.

## Prerequisites

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Install the gym_scaling package:
```bash
pip install -e .
```

## Training RL Agents

### Train DQN Agent

```bash
python src/train_dqn_sb3.py --pattern RANDOM --timesteps 200000
```

Options:
- `--pattern`: Load pattern (RANDOM, SINUSOIDAL, STEADY, SPIKE, POISSON, SINE_CURVE)
- `--timesteps`: Total training timesteps (default: 200000)
- `--name`: Custom model name (default: auto-generated with timestamp)

The trained model will be saved to `models/dqn_<pattern>_<timestamp>.zip`

### Train PPO Agent

```bash
python src/train_ppo_sb3.py --pattern RANDOM --timesteps 500000
```

Options:
- `--pattern`: Load pattern (RANDOM, SINUSOIDAL, STEADY, SPIKE, POISSON, SINE_CURVE)
- `--timesteps`: Total training timesteps (default: 500000)
- `--name`: Custom model name (default: auto-generated with timestamp)

The trained model will be saved to `models/ppo_<pattern>_<timestamp>.zip`

## Running Evaluation

### Full Comparative Evaluation

```bash
python src/evaluate.py --patterns STEADY SINUSOIDAL SPIKE POISSON --episodes 5
```

Options:
- `--patterns`: Load patterns to evaluate (default: STEADY SINUSOIDAL SPIKE POISSON)
- `--episodes`: Number of episodes per method (default: 5)
- `--steps`: Steps per episode (default: 1000)
- `--dqn-model`: Path to DQN model (default: models/dqn_model.zip)
- `--ppo-model`: Path to PPO model (default: models/ppo_model.zip)
- `--output`: Output directory for results (default: results/evaluation_<timestamp>)

Results will be saved to:
- JSON files with metrics: `results/evaluation_<timestamp>/results_<pattern>.json`
- Comparison plots: `results/evaluation_<timestamp>/plots_<pattern>/`

## Training Recommendations

For best results:

1. **Train on diverse load patterns**: Train on RANDOM or multiple patterns to improve generalization
2. **Training time**: 
   - DQN: ~30-60 minutes for 200K timesteps (depending on hardware)
   - PPO: ~2-4 hours for 500K timesteps (depending on hardware)
3. **Model checkpoints**: Checkpoints are saved automatically every 50K timesteps (DQN) or 100K timesteps (PPO)
4. **Best models**: Best models during training are saved to `models/best/`

## Quick Test

To quickly test the HPA autoscaler (no training required):

```bash
python src/hpa_autoscaler.py
```

To test static autoscaler:

```bash
python src/static_autoscaler.py
```

## Notes

- Training will create logs in the `logs/` directory
- TensorBoard logs for PPO are saved to `logs/ppo_tensorboard/`
- Evaluation results include statistical analysis (mean ± std) across multiple runs
- All visualizations are automatically generated during evaluation

