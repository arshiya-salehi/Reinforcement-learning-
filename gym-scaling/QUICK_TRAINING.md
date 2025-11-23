# Quick Training Options

If you're running short on time, here are faster training configurations:

## Option 1: Reduced Timesteps (Faster but potentially lower quality)

### Quick DQN Training (~15-30 minutes)
```bash
python src/train_dqn_sb3.py --pattern RANDOM --timesteps 100000 --name dqn_model
```
- Reduces from 200K to 100K timesteps
- May still produce reasonable results

### Quick PPO Training (~1-2 hours)
```bash
python src/train_ppo_sb3.py --pattern RANDOM --timesteps 250000 --name ppo_model
```
- Reduces from 500K to 250K timesteps
- PPO typically needs more training than DQN

## Option 2: Very Fast Training (Demo Only)

### Minimal DQN (~5-10 minutes)
```bash
python src/train_dqn_sb3.py --pattern RANDOM --timesteps 50000 --name dqn_model
```
- 50K timesteps - enough for basic demonstration
- May not learn optimal policy

### Minimal PPO (~30-60 minutes)
```bash
python src/train_ppo_sb3.py --pattern RANDOM --timesteps 100000 --name ppo_model
```
- 100K timesteps - minimum for PPO
- Results may be suboptimal

## Option 3: Skip Training, Use Baselines Only

For the project submission, you can demonstrate using **only HPA and Static baselines** without training RL models:

```bash
# This will only evaluate HPA and Static (no RL models needed)
python src/evaluate.py --patterns STEADY SINUSOIDAL SPIKE POISSON --episodes 5
```

The notebook (`project.ipynb`) will still work - it will just show HPA results.

## Recommended Approach

**If you have 2-4 hours:**
- Use Option 1 (reduced timesteps)
- Should produce acceptable results for comparison

**If you're in a rush (< 1 hour):**
- Use Option 3 (baselines only)
- Still demonstrates the evaluation framework
- Can note in report that RL training was not completed due to time constraints

**If you have time for full training:**
```bash
# Full training (recommended for best results)
python src/train_dqn_sb3.py --pattern RANDOM --timesteps 200000 --name dqn_model
python src/train_ppo_sb3.py --pattern RANDOM --timesteps 500000 --name ppo_model
```

## Monitoring Training Progress

Training will show progress every 10 steps. You can:
- Watch the console output for episode rewards
- Check logs in `logs/` directory
- Stop training early if results look good (Ctrl+C)

## Checkpoint Recovery

If training is interrupted, you can:
- Use checkpoint files in `models/checkpoints/`
- Or use best model saved in `models/best/`
- Just copy to main models directory: `cp models/best/dqn_model_* models/dqn_model.zip`

