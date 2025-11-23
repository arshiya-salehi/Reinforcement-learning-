"""
Configuration file for training and evaluation
"""
import os

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Training hyperparameters
DQN_TOTAL_TIMESTEPS = 200000
PPO_TOTAL_TIMESTEPS = 500000

# DQN hyperparameters
DQN_CONFIG = {
    'learning_rate': 1e-4,
    'buffer_size': 100000,
    'learning_starts': 1000,
    'batch_size': 32,
    'tau': 1.0,  # Target network update rate
    'gamma': 0.99,
    'train_freq': 4,
    'gradient_steps': 1,
    'target_update_interval': 1000,
    'exploration_fraction': 0.1,
    'exploration_initial_eps': 1.0,
    'exploration_final_eps': 0.05,
    'policy_kwargs': dict(net_arch=[128, 128])
}

# PPO hyperparameters
PPO_CONFIG = {
    'learning_rate': 3e-4,
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.01,
    'vf_coef': 0.5,
    'policy_kwargs': dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])])
}

# Environment parameters
ENV_CONFIG = {
    'max_instances': 100.0,
    'min_instances': 2.0,
    'capacity_per_instance': 87,
    'change_rate': 100,  # How often load changes (in steps)
}

# Evaluation parameters
EVAL_EPISODES = 5  # Number of episodes to run per evaluation
EVAL_STEPS = 1000  # Steps per evaluation episode

# HPA thresholds
HPA_CONFIG = {
    'threshold_high': 70,  # Scale up if load > 70%
    'threshold_low': 40,   # Scale down if load < 40%
    'cooldown_period': 10, # Steps to wait before allowing another scaling action
}

# Random seed for reproducibility
RANDOM_SEED = 42

