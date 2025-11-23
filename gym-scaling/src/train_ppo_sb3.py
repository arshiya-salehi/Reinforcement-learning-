"""
Train PPO agent using stable-baselines3
"""
import os
import sys
import numpy as np

# Compatibility shim for numpy 2.x (gym expects np.bool8 which was removed)
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

import gymnasium as gym
import gym_scaling
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import PPO_TOTAL_TIMESTEPS, PPO_CONFIG, MODELS_DIR, LOGS_DIR, RANDOM_SEED
from src.load_generators import LOAD_PATTERNS
from src.env_wrapper import ScalingEnvWrapper
from gym_scaling.envs.scaling_env import INPUTS


def create_env_with_load_pattern(load_pattern='RANDOM'):
    """Create environment with specified load pattern"""
    def _init():
        env = gym.make('Scaling-v0')
        env = ScalingEnvWrapper(env)
        # Set up load pattern if it's one of our custom patterns
        if load_pattern in LOAD_PATTERNS:
            pattern = LOAD_PATTERNS[load_pattern]
            # Create a wrapper function that matches the expected signature
            def load_func(step, max_influx, offset):
                return pattern['function'](step, max_influx, offset, **pattern['options'])
            env.scaling_env_options['input'] = {
                'function': load_func,
                'options': {}
            }
        elif load_pattern == 'SINE_CURVE':
            env.scaling_env_options['input'] = INPUTS['SINE_CURVE']
        else:
            env.scaling_env_options['input'] = INPUTS['RANDOM']
        return env
    return _init


def train_ppo(load_pattern='RANDOM', model_name=None, total_timesteps=None):
    """
    Train PPO agent
    
    Args:
        load_pattern: Load pattern to use during training
        model_name: Name for the saved model (if None, auto-generates)
        total_timesteps: Total training timesteps (if None, uses config default)
    """
    if total_timesteps is None:
        total_timesteps = PPO_TOTAL_TIMESTEPS
    
    if model_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"ppo_{load_pattern.lower()}_{timestamp}"
    
    # Create vectorized environment
    env = make_vec_env(create_env_with_load_pattern(load_pattern), n_envs=1, seed=RANDOM_SEED)
    
    # Create model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=RANDOM_SEED,
        tensorboard_log=os.path.join(LOGS_DIR, 'ppo_tensorboard'),
        **PPO_CONFIG
    )
    
    # Set up callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=os.path.join(MODELS_DIR, 'checkpoints'),
        name_prefix=model_name
    )
    
    # Create eval environment
    eval_env = make_vec_env(create_env_with_load_pattern(load_pattern), n_envs=1, seed=RANDOM_SEED)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(MODELS_DIR, 'best'),
        log_path=os.path.join(LOGS_DIR, 'ppo_eval'),
        eval_freq=20000,
        deterministic=True,
        render=False
    )
    
    print(f"Training PPO agent with load pattern: {load_pattern}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Model will be saved as: {model_name}")
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        log_interval=10
    )
    
    # Save final model
    model_path = os.path.join(MODELS_DIR, f"{model_name}.zip")
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    env.close()
    eval_env.close()
    
    return model, model_path


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train PPO agent')
    parser.add_argument('--pattern', type=str, default='RANDOM',
                       choices=['RANDOM', 'SINUSOIDAL', 'STEADY', 'SPIKE', 'POISSON', 'SINE_CURVE'],
                       help='Load pattern for training')
    parser.add_argument('--timesteps', type=int, default=None,
                       help='Total training timesteps')
    parser.add_argument('--name', type=str, default=None,
                       help='Model name for saving')
    
    args = parser.parse_args()
    
    train_ppo(
        load_pattern=args.pattern,
        model_name=args.name,
        total_timesteps=args.timesteps
    )

