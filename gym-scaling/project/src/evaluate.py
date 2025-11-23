"""
Main evaluation script to compare RL agents, HPA, and static baselines
"""
import os
import sys
import gym
import gym_scaling
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import EVAL_EPISODES, EVAL_STEPS, RANDOM_SEED, RESULTS_DIR
from src.hpa_autoscaler import run_hpa_episode
from src.static_autoscaler import run_static_episode
from src.metrics import aggregate_metrics, save_metrics, compare_methods
from src.visualize import plot_all_comparisons
from src.env_wrapper import ScalingEnvWrapper
from stable_baselines3 import DQN, PPO
from gym_scaling.envs.scaling_env import INPUTS
from src.load_generators import LOAD_PATTERNS


def run_rl_episode(env, model, load_pattern=None, max_steps=1000):
    """
    Run a single episode with RL agent (DQN or PPO)
    
    Args:
        env: Gym environment
        model: Trained RL model (stable-baselines3)
        load_pattern: Load pattern name (optional)
        max_steps: Maximum number of steps
        
    Returns:
        dict with episode metrics
    """
    from gym_scaling.envs.scaling_env import INPUTS
    from src.load_generators import LOAD_PATTERNS
    
    # Set up load pattern if specified
    if load_pattern:
        if load_pattern in LOAD_PATTERNS:
            pattern = LOAD_PATTERNS[load_pattern]
            def load_func(step, max_influx, offset):
                return pattern['function'](step, max_influx, offset, **pattern['options'])
            env.scaling_env_options['input'] = {
                'function': load_func,
                'options': {}
            }
        elif load_pattern == 'SINE_CURVE':
            env.scaling_env_options['input'] = INPUTS['SINE_CURVE']
    
    obs = env.reset()
    
    metrics = {
        'total_cost': 0.0,
        'total_reward': 0.0,
        'queue_sizes': [],
        'loads': [],
        'instances': [],
        'actions': [],
        'num_scaling_events': 0,
        'steps': 0
    }
    
    step_idx = 0
    for step in range(max_steps):
        # Get action from RL model
        action, _ = model.predict(obs, deterministic=True)
        
        # Track scaling events (action 0 or 2 are scaling actions, 1 is hold)
        if action != 1:
            metrics['num_scaling_events'] += 1
        
        # Step environment
        obs, reward, done, info = env.step(action)
        
        # Collect metrics
        metrics['total_cost'] += env.total_cost
        metrics['total_reward'] += reward
        metrics['queue_sizes'].append(env.queue_size)
        metrics['loads'].append(env.load)
        metrics['instances'].append(len(env.instances))
        metrics['actions'].append(action)
        metrics['steps'] += 1
        
        if done:
            break
        
        step_idx += 1
    
    # Calculate averages
    if len(metrics['queue_sizes']) > 0:
        metrics['avg_queue_size'] = sum(metrics['queue_sizes']) / len(metrics['queue_sizes'])
        metrics['avg_load'] = sum(metrics['loads']) / len(metrics['loads'])
        metrics['avg_instances'] = sum(metrics['instances']) / len(metrics['instances'])
    else:
        metrics['avg_queue_size'] = 0.0
        metrics['avg_load'] = 0.0
        metrics['avg_instances'] = 0.0
    
    return metrics


def evaluate_method(env, method_name, method_config, load_pattern, num_episodes=5, max_steps=1000):
    """
    Evaluate a single method across multiple episodes
    
    Args:
        env: Gym environment
        method_name: Name of the method (e.g., 'DQN', 'HPA', 'Static')
        method_config: Configuration dict with method-specific parameters
        load_pattern: Load pattern to use
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        
    Returns:
        dict with aggregated metrics and individual episode metrics
    """
    print(f"Evaluating {method_name} with {load_pattern} load pattern...")
    
    episodes_metrics = []
    episode_sample = None  # Store first episode for trajectory plots
    
    for episode in range(num_episodes):
        print(f"  Episode {episode + 1}/{num_episodes}")
        
        if method_name == 'DQN':
            model_path = method_config.get('model_path')
            # Create a new environment for loading (SB3 needs env to determine shape)
            load_env = gym.make('Scaling-v0')
            load_env = ScalingEnvWrapper(load_env)
            model = DQN.load(model_path, env=load_env)
            load_env.close()
            ep_metrics = run_rl_episode(env, model, load_pattern, max_steps)
        elif method_name == 'PPO':
            model_path = method_config.get('model_path')
            # Create a new environment for loading (SB3 needs env to determine shape)
            load_env = gym.make('Scaling-v0')
            load_env = ScalingEnvWrapper(load_env)
            model = PPO.load(model_path, env=load_env)
            load_env.close()
            ep_metrics = run_rl_episode(env, model, load_pattern, max_steps)
        elif method_name == 'HPA':
            ep_metrics = run_hpa_episode(env, load_pattern, max_steps)
        elif method_name == 'Static':
            fixed_instances = method_config.get('fixed_instances')
            ep_metrics = run_static_episode(env, fixed_instances, load_pattern, max_steps)
        else:
            raise ValueError(f"Unknown method: {method_name}")
        
        episodes_metrics.append(ep_metrics)
        
        # Store first episode for trajectory visualization
        if episode_sample is None:
            episode_sample = ep_metrics.copy()
    
    # Aggregate metrics
    aggregated = aggregate_metrics(episodes_metrics)
    
    return {
        'aggregated': aggregated,
        'episodes': episodes_metrics,
        'sample_episode': episode_sample
    }


def run_comparative_evaluation(load_patterns=None, methods_config=None, output_dir=None):
    """
    Run comparative evaluation across all methods and load patterns
    
    Args:
        load_patterns: List of load pattern names to test (if None, uses defaults)
        methods_config: Dict mapping method names to their configs (if None, uses defaults)
        output_dir: Directory to save results (if None, uses RESULTS_DIR)
    """
    if load_patterns is None:
        load_patterns = ['STEADY', 'SINUSOIDAL', 'SPIKE', 'POISSON']
    
    if methods_config is None:
        # Default configuration - load from models directory
        from src.config import MODELS_DIR
        methods_config = {
            'DQN': {'model_path': os.path.join(MODELS_DIR, 'dqn_model.zip')},
            'PPO': {'model_path': os.path.join(MODELS_DIR, 'ppo_model.zip')},
            'HPA': {},
            'Static': {'fixed_instances': None}  # Will use default
        }
    
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(RESULTS_DIR, f'evaluation_{timestamp}')
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    sample_episodes = {}  # For trajectory plots
    
    for load_pattern in load_patterns:
        print(f"\n{'='*60}")
        print(f"Evaluating load pattern: {load_pattern}")
        print(f"{'='*60}\n")
        
        pattern_results = {}
        pattern_sample_episodes = {}
        
        env = gym.make('Scaling-v0')
        env = ScalingEnvWrapper(env)
        
        for method_name, method_config in methods_config.items():
            try:
                result = evaluate_method(
                    env,
                    method_name,
                    method_config,
                    load_pattern,
                    num_episodes=EVAL_EPISODES,
                    max_steps=EVAL_STEPS
                )
                pattern_results[method_name] = result['aggregated']
                pattern_sample_episodes[method_name] = result['sample_episode']
            except Exception as e:
                print(f"Error evaluating {method_name}: {e}")
                continue
        
        env.close()
        
        all_results[load_pattern] = pattern_results
        sample_episodes[load_pattern] = pattern_sample_episodes
        
        # Save results for this load pattern
        pattern_file = os.path.join(output_dir, f'results_{load_pattern.lower()}.json')
        save_metrics({'methods': pattern_results}, pattern_file)
        
        # Generate plots for this load pattern
        from src.visualize import plot_all_comparisons
        plot_dir = os.path.join(output_dir, f'plots_{load_pattern.lower()}')
        plot_all_comparisons(pattern_results, pattern_sample_episodes, plot_dir)
    
    # Save overall results
    overall_file = os.path.join(output_dir, 'all_results.json')
    save_metrics(all_results, overall_file)
    
    print(f"\n{'='*60}")
    print(f"Evaluation complete! Results saved to: {output_dir}")
    print(f"{'='*60}\n")
    
    return all_results, sample_episodes


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run comparative evaluation')
    parser.add_argument('--patterns', nargs='+', 
                       choices=['STEADY', 'SINUSOIDAL', 'SPIKE', 'POISSON', 'RANDOM', 'SINE_CURVE'],
                       default=['STEADY', 'SINUSOIDAL', 'SPIKE', 'POISSON'],
                       help='Load patterns to evaluate')
    parser.add_argument('--episodes', type=int, default=None,
                       help='Number of episodes per method (default: from config)')
    parser.add_argument('--steps', type=int, default=None,
                       help='Steps per episode (default: from config)')
    parser.add_argument('--dqn-model', type=str, default=None,
                       help='Path to DQN model (default: models/dqn_model.zip)')
    parser.add_argument('--ppo-model', type=str, default=None,
                       help='Path to PPO model (default: models/ppo_model.zip)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Update config if provided
    if args.episodes:
        from src.config import EVAL_EPISODES
        EVAL_EPISODES = args.episodes
    if args.steps:
        from src.config import EVAL_STEPS
        EVAL_STEPS = args.steps
    
    # Set up methods config
    from src.config import MODELS_DIR
    methods_config = {
        'HPA': {},
        'Static': {'fixed_instances': None}
    }
    
    # Add RL methods if models are specified
    if args.dqn_model or os.path.exists(os.path.join(MODELS_DIR, 'dqn_model.zip')):
        dqn_path = args.dqn_model or os.path.join(MODELS_DIR, 'dqn_model.zip')
        if os.path.exists(dqn_path):
            methods_config['DQN'] = {'model_path': dqn_path}
    
    if args.ppo_model or os.path.exists(os.path.join(MODELS_DIR, 'ppo_model.zip')):
        ppo_path = args.ppo_model or os.path.join(MODELS_DIR, 'ppo_model.zip')
        if os.path.exists(ppo_path):
            methods_config['PPO'] = {'model_path': ppo_path}
    
    run_comparative_evaluation(
        load_patterns=args.patterns,
        methods_config=methods_config,
        output_dir=args.output
    )

