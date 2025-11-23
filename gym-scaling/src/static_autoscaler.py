"""
Static baseline autoscaler (fixed instance count)
"""
import gymnasium as gym
import gym_scaling
from src.config import ENV_CONFIG


class StaticAutoscaling:
    """
    Static autoscaler that maintains a fixed number of instances
    
    This serves as a baseline for comparison
    """
    
    def __init__(self, fixed_instances=None):
        """
        Initialize static autoscaler
        
        Args:
            fixed_instances: Fixed number of instances to maintain
                           If None, uses half of max_instances as default
        """
        self.fixed_instances = fixed_instances
        self.target_instances = None
        
    def get_action(self, observation, current_instances, step_idx):
        """
        Get action to reach target instance count
        
        Args:
            observation: Current environment observation
            current_instances: Current number of instances
            step_idx: Current step index
            
        Returns:
            Action: 0 (scale down), 1 (hold), or 2 (scale up)
        """
        if self.target_instances is None:
            # Set target on first call
            if self.fixed_instances is not None:
                self.target_instances = self.fixed_instances
            else:
                # Default to mid-point between min and max
                max_instances = ENV_CONFIG['max_instances']
                min_instances = ENV_CONFIG['min_instances']
                self.target_instances = int((max_instances + min_instances) / 2)
        
        # Calculate difference
        diff = current_instances - self.target_instances
        
        if diff > 0:
            return 0  # Scale down
        elif diff < 0:
            return 2  # Scale up
        else:
            return 1  # Hold
    
    def reset(self):
        """Reset the autoscaler state"""
        self.target_instances = None


def run_static_episode(env, fixed_instances=None, load_pattern=None, max_steps=1000):
    """
    Run a single episode with static autoscaler
    
    Args:
        env: Gym environment
        fixed_instances: Fixed number of instances (if None, uses default)
        load_pattern: Load pattern name (optional, for logging)
        max_steps: Maximum number of steps to run
        
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
    
    autoscaler = StaticAutoscaling(fixed_instances=fixed_instances)
    obs = env.reset()
    autoscaler.reset()
    
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
        # Get current number of instances
        current_instances = len(env.instances)
        
        # Get action from static autoscaler
        action = autoscaler.get_action(obs, current_instances, step_idx)
        
        # Track scaling events
        if action != 1:  # Non-hold action
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


if __name__ == '__main__':
    # Test the static autoscaler
    env = gym.make('Scaling-v0')
    metrics = run_static_episode(env, fixed_instances=50, load_pattern='SINUSOIDAL', max_steps=1000)
    print("Static Episode Metrics:")
    print(f"Total Cost: {metrics['total_cost']:.2f}")
    print(f"Total Reward: {metrics['total_reward']:.2f}")
    print(f"Average Queue Size: {metrics['avg_queue_size']:.2f}")
    print(f"Average Load: {metrics['avg_load']:.2f}")
    print(f"Average Instances: {metrics['avg_instances']:.2f}")
    print(f"Number of Scaling Events: {metrics['num_scaling_events']}")
    env.close()

