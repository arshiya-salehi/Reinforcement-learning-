"""
Kubernetes HPA-style rule-based autoscaler
"""
import gymnasium as gym
import gym_scaling
from src.config import HPA_CONFIG


class HPAAutoscaling:
    """
    Horizontal Pod Autoscaler-style rule-based autoscaler
    
    Rules:
    - Scale up if load > threshold_high
    - Scale down if load < threshold_low
    - Cooldown period to prevent oscillation
    """
    
    def __init__(self, threshold_high=None, threshold_low=None, cooldown_period=None):
        """
        Initialize HPA autoscaler
        
        Args:
            threshold_high: Load percentage above which to scale up (default: 70%)
            threshold_low: Load percentage below which to scale down (default: 40%)
            cooldown_period: Steps to wait before allowing another scaling action (default: 10)
        """
        self.threshold_high = threshold_high if threshold_high is not None else HPA_CONFIG['threshold_high']
        self.threshold_low = threshold_low if threshold_low is not None else HPA_CONFIG['threshold_low']
        self.cooldown_period = cooldown_period if cooldown_period is not None else HPA_CONFIG['cooldown_period']
        self.last_action_step = -self.cooldown_period - 1
        self.last_action = 0  # -1, 0, or 1
        
    def get_action(self, observation, load, step_idx):
        """
        Get scaling action based on current load
        
        Args:
            observation: Current environment observation
            load: Current load percentage (0-100)
            step_idx: Current step index
            
        Returns:
            Action: 0 (hold), 1 (scale up), or 2 (scale down)
            Action mapping: 0 -> -1 (scale down), 1 -> 0 (hold), 2 -> +1 (scale up)
            But the environment uses: action 0 -> -1, action 1 -> 0, action 2 -> +1
            Actually, environment has discrete actions (-1, 0, 1) mapped to indices (0, 1, 2)
        """
        # Check cooldown period
        steps_since_last_action = step_idx - self.last_action_step
        if steps_since_last_action < self.cooldown_period:
            return 1  # Hold (action index 1 = 0 change)
        
        # Determine action based on load
        if load > self.threshold_high:
            action = 2  # Scale up (action index 2 = +1)
        elif load < self.threshold_low:
            action = 0  # Scale down (action index 0 = -1)
        else:
            action = 1  # Hold (action index 1 = 0)
        
        # Update last action step if we're taking a non-hold action
        if action != 1:
            self.last_action_step = step_idx
            self.last_action = action
        
        return action
    
    def reset(self):
        """Reset the autoscaler state"""
        self.last_action_step = -self.cooldown_period - 1
        self.last_action = 0


def run_hpa_episode(env, load_pattern=None, max_steps=1000):
    """
    Run a single episode with HPA autoscaler
    
    Args:
        env: Gym environment
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
    
    autoscaler = HPAAutoscaling()
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
        # Get current load from environment
        load = env.load
        
        # Get action from HPA
        action = autoscaler.get_action(obs, load, step_idx)
        
        # Track scaling events
        if action != 1:  # Non-hold action
            metrics['num_scaling_events'] += 1
        
        # Step environment
        obs, reward, done, info = env.step(action)
        
        # Collect metrics
        metrics['total_cost'] += env.total_cost
        metrics['total_reward'] += reward
        metrics['queue_sizes'].append(env.queue_size)
        metrics['loads'].append(load)
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
    # Test the HPA autoscaler
    env = gym.make('Scaling-v0')
    metrics = run_hpa_episode(env, load_pattern='SINUSOIDAL', max_steps=1000)
    print("HPA Episode Metrics:")
    print(f"Total Cost: {metrics['total_cost']:.2f}")
    print(f"Total Reward: {metrics['total_reward']:.2f}")
    print(f"Average Queue Size: {metrics['avg_queue_size']:.2f}")
    print(f"Average Load: {metrics['avg_load']:.2f}")
    print(f"Average Instances: {metrics['avg_instances']:.2f}")
    print(f"Number of Scaling Events: {metrics['num_scaling_events']}")
    env.close()

