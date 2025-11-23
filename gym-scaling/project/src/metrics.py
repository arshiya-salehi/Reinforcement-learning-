"""
Metrics collection and analysis utilities
"""
import numpy as np
import json
import os
from collections import defaultdict


def calculate_statistics(values):
    """
    Calculate mean and standard deviation
    
    Args:
        values: List of numerical values
        
    Returns:
        dict with 'mean', 'std', 'min', 'max'
    """
    if len(values) == 0:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
    
    values = np.array(values)
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values))
    }


def aggregate_metrics(episodes_metrics):
    """
    Aggregate metrics across multiple episodes
    
    Args:
        episodes_metrics: List of metric dictionaries from episodes
        
    Returns:
        Aggregated metrics dictionary
    """
    if len(episodes_metrics) == 0:
        return {}
    
    aggregated = {}
    
    # Aggregate scalar metrics
    scalar_keys = ['total_cost', 'total_reward', 'avg_queue_size', 'avg_load', 
                   'avg_instances', 'num_scaling_events', 'steps']
    
    for key in scalar_keys:
        values = [ep[key] for ep in episodes_metrics if key in ep]
        if values:
            aggregated[key] = calculate_statistics(values)
    
    # Calculate oscillation frequency (scale up/down cycles)
    oscillation_counts = []
    for ep in episodes_metrics:
        if 'actions' in ep:
            actions = ep['actions']
            oscillations = 0
            prev_action = None
            for action in actions:
                # Track transitions: up->down or down->up
                if prev_action is not None:
                    if (prev_action == 2 and action == 0) or (prev_action == 0 and action == 2):
                        oscillations += 1
                prev_action = action
            oscillation_counts.append(oscillations)
    
    if oscillation_counts:
        aggregated['oscillations'] = calculate_statistics(oscillation_counts)
    
    return aggregated


def compare_methods(methods_results):
    """
    Compare results across different methods
    
    Args:
        methods_results: Dict mapping method name to aggregated metrics
        
    Returns:
        Comparison table as dict
    """
    comparison = {}
    
    # Extract common metrics
    metrics_to_compare = ['total_cost', 'total_reward', 'avg_queue_size', 
                         'avg_load', 'num_scaling_events', 'oscillations']
    
    for metric in metrics_to_compare:
        comparison[metric] = {}
        for method_name, method_metrics in methods_results.items():
            if metric in method_metrics:
                comparison[metric][method_name] = method_metrics[metric]
    
    return comparison


def save_metrics(metrics, filepath):
    """
    Save metrics to JSON file
    
    Args:
        metrics: Metrics dictionary (can contain numpy types)
        filepath: Path to save JSON file
    """
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_native(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj
    
    metrics_native = convert_to_native(metrics)
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(metrics_native, f, indent=2)


def load_metrics(filepath):
    """Load metrics from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def format_statistics(stats, precision=2):
    """
    Format statistics for display
    
    Args:
        stats: Dictionary with 'mean' and 'std' keys
        precision: Number of decimal places
        
    Returns:
        Formatted string: "mean ± std"
    """
    if isinstance(stats, dict) and 'mean' in stats and 'std' in stats:
        return f"{stats['mean']:.{precision}f} ± {stats['std']:.{precision}f}"
    return str(stats)


def create_comparison_table(methods_results, metrics_to_show=None):
    """
    Create a formatted comparison table
    
    Args:
        methods_results: Dict mapping method name to aggregated metrics
        metrics_to_show: List of metric keys to include (if None, shows all)
        
    Returns:
        List of rows (each row is a list of values)
    """
    if metrics_to_show is None:
        # Default metrics to show
        metrics_to_show = ['total_cost', 'total_reward', 'avg_queue_size', 
                          'avg_load', 'num_scaling_events']
    
    table = []
    header = ['Method'] + [m.replace('_', ' ').title() for m in metrics_to_show]
    table.append(header)
    
    for method_name, method_metrics in methods_results.items():
        row = [method_name]
        for metric in metrics_to_show:
            if metric in method_metrics:
                stats = method_metrics[metric]
                if isinstance(stats, dict):
                    row.append(format_statistics(stats))
                else:
                    row.append(str(stats))
            else:
                row.append('N/A')
        table.append(row)
    
    return table

