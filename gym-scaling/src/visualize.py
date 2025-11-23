"""
Visualization utilities for comparison plots
"""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os
from src.config import RESULTS_DIR


def plot_scaling_curves(episode_metrics_dict, save_path=None, title="Scaling Curves Comparison"):
    """
    Plot scaling curves (instances over time) for different methods
    
    Args:
        episode_metrics_dict: Dict mapping method name to episode metrics
                            Each episode metrics should have 'instances' list
        save_path: Path to save figure (if None, shows plot)
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for method_name, metrics in episode_metrics_dict.items():
        if 'instances' in metrics and len(metrics['instances']) > 0:
            instances = metrics['instances']
            steps = range(len(instances))
            ax.plot(steps, instances, label=method_name, alpha=0.7)
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Number of Instances')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_queue_sizes(episode_metrics_dict, save_path=None, title="Queue Size Comparison"):
    """
    Plot queue sizes over time for different methods
    
    Args:
        episode_metrics_dict: Dict mapping method name to episode metrics
        save_path: Path to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for method_name, metrics in episode_metrics_dict.items():
        if 'queue_sizes' in metrics and len(metrics['queue_sizes']) > 0:
            queue_sizes = metrics['queue_sizes']
            steps = range(len(queue_sizes))
            ax.plot(steps, queue_sizes, label=method_name, alpha=0.7)
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Queue Size')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_load_comparison(episode_metrics_dict, save_path=None, title="Load Comparison"):
    """
    Plot load percentages over time for different methods
    
    Args:
        episode_metrics_dict: Dict mapping method name to episode metrics
        save_path: Path to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for method_name, metrics in episode_metrics_dict.items():
        if 'loads' in metrics and len(metrics['loads']) > 0:
            loads = metrics['loads']
            steps = range(len(loads))
            ax.plot(steps, loads, label=method_name, alpha=0.7)
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Load (%)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_cost_comparison(aggregated_metrics_dict, save_path=None, title="Cost Comparison"):
    """
    Plot cost comparison across methods using bar chart
    
    Args:
        aggregated_metrics_dict: Dict mapping method name to aggregated metrics
        save_path: Path to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = []
    costs = []
    errors = []
    
    for method_name, metrics in aggregated_metrics_dict.items():
        if 'total_cost' in metrics:
            cost_stats = metrics['total_cost']
            methods.append(method_name)
            costs.append(cost_stats.get('mean', 0))
            errors.append(cost_stats.get('std', 0))
    
    x_pos = np.arange(len(methods))
    ax.bar(x_pos, costs, yerr=errors, alpha=0.7, capsize=5)
    ax.set_xlabel('Method')
    ax.set_ylabel('Total Cost')
    ax.set_title(title)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_metric_comparison(aggregated_metrics_dict, metric_name, save_path=None, title=None):
    """
    Plot comparison of a specific metric across methods
    
    Args:
        aggregated_metrics_dict: Dict mapping method name to aggregated metrics
        metric_name: Name of metric to compare
        save_path: Path to save figure
        title: Plot title (if None, auto-generates from metric_name)
    """
    if title is None:
        title = f"{metric_name.replace('_', ' ').title()} Comparison"
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = []
    means = []
    stds = []
    
    for method_name, metrics in aggregated_metrics_dict.items():
        if metric_name in metrics:
            stats = metrics[metric_name]
            if isinstance(stats, dict):
                methods.append(method_name)
                means.append(stats.get('mean', 0))
                stds.append(stats.get('std', 0))
    
    x_pos = np.arange(len(methods))
    ax.bar(x_pos, means, yerr=stds, alpha=0.7, capsize=5)
    ax.set_xlabel('Method')
    ax.set_ylabel(metric_name.replace('_', ' ').title())
    ax.set_title(title)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_all_comparisons(aggregated_metrics_dict, episode_metrics_dict, output_dir=None):
    """
    Generate all comparison plots
    
    Args:
        aggregated_metrics_dict: Dict mapping method name to aggregated metrics
        episode_metrics_dict: Dict mapping method name to single episode metrics (for trajectories)
        output_dir: Directory to save plots (if None, uses RESULTS_DIR)
    """
    if output_dir is None:
        output_dir = os.path.join(RESULTS_DIR, 'plots')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot trajectories
    if episode_metrics_dict:
        plot_scaling_curves(episode_metrics_dict, 
                          save_path=os.path.join(output_dir, 'scaling_curves.png'))
        plot_queue_sizes(episode_metrics_dict,
                        save_path=os.path.join(output_dir, 'queue_sizes.png'))
        plot_load_comparison(episode_metrics_dict,
                           save_path=os.path.join(output_dir, 'load_comparison.png'))
    
    # Plot aggregated metrics
    if aggregated_metrics_dict:
        plot_cost_comparison(aggregated_metrics_dict,
                           save_path=os.path.join(output_dir, 'cost_comparison.png'))
        
        metrics_to_plot = ['total_reward', 'avg_queue_size', 'num_scaling_events']
        for metric in metrics_to_plot:
            plot_metric_comparison(aggregated_metrics_dict, metric,
                                 save_path=os.path.join(output_dir, f'{metric}_comparison.png'))
    
    print(f"All plots saved to {output_dir}")

