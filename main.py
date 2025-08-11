#!/usr/bin/env python3

import os
import sys
import argparse
from datetime import datetime

# Add sys_alert_tuner directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'sys_alert_tuner'))

from sys_alert_tuner.trainer import ThresholdTuner, main as train_main


def evaluate_model():
    """Evaluate a trained model"""
    config = {
        'agent': {
            'learning_rate': 0.001,
            'gamma': 0.95,
            'epsilon': 0.0,  # No exploration for evaluation
            'epsilon_min': 0.0,
            'epsilon_decay': 1.0,
            'batch_size': 32,
            'memory_size': 10000,
            'target_update': 100
        }
    }
    
    tuner = ThresholdTuner(config)
    
    # Load sample data
    print("Loading sample data for evaluation...")
    data = tuner.load_sample_data()
    prepared_data = tuner.prepare_data(data)
    
    # Setup environment
    item_data = tuner.setup_training(prepared_data)
    
    # Find the most recent model
    models_dir = 'models'
    if not os.path.exists(models_dir):
        print("No models directory found. Please train a model first.")
        return
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    if not model_files:
        print("No trained models found. Please train a model first.")
        return
    
    # Use most recent model
    model_files.sort()
    model_path = os.path.join(models_dir, model_files[-1])
    
    print(f"Loading model: {model_path}")
    tuner.load_model(model_path)
    
    # Evaluate
    results, detailed_results = tuner.evaluate(episodes=20)
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    print(f"Recommended Threshold: {results['recommended_threshold']:.2f}")
    print(f"Average Reward: {results['avg_total_reward']:.2f}")
    print(f"Average Accuracy: {results['avg_accuracy']:.3f}")
    print(f"Average Precision: {results['avg_precision']:.3f}")
    print(f"Average Recall: {results['avg_recall']:.3f}")
    print(f"Average F1 Score: {results['avg_f1_score']:.3f}")
    print(f"False Positive Rate: {results['avg_false_positive_rate']:.3f}")
    print(f"False Negative Rate: {results['avg_false_negative_rate']:.3f}")


def compare_thresholds():
    """Compare different threshold values"""
    print("Comparing different threshold strategies...")
    
    # Load sample data
    from sys_alert_tuner.data_processor import DataProcessor
    processor = DataProcessor()
    
    # Generate sample data
    tuner = ThresholdTuner({})
    data = tuner.load_sample_data()
    prepared_data = tuner.prepare_data(data)
    
    # Get first item for comparison
    item_id = prepared_data['itemid'].iloc[0]
    item_data = prepared_data[prepared_data['itemid'] == item_id]
    
    # Test different threshold strategies
    thresholds_to_test = [
        ('Conservative (95th percentile)', item_data['value'].quantile(0.95)),
        ('Moderate (90th percentile)', item_data['value'].quantile(0.90)),
        ('Aggressive (80th percentile)', item_data['value'].quantile(0.80)),
        ('Very Aggressive (70th percentile)', item_data['value'].quantile(0.70))
    ]
    
    print(f"\nThreshold Comparison for item: {item_id}")
    print("="*60)
    print(f"{'Strategy':<25} {'Threshold':<10} {'Alert Rate':<12} {'Coverage':<10}")
    print("-"*60)
    
    for strategy_name, threshold in thresholds_to_test:
        alerts = (item_data['value'] > threshold).sum()
        alert_rate = alerts / len(item_data) * 100
        
        # Simulate coverage (simplified)
        high_values = (item_data['value'] > item_data['value'].quantile(0.95)).sum()
        coverage = min(100, (alerts / max(1, high_values)) * 100) if high_values > 0 else 0
        
        print(f"{strategy_name:<25} {threshold:<10.1f} {alert_rate:<12.2f}% {coverage:<10.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Zabbix Threshold Tuner')
    parser.add_argument('command', choices=['train', 'evaluate', 'compare'], 
                       help='Command to execute')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes (default: 1000)')
    parser.add_argument('--model-path', type=str,
                       help='Path to model file for evaluation')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        print("Starting training...")
        train_main()
    elif args.command == 'evaluate':
        evaluate_model()
    elif args.command == 'compare':
        compare_thresholds()


if __name__ == "__main__":
    main()