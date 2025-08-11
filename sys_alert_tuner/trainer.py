import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import torch
from tqdm import tqdm

from .zabbix_client import ZabbixClient
from .data_processor import DataProcessor
from .threshold_environment import ThresholdTuningEnvironment
from .dqn_agent import DQNAgent


class ThresholdTuner:
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize components
        self.zabbix_client = None
        self.data_processor = DataProcessor()
        self.environment = None
        self.agent = None
        
        # Training tracking
        self.training_history = {
            'episodes': [],
            'rewards': [],
            'losses': [],
            'epsilon': [],
            'metrics': []
        }
    
    def load_data_from_zabbix(self, days_back: int = 30) -> pd.DataFrame:
        """Load historical data from Zabbix"""
        print("Connecting to Zabbix...")
        self.zabbix_client = ZabbixClient()
        
        # Get hosts
        hosts = self.zabbix_client.get_hosts()
        if not hosts:
            raise ValueError("No hosts found in Zabbix")
        
        print(f"Found {len(hosts)} hosts")
        
        # Select first few hosts for demo
        selected_hosts = hosts[:5]  # Limit to 5 hosts for demo
        hostids = [host['hostid'] for host in selected_hosts]
        
        # Get items (focusing on CPU, memory, disk metrics)
        items = self.zabbix_client.get_items(
            hostids=hostids,
            key_pattern='system.cpu'  # Focus on CPU metrics
        )
        
        if not items:
            print("No CPU items found, trying memory metrics...")
            items = self.zabbix_client.get_items(
                hostids=hostids,
                key_pattern='memory'
            )
        
        if not items:
            print("No specific items found, getting all items...")
            items = self.zabbix_client.get_items(hostids=hostids)
        
        if not items:
            raise ValueError("No items found for selected hosts")
        
        print(f"Found {len(items)} items")
        
        # Get historical data
        itemids = [item['itemid'] for item in items[:10]]  # Limit to 10 items
        time_from = datetime.now() - timedelta(days=days_back)
        
        print(f"Loading historical data from {time_from}...")
        historical_data = self.zabbix_client.get_historical_data(
            itemids=itemids,
            time_from=time_from
        )
        
        self.zabbix_client.disconnect()
        
        if historical_data.empty:
            raise ValueError("No historical data found")
        
        print(f"Loaded {len(historical_data)} data points")
        return historical_data
    
    def load_sample_data(self) -> pd.DataFrame:
        """Generate sample data for demo purposes"""
        print("Generating sample data...")
        
        # Generate synthetic time series data
        start_time = datetime.now() - timedelta(days=30)
        timestamps = pd.date_range(start_time, periods=10000, freq='5T')
        
        data = []
        
        # Create 3 synthetic items
        for item_id in ['item_001', 'item_002', 'item_003']:
            base_value = np.random.uniform(20, 80)
            
            for i, ts in enumerate(timestamps):
                # Add daily and weekly patterns
                hour_factor = 1 + 0.3 * np.sin(2 * np.pi * ts.hour / 24)
                day_factor = 1 + 0.2 * np.sin(2 * np.pi * ts.dayofweek / 7)
                
                # Add trend
                trend = 0.001 * i
                
                # Add noise
                noise = np.random.normal(0, 5)
                
                # Occasional spikes (simulate incidents)
                if np.random.random() < 0.02:  # 2% chance of spike
                    spike = np.random.uniform(50, 100)
                else:
                    spike = 0
                
                value = base_value * hour_factor * day_factor + trend + noise + spike
                value = max(0, value)  # Ensure non-negative
                
                data.append({
                    'itemid': item_id,
                    'timestamp': ts,
                    'value': value
                })
        
        df = pd.DataFrame(data)
        print(f"Generated {len(df)} synthetic data points")
        return df
    
    def prepare_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for training"""
        print("Preparing data...")
        
        # Clean and add features
        clean_data = self.data_processor.load_and_clean_data(raw_data)
        featured_data = self.data_processor.add_features(clean_data)
        
        # Generate synthetic incident labels
        final_data = self.data_processor.generate_synthetic_incidents(featured_data)
        
        # Create summary report
        report = self.data_processor.create_summary_report(final_data)
        print("Data Summary:")
        for key, value in report.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for subkey, subvalue in value.items():
                    print(f"    {subkey}: {subvalue}")
            else:
                print(f"  {key}: {value}")
        
        return final_data
    
    def setup_training(self, data: pd.DataFrame, item_id: str = None):
        """Setup environment and agent for training"""
        if item_id is None:
            item_id = data['itemid'].iloc[0]
        
        # Filter data for specific item
        item_data = data[data['itemid'] == item_id].copy().reset_index(drop=True)
        
        print(f"Setting up training for item: {item_id}")
        print(f"Data points: {len(item_data)}")
        
        # Environment configuration
        env_config = {
            'initial_threshold': item_data['value'].quantile(0.8),
            'min_threshold': item_data['value'].quantile(0.1),
            'max_threshold': item_data['value'].quantile(0.99)
        }
        
        print(f"Threshold range: {env_config['min_threshold']:.2f} - {env_config['max_threshold']:.2f}")
        print(f"Initial threshold: {env_config['initial_threshold']:.2f}")
        
        # Create environment
        self.environment = ThresholdTuningEnvironment(item_data, env_config)
        
        # Create agent
        state_size = self.environment.observation_space.shape[0]
        action_size = self.environment.action_space.n
        
        agent_config = self.config.get('agent', {})
        self.agent = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            lr=agent_config.get('learning_rate', 0.001),
            gamma=agent_config.get('gamma', 0.95),
            epsilon=agent_config.get('epsilon', 1.0),
            epsilon_min=agent_config.get('epsilon_min', 0.01),
            epsilon_decay=agent_config.get('epsilon_decay', 0.995),
            batch_size=agent_config.get('batch_size', 32),
            memory_size=agent_config.get('memory_size', 10000),
            target_update=agent_config.get('target_update', 100)
        )
        
        return item_data
    
    def train(self, episodes: int = 1000):
        """Train the agent"""
        if self.environment is None or self.agent is None:
            raise ValueError("Environment and agent must be set up before training")
        
        print(f"Starting training for {episodes} episodes...")
        
        for episode in tqdm(range(episodes), desc="Training"):
            state, _ = self.environment.reset()
            total_reward = 0
            steps = 0
            
            done = False
            while not done:
                action = self.agent.act(state, training=True)
                next_state, reward, done, truncated, info = self.environment.step(action)
                
                self.agent.remember(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                steps += 1
                
                # Train agent
                self.agent.replay()
            
            # Record training metrics
            self.training_history['episodes'].append(episode)
            self.training_history['rewards'].append(total_reward)
            self.training_history['epsilon'].append(self.agent.epsilon)
            
            # Get environment metrics
            env_metrics = self.environment.get_metrics()
            self.training_history['metrics'].append(env_metrics)
            
            # Get agent stats
            agent_stats = self.agent.get_training_stats()
            if 'avg_loss' in agent_stats:
                self.training_history['losses'].append(agent_stats['avg_loss'])
            
            # Print progress
            if episode % 100 == 0:
                avg_reward = np.mean(self.training_history['rewards'][-100:])
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, "
                      f"Epsilon: {self.agent.epsilon:.3f}, "
                      f"Threshold: {info.get('threshold', 0):.2f}")
                
                if env_metrics:
                    print(f"  Metrics - F1: {env_metrics.get('f1_score', 0):.3f}, "
                          f"FP Rate: {env_metrics.get('false_positive_rate', 0):.3f}")
        
        print("Training completed!")
    
    def evaluate(self, episodes: int = 10):
        """Evaluate the trained agent"""
        if self.environment is None or self.agent is None:
            raise ValueError("Environment and agent must be set up before evaluation")
        
        print(f"Evaluating agent over {episodes} episodes...")
        
        eval_results = []
        
        for episode in range(episodes):
            state, _ = self.environment.reset()
            total_reward = 0
            
            done = False
            while not done:
                action = self.agent.act(state, training=False)  # No exploration
                state, reward, done, truncated, info = self.environment.step(action)
                total_reward += reward
            
            metrics = self.environment.get_metrics()
            metrics['total_reward'] = total_reward
            eval_results.append(metrics)
        
        # Calculate average metrics
        avg_metrics = {}
        for key in eval_results[0].keys():
            if key != 'threshold':  # Don't average threshold
                avg_metrics[f'avg_{key}'] = np.mean([r[key] for r in eval_results])
        
        # Get final threshold recommendation
        final_threshold = eval_results[-1]['threshold']
        avg_metrics['recommended_threshold'] = final_threshold
        
        print("Evaluation Results:")
        for key, value in avg_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        return avg_metrics, eval_results
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if self.agent is None:
            raise ValueError("No agent to save")
        
        self.agent.save_model(filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        if self.agent is None:
            raise ValueError("Agent must be initialized before loading")
        
        self.agent.load_model(filepath)
    
    def plot_training_progress(self, save_path: str = None):
        """Plot training progress"""
        if not self.training_history['episodes']:
            print("No training history to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress', fontsize=16)
        
        episodes = self.training_history['episodes']
        
        # Rewards
        axes[0, 0].plot(episodes, self.training_history['rewards'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        
        # Epsilon
        axes[0, 1].plot(episodes, self.training_history['epsilon'])
        axes[0, 1].set_title('Exploration Rate (Epsilon)')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Epsilon')
        
        # F1 Score
        f1_scores = [m.get('f1_score', 0) for m in self.training_history['metrics']]
        axes[1, 0].plot(episodes, f1_scores)
        axes[1, 0].set_title('F1 Score')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('F1 Score')
        
        # False Positive Rate
        fp_rates = [m.get('false_positive_rate', 0) for m in self.training_history['metrics']]
        axes[1, 1].plot(episodes, fp_rates)
        axes[1, 1].set_title('False Positive Rate')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('FP Rate')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training progress plot saved to {save_path}")
        
        plt.show()


def main():
    """Main training function"""
    # Configuration
    config = {
        'agent': {
            'learning_rate': float(os.getenv('LEARNING_RATE', 0.001)),
            'gamma': 0.95,
            'epsilon': 1.0,
            'epsilon_min': 0.01,
            'epsilon_decay': 0.995,
            'batch_size': int(os.getenv('BATCH_SIZE', 32)),
            'memory_size': 10000,
            'target_update': 100
        }
    }
    
    # Initialize tuner
    tuner = ThresholdTuner(config)
    
    try:
        # Try to load real Zabbix data
        print("Attempting to load data from Zabbix...")
        data = tuner.load_data_from_zabbix(days_back=7)
    except Exception as e:
        print(f"Failed to load Zabbix data: {e}")
        print("Using sample data instead...")
        data = tuner.load_sample_data()
    
    # Prepare data
    prepared_data = tuner.prepare_data(data)
    
    # Setup training
    item_data = tuner.setup_training(prepared_data)
    
    # Train agent
    episodes = int(os.getenv('TRAINING_EPISODES', 1000))
    tuner.train(episodes=episodes)
    
    # Evaluate
    eval_results, _ = tuner.evaluate(episodes=10)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = f"models/threshold_tuner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    tuner.save_model(model_path)
    
    # Plot results
    os.makedirs('plots', exist_ok=True)
    plot_path = f"plots/training_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    tuner.plot_training_progress(save_path=plot_path)
    
    print(f"\nTraining completed!")
    print(f"Model saved to: {model_path}")
    print(f"Plots saved to: {plot_path}")
    print(f"\nFinal recommended threshold: {eval_results['recommended_threshold']:.2f}")


if __name__ == "__main__":
    main()