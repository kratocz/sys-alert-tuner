import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Any
import pandas as pd
from datetime import datetime, timedelta


class ThresholdTuningEnvironment(gym.Env):
    def __init__(self, historical_data: pd.DataFrame, trigger_config: Dict[str, Any]):
        super(ThresholdTuningEnvironment, self).__init__()
        
        self.historical_data = historical_data
        self.trigger_config = trigger_config
        
        # Environment parameters
        self.max_steps = 100
        self.current_step = 0
        self.window_size = 24  # hours
        
        # Initialize thresholds
        self.current_threshold = trigger_config.get('initial_threshold', 80.0)
        self.min_threshold = trigger_config.get('min_threshold', 10.0)
        self.max_threshold = trigger_config.get('max_threshold', 200.0)
        
        # Action space: 0=decrease, 1=no change, 2=increase
        self.action_space = spaces.Discrete(3)
        
        # State space: [current_metrics, threshold, recent_trend, alert_history]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )
        
        # Metrics tracking
        self.false_positives = 0
        self.false_negatives = 0
        self.true_positives = 0
        self.true_negatives = 0
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.current_threshold = self.trigger_config.get('initial_threshold', 80.0)
        
        # Reset metrics
        self.false_positives = 0
        self.false_negatives = 0
        self.true_positives = 0
        self.true_negatives = 0
        
        # Select random starting point in historical data
        max_start = len(self.historical_data) - self.max_steps - self.window_size
        self.start_idx = np.random.randint(0, max_start) if max_start > 0 else 0
        
        return self._get_observation(), {}
    
    def step(self, action):
        self.current_step += 1
        
        # Apply action to threshold
        threshold_change = 0
        if action == 0:  # Decrease threshold
            threshold_change = -5.0
        elif action == 2:  # Increase threshold
            threshold_change = 5.0
        # action == 1 means no change
        
        # Update threshold within bounds
        new_threshold = np.clip(
            self.current_threshold + threshold_change,
            self.min_threshold,
            self.max_threshold
        )
        self.current_threshold = new_threshold
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        
        # Get next observation
        observation = self._get_observation()
        
        info = {
            'threshold': self.current_threshold,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives,
            'true_positives': self.true_positives,
            'true_negatives': self.true_negatives
        }
        
        return observation, reward, done, False, info
    
    def _get_observation(self):
        """Get current state observation"""
        current_idx = self.start_idx + self.current_step
        
        if current_idx >= len(self.historical_data):
            current_idx = len(self.historical_data) - 1
        
        # Get current window of data
        window_start = max(0, current_idx - self.window_size)
        window_data = self.historical_data.iloc[window_start:current_idx + 1]
        
        if len(window_data) == 0:
            return np.zeros(10, dtype=np.float32)
        
        # Extract features
        current_value = window_data['value'].iloc[-1] if len(window_data) > 0 else 0
        mean_value = window_data['value'].mean()
        std_value = window_data['value'].std()
        min_value = window_data['value'].min()
        max_value = window_data['value'].max()
        
        # Trend calculation
        if len(window_data) > 1:
            trend = np.polyfit(range(len(window_data)), window_data['value'], 1)[0]
        else:
            trend = 0
        
        # Recent alert history (simplified)
        recent_alerts = len(window_data[window_data['value'] > self.current_threshold])
        
        observation = np.array([
            current_value / 100.0,  # Normalize
            mean_value / 100.0,
            std_value / 100.0 if not np.isnan(std_value) else 0,
            min_value / 100.0,
            max_value / 100.0,
            self.current_threshold / 100.0,
            trend / 100.0,
            recent_alerts / len(window_data) if len(window_data) > 0 else 0,
            self.false_positives / max(1, self.current_step),
            self.false_negatives / max(1, self.current_step)
        ], dtype=np.float32)
        
        return observation
    
    def _calculate_reward(self):
        """Calculate reward based on threshold performance"""
        current_idx = self.start_idx + self.current_step
        
        if current_idx >= len(self.historical_data):
            return 0
        
        # Get current and recent data
        window_start = max(0, current_idx - 10)  # Look at last 10 points
        recent_data = self.historical_data.iloc[window_start:current_idx + 1]
        
        if len(recent_data) == 0:
            return 0
        
        # Simulate ground truth (simplified)
        # In real scenario, this would come from incident reports or manual labeling
        current_value = recent_data['value'].iloc[-1]
        
        # Simple heuristic: values > 90 are "real issues", < 30 are "normal"
        is_real_issue = current_value > 90
        is_normal = current_value < 30
        
        # Check if threshold would trigger
        would_trigger = current_value > self.current_threshold
        
        reward = 0
        
        if is_real_issue and would_trigger:
            # True positive
            self.true_positives += 1
            reward += 10
        elif is_real_issue and not would_trigger:
            # False negative (missed alert)
            self.false_negatives += 1
            reward -= 15
        elif is_normal and would_trigger:
            # False positive (unnecessary alert)
            self.false_positives += 1
            reward -= 10
        elif is_normal and not would_trigger:
            # True negative
            self.true_negatives += 1
            reward += 1
        
        # Penalty for extreme thresholds
        if self.current_threshold < self.min_threshold + 5:
            reward -= 2
        elif self.current_threshold > self.max_threshold - 5:
            reward -= 2
        
        return reward
    
    def get_metrics(self):
        """Get current performance metrics"""
        total = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        
        if total == 0:
            return {}
        
        accuracy = (self.true_positives + self.true_negatives) / total
        
        precision = self.true_positives / max(1, self.true_positives + self.false_positives)
        recall = self.true_positives / max(1, self.true_positives + self.false_negatives)
        
        f1_score = 2 * (precision * recall) / max(0.01, precision + recall)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'false_positive_rate': self.false_positives / total,
            'false_negative_rate': self.false_negatives / total,
            'threshold': self.current_threshold
        }