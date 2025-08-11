import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt


class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def load_and_clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare historical data"""
        if data.empty:
            raise ValueError("Input data is empty")
        
        # Ensure required columns exist
        required_columns = ['timestamp', 'value', 'itemid']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Sort by timestamp
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        # Remove duplicates
        data = data.drop_duplicates(subset=['timestamp', 'itemid']).reset_index(drop=True)
        
        # Handle missing values
        data['value'] = pd.to_numeric(data['value'], errors='coerce')
        data = data.dropna(subset=['value']).reset_index(drop=True)
        
        # Remove obvious outliers (values beyond 99.9th percentile)
        q99_9 = data['value'].quantile(0.999)
        q0_1 = data['value'].quantile(0.001)
        data = data[(data['value'] >= q0_1) & (data['value'] <= q99_9)].reset_index(drop=True)
        
        print(f"Data cleaned: {len(data)} records remaining")
        return data
    
    def add_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features to the dataset"""
        data = data.copy()
        
        # Time-based features
        data['hour'] = data['timestamp'].dt.hour
        data['day_of_week'] = data['timestamp'].dt.dayofweek
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
        
        # Rolling statistics (for each itemid separately)
        for itemid in data['itemid'].unique():
            mask = data['itemid'] == itemid
            item_data = data.loc[mask, 'value']
            
            # Rolling mean and std
            data.loc[mask, 'rolling_mean_1h'] = item_data.rolling(
                window=12, min_periods=1  # 12 points = 1 hour if 5min intervals
            ).mean()
            
            data.loc[mask, 'rolling_std_1h'] = item_data.rolling(
                window=12, min_periods=1
            ).std().fillna(0)
            
            data.loc[mask, 'rolling_mean_6h'] = item_data.rolling(
                window=72, min_periods=1  # 72 points = 6 hours
            ).mean()
            
            data.loc[mask, 'rolling_mean_24h'] = item_data.rolling(
                window=288, min_periods=1  # 288 points = 24 hours
            ).mean()
            
            # Value changes
            data.loc[mask, 'value_diff'] = item_data.diff().fillna(0)
            data.loc[mask, 'value_pct_change'] = item_data.pct_change().fillna(0)
            
            # Zscore within item
            mean_val = item_data.mean()
            std_val = item_data.std()
            if std_val > 0:
                data.loc[mask, 'zscore'] = (item_data - mean_val) / std_val
            else:
                data.loc[mask, 'zscore'] = 0
        
        # Fill any remaining NaN values
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data[numeric_columns] = data[numeric_columns].fillna(0)
        
        return data
    
    def create_threshold_scenarios(self, data: pd.DataFrame, 
                                 thresholds: List[float]) -> Dict[str, pd.DataFrame]:
        """Create different threshold scenarios for evaluation"""
        scenarios = {}
        
        for threshold in thresholds:
            scenario_data = data.copy()
            
            # Simulate alerts
            scenario_data['would_alert'] = (scenario_data['value'] > threshold).astype(int)
            
            # Add alert patterns
            scenario_data['alert_duration'] = 0
            scenario_data['time_since_last_alert'] = 0
            
            # Calculate alert durations and gaps for each item
            for itemid in scenario_data['itemid'].unique():
                mask = scenario_data['itemid'] == itemid
                item_data = scenario_data.loc[mask].copy()
                
                # Alert duration calculation
                alert_groups = (item_data['would_alert'] != 
                              item_data['would_alert'].shift()).cumsum()
                
                for group in alert_groups.unique():
                    group_mask = alert_groups == group
                    group_data = item_data.loc[group_mask]
                    
                    if len(group_data) > 0 and group_data['would_alert'].iloc[0] == 1:
                        duration = len(group_data)
                        scenario_data.loc[mask & (alert_groups == group), 'alert_duration'] = duration
                
                # Time since last alert
                last_alert_time = None
                for idx in item_data.index:
                    if scenario_data.loc[idx, 'would_alert'] == 1:
                        last_alert_time = scenario_data.loc[idx, 'timestamp']
                        scenario_data.loc[idx, 'time_since_last_alert'] = 0
                    elif last_alert_time is not None:
                        time_diff = (scenario_data.loc[idx, 'timestamp'] - last_alert_time).total_seconds() / 3600
                        scenario_data.loc[idx, 'time_since_last_alert'] = time_diff
            
            scenarios[f'threshold_{threshold}'] = scenario_data
        
        return scenarios
    
    def generate_synthetic_incidents(self, data: pd.DataFrame, 
                                   incident_rate: float = 0.05) -> pd.DataFrame:
        """Generate synthetic incident labels for training"""
        data = data.copy()
        
        # Simple heuristic for incident generation
        # Real incidents are more likely when:
        # 1. Values are significantly above normal
        # 2. Trend is increasing
        # 3. Multiple consecutive high values
        
        for itemid in data['itemid'].unique():
            mask = data['itemid'] == itemid
            item_data = data.loc[mask].copy()
            
            # Calculate percentiles for this item
            p95 = item_data['value'].quantile(0.95)
            p99 = item_data['value'].quantile(0.99)
            
            # Base incident probability
            incident_prob = np.zeros(len(item_data))
            
            # Higher probability for high values
            incident_prob += (item_data['value'] > p95).astype(float) * 0.3
            incident_prob += (item_data['value'] > p99).astype(float) * 0.4
            
            # Higher probability for increasing trends
            if 'value_diff' in item_data.columns:
                incident_prob += (item_data['value_diff'] > 0).astype(float) * 0.1
            
            # Higher probability during business hours (simplified)
            business_hours = (item_data['hour'] >= 9) & (item_data['hour'] <= 17)
            incident_prob += business_hours.astype(float) * 0.1
            
            # Generate incidents based on probability
            random_vals = np.random.random(len(item_data))
            incidents = (random_vals < incident_prob * incident_rate).astype(int)
            
            data.loc[mask, 'is_incident'] = incidents
        
        return data
    
    def split_data_temporal(self, data: pd.DataFrame, 
                          train_ratio: float = 0.7, 
                          val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data temporally for training/validation/test"""
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        n = len(data)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_data = data.iloc[:train_end].reset_index(drop=True)
        val_data = data.iloc[train_end:val_end].reset_index(drop=True)
        test_data = data.iloc[val_end:].reset_index(drop=True)
        
        print(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        return train_data, val_data, test_data
    
    def create_summary_report(self, data: pd.DataFrame) -> Dict:
        """Create a summary report of the data"""
        report = {
            'total_records': len(data),
            'unique_items': data['itemid'].nunique(),
            'time_range': {
                'start': data['timestamp'].min(),
                'end': data['timestamp'].max(),
                'duration_hours': (data['timestamp'].max() - data['timestamp'].min()).total_seconds() / 3600
            },
            'value_statistics': {
                'min': data['value'].min(),
                'max': data['value'].max(),
                'mean': data['value'].mean(),
                'median': data['value'].median(),
                'std': data['value'].std()
            }
        }
        
        if 'is_incident' in data.columns:
            report['incidents'] = {
                'total_incidents': data['is_incident'].sum(),
                'incident_rate': data['is_incident'].mean()
            }
        
        return report
    
    def visualize_data(self, data: pd.DataFrame, itemid: str = None, 
                      save_path: str = None) -> None:
        """Create visualizations of the data"""
        if itemid:
            plot_data = data[data['itemid'] == itemid].copy()
            title_suffix = f" (Item: {itemid})"
        else:
            # Use first item if none specified
            first_item = data['itemid'].iloc[0]
            plot_data = data[data['itemid'] == first_item].copy()
            title_suffix = f" (Item: {first_item})"
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Data Analysis{title_suffix}', fontsize=16)
        
        # Time series plot
        axes[0, 0].plot(plot_data['timestamp'], plot_data['value'])
        axes[0, 0].set_title('Value Over Time')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Value')
        
        # Value distribution
        axes[0, 1].hist(plot_data['value'], bins=50, alpha=0.7)
        axes[0, 1].set_title('Value Distribution')
        axes[0, 1].set_xlabel('Value')
        axes[0, 1].set_ylabel('Frequency')
        
        # Hourly pattern
        hourly_avg = plot_data.groupby('hour')['value'].mean()
        axes[1, 0].plot(hourly_avg.index, hourly_avg.values, marker='o')
        axes[1, 0].set_title('Average Value by Hour')
        axes[1, 0].set_xlabel('Hour of Day')
        axes[1, 0].set_ylabel('Average Value')
        
        # Weekly pattern
        weekly_avg = plot_data.groupby('day_of_week')['value'].mean()
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        axes[1, 1].bar(range(7), weekly_avg.values)
        axes[1, 1].set_title('Average Value by Day of Week')
        axes[1, 1].set_xlabel('Day of Week')
        axes[1, 1].set_ylabel('Average Value')
        axes[1, 1].set_xticks(range(7))
        axes[1, 1].set_xticklabels(days)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()