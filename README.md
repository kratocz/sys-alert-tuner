# Zabbix Threshold Tuner

AI-powered system for optimizing trigger thresholds in Zabbix monitoring using Reinforcement Learning.

## Overview

This project implements a Deep Q-Network (DQN) agent that learns to optimize alert thresholds based on historical monitoring data. The system aims to reduce false positives while maintaining coverage of real incidents.

## Features

- **Zabbix Integration**: Connects to Zabbix API to fetch historical metrics
- **Deep Reinforcement Learning**: Uses DQN to learn optimal threshold adjustments
- **Data Processing**: Advanced preprocessing with feature engineering
- **Performance Metrics**: Tracks accuracy, precision, recall, and F1-score
- **Visualization**: Training progress and data analysis plots

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Zabbix API    │───▶│  Data Processor  │───▶│ RL Environment  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Evaluation    │◀───│   DQN Agent      │───▶│ Threshold Tuner │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Installation

### Prerequisites

Install [uv](https://docs.astral.sh/uv/) - the fast Python package manager:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

### Project Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd sys-alert-tuner
```

2. Install dependencies and create virtual environment:
```bash
uv sync
```

3. Activate the environment:
```bash
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows
```

3. Configure Zabbix credentials:
```bash
cp .env.example .env
# Edit .env with your Zabbix server details
```

## Usage

### Training

Train the agent on historical data:
```bash
# Using uv
uv run sys-alert-tuner train --episodes 1000

# Or with activated environment
python main.py train --episodes 1000
```

### Evaluation

Evaluate a trained model:
```bash
# Using uv
uv run sys-alert-tuner evaluate

# Or with activated environment
python main.py evaluate
```

### Compare Thresholds

Compare different threshold strategies:
```bash
# Using uv
uv run sys-alert-tuner compare

# Or with activated environment
python main.py compare
```

## Configuration

Environment variables in `.env`:
- `ZABBIX_URL`: Zabbix server URL
- `ZABBIX_USER`: Username for Zabbix API
- `ZABBIX_PASSWORD`: Password for Zabbix API
- `TRAINING_EPISODES`: Number of training episodes
- `LEARNING_RATE`: Agent learning rate
- `BATCH_SIZE`: Training batch size

## Components

### ZabbixClient (`src/zabbix_client.py`)
- Handles Zabbix API connections
- Fetches hosts, items, and historical data
- Manages authentication and data retrieval

### DataProcessor (`src/data_processor.py`)
- Cleans and preprocesses historical data
- Generates engineered features
- Creates synthetic incident labels
- Provides data visualization

### ThresholdTuningEnvironment (`src/threshold_environment.py`)
- Gymnasium-compatible RL environment
- Simulates threshold adjustment scenarios
- Calculates rewards based on alert performance
- Tracks false positives/negatives

### DQNAgent (`src/dqn_agent.py`)
- Deep Q-Network implementation
- Experience replay buffer
- Target network for stability
- Epsilon-greedy exploration

### Trainer (`src/trainer.py`)
- Main training orchestration
- Model saving/loading
- Performance evaluation
- Progress visualization

## Metrics

The system optimizes for:
- **Accuracy**: Overall correctness of alert decisions
- **Precision**: Proportion of relevant alerts
- **Recall**: Coverage of actual incidents
- **F1-Score**: Harmonic mean of precision and recall
- **False Positive Rate**: Rate of unnecessary alerts
- **False Negative Rate**: Rate of missed incidents

## Model Output

After training, the system provides:
- Recommended threshold values
- Performance metrics
- Training progress plots
- Saved model files for reuse

## Demo Mode

If Zabbix connection fails, the system automatically generates synthetic data for demonstration purposes.

## File Structure

```
sys-alert-tuner/
├── main.py                      # Main entry point
├── pyproject.toml              # Project configuration and dependencies
├── uv.lock                     # Dependency lock file
├── .python-version             # Python version specification
├── .env.example               # Environment template
├── README.md                  # This file
├── sys_alert_tuner/           # Main package
│   ├── __init__.py
│   ├── zabbix_client.py       # Zabbix API integration
│   ├── data_processor.py      # Data preprocessing
│   ├── threshold_environment.py  # RL environment
│   ├── dqn_agent.py          # DQN implementation
│   └── trainer.py            # Training orchestration
├── models/                    # Saved models (created during training)
├── plots/                     # Generated visualizations
└── tests/                     # Test files (optional)
```

## Development

### Installing Development Dependencies

```bash
uv sync --all-extras
```

### Running Tests

```bash
uv run pytest
```

### Code Formatting

```bash
uv run black sys_alert_tuner/
```

### Type Checking

```bash
uv run mypy sys_alert_tuner/
```

### Adding Dependencies

```bash
# Add runtime dependency
uv add numpy

# Add development dependency
uv add --dev pytest
```

## Future Enhancements

- Support for multiple threshold types
- Integration with Prometheus and Elasticsearch
- Advanced anomaly detection
- Multi-agent threshold coordination
- Web-based dashboard
- A/B testing framework