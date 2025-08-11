# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Nothing yet

## [0.2.0] - 2025-08-12

### Added
- **Demo Mode** - New `demo` command for testing with synthetic data
- **Shell Scripts for Linux/macOS**:
  - `train-zabbix.sh` - Production mode launcher with error handling
  - `train-demo.sh` - Demo mode launcher for quick testing
- **Separated Training Modes**:
  - Production mode (`train`) - Real Zabbix data only, fails gracefully
  - Demo mode (`demo`) - Synthetic data, no Zabbix server required
- **Performance Optimizations**:
  - Fixed PyTorch tensor creation warnings (numpy array conversion)
  - Updated pandas deprecation warnings (freq parameter)
  - Apple Silicon GPU (Metal) support for faster training
- **Enhanced Error Handling**:
  - Production mode no longer falls back to demo data automatically
  - Helpful error messages with demo mode suggestions
  - Shell script validation for `.env` file existence

### Changed
- **Breaking Change**: `train` command no longer uses synthetic data fallback
- **CLI Interface**: Added `demo` as fourth command option
- **User Experience**: Clear separation between production and demo workflows
- **Documentation**: Updated README.md with distinct usage patterns for both modes

### Technical Improvements
- **GPU Acceleration**: Metal Performance Shaders support for Apple Silicon
- **Tensor Operations**: Optimized PyTorch tensor creation for better performance
- **Authentication Flow**: Improved Zabbix API connection handling
- **Error Messages**: More informative feedback for connection failures

### Documentation
- **Updated README**: Clear production vs demo mode documentation
- **Shell Scripts**: Included in project structure and usage examples
- **File Structure**: Updated to reflect new launcher scripts

## [0.1.0] - 2025-08-12

### Added
- **Complete Reinforcement Learning System** for Zabbix threshold optimization
- **Deep Q-Network (DQN) Agent** with experience replay and target network
- **Zabbix API Integration** supporting both API tokens (Zabbix 5.4+) and username/password authentication
- **Advanced Data Processing Pipeline**:
  - Historical data cleaning and validation
  - Feature engineering (rolling statistics, time-based features, Z-scores)
  - Synthetic incident generation for training
  - Data visualization and analysis tools
- **Reinforcement Learning Environment**:
  - Gymnasium-compatible interface
  - Threshold adjustment simulation
  - Reward system based on false positive/negative optimization
  - Performance metrics tracking (accuracy, precision, recall, F1-score)
- **Training and Evaluation Framework**:
  - Complete training pipeline with progress visualization
  - Model saving/loading functionality
  - Evaluation tools with detailed metrics
  - Threshold comparison utilities
- **Command-Line Interface** with three main modes:
  - `train` - Train RL agent on historical data
  - `evaluate` - Evaluate trained models
  - `compare` - Compare different threshold strategies
- **Demo Capabilities**:
  - Synthetic data generation when Zabbix unavailable
  - Sample scenarios for testing and demonstration
- **Modern Python Development Stack**:
  - uv package manager for fast dependency management
  - pyproject.toml-based project configuration
  - Comprehensive test suite with pytest
  - Code quality tools (black, mypy, flake8)
  - Pre-commit hooks support
  - Professional project structure

### Technical Architecture
- **Environment**: Custom Gymnasium environment for threshold tuning simulation
- **Agent**: DQN with neural network, epsilon-greedy exploration, and replay buffer
- **Metrics**: Multi-objective optimization (minimize false positives/negatives)
- **Visualization**: Matplotlib-based training progress and data analysis plots
- **Authentication**: Flexible Zabbix authentication (API tokens preferred)
- **Package Management**: Modern uv-based dependency resolution and virtual environments

### Security
- **Credential Management**: Environment variable-based configuration
- **API Token Support**: Secure authentication for Zabbix 5.4+
- **No Hardcoded Secrets**: All sensitive data externalized
- **Safe Defaults**: Secure configuration examples

### Documentation
- **Comprehensive README**: Installation, usage, and architecture documentation
- **API Documentation**: Detailed component descriptions
- **Example Configuration**: Complete .env.example with all options
- **Development Guide**: Setup instructions for contributors
- **Architecture Diagrams**: Visual system overview

### Developer Experience
- **Fast Setup**: One-command environment setup with `uv sync`
- **Modern Tooling**: Latest Python packaging standards
- **Quality Assurance**: Automated testing and code formatting
- **Type Safety**: MyPy type checking integration
- **Documentation**: Inline code documentation and examples

---

## Release Notes Format

### Types of Changes
- **Added** for new features
- **Changed** for changes in existing functionality
- **Deprecated** for soon-to-be removed features
- **Removed** for now removed features
- **Fixed** for any bug fixes
- **Security** for vulnerability fixes

### Version Numbering
- **Major** version when you make incompatible API changes
- **Minor** version when you add functionality in a backwards compatible manner
- **Patch** version when you make backwards compatible bug fixes