#!/bin/bash
# Sys Alert Tuner - Demo mode with synthetic data
#
# This script trains the AI agent using synthetic monitoring data.
# No Zabbix server required - perfect for testing and demonstration.

echo "🎬 Starting Sys Alert Tuner DEMO mode..."
echo "🔬 Training with synthetic monitoring data (no Zabbix server needed)"
echo

# Check if uv is available
if command -v uv &> /dev/null; then
    echo "📦 Using uv package manager..."
    uv run python main.py demo "$@"
elif command -v python &> /dev/null; then
    echo "🐍 Using system Python..."
    python main.py demo "$@"
else
    echo "❌ Error: Neither uv nor python found!"
    echo "💡 Please install uv or ensure Python is in your PATH"
    exit 1
fi

echo
echo "🎯 Demo completed! Check the generated files:"
echo "   📊 models/ - Trained AI models"
echo "   📈 plots/ - Training progress visualizations"