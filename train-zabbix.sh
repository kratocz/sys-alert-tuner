#!/bin/bash
# Sys Alert Tuner - Train with real Zabbix data
#
# This script trains the AI agent using real data from your Zabbix server.
# Make sure you have configured your .env file with Zabbix credentials.

echo "🚀 Starting Sys Alert Tuner with real Zabbix data..."
echo "📋 Make sure your .env file is configured with Zabbix credentials!"
echo

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "❌ Error: .env file not found!"
    echo "💡 Please copy .env.example to .env and configure your Zabbix settings:"
    echo "   cp .env.example .env"
    echo "   # Then edit .env with your Zabbix server details"
    exit 1
fi

# Check if uv is available
if command -v uv &> /dev/null; then
    echo "📦 Using uv package manager..."
    uv run python main.py train "$@"
elif command -v python &> /dev/null; then
    echo "🐍 Using system Python..."
    python main.py train "$@"
else
    echo "❌ Error: Neither uv nor python found!"
    echo "💡 Please install uv or ensure Python is in your PATH"
    exit 1
fi