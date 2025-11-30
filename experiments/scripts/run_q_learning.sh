#!/bin/bash
# Run all Q-Learning experiments

echo "================================"
echo "Running Q-Learning Experiments"
echo "================================"

# Set project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# CartPole experiments
echo ""
echo "--- CartPole-v1 ---"
echo "Running Q-Learning on CartPole with default reward..."
python agents/q_learning/train.py --config experiments/configs/q_learning_cartpole_default.json

echo ""
echo "Running Q-Learning on CartPole with modified reward (bonus_center)..."
python agents/q_learning/train.py --config experiments/configs/q_learning_cartpole_modified.json

# MountainCar experiments
echo ""
echo "--- MountainCar-v0 ---"
echo "Running Q-Learning on MountainCar with default reward..."
python agents/q_learning/train.py --config experiments/configs/q_learning_mountaincar_default.json

echo ""
echo "Running Q-Learning on MountainCar with modified reward (potential_based)..."
python agents/q_learning/train.py --config experiments/configs/q_learning_mountaincar_modified.json

# LunarLander experiments
echo ""
echo "--- LunarLander-v2 ---"
echo "Running Q-Learning on LunarLander with default reward..."
python agents/q_learning/train.py --config experiments/configs/q_learning_lunarlander_default.json

echo ""
echo "Running Q-Learning on LunarLander with modified reward (fuel_efficiency)..."
python agents/q_learning/train.py --config experiments/configs/q_learning_lunarlander_modified.json

echo ""
echo "================================"
echo "All Q-Learning experiments completed!"
echo "================================"
