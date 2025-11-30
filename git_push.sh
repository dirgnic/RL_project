#!/bin/bash
# Script to commit and push all changes to GitHub

echo "======================================"
echo "Git Commit and Push Script"
echo "======================================"
echo ""

# Navigate to project directory
cd /Users/ingridcorobana/Desktop/An_III/final_projs/RL_proj

# Show status
echo "Current git status:"
git status
echo ""

# Add all files
echo "Adding all files..."
git add -A
echo ""

# Show what will be committed
echo "Files to be committed:"
git status --short
echo ""

# Check if there are changes to commit
if git diff --cached --quiet; then
    echo "✓ No new changes to commit (everything already committed)"
else
    # Commit with message
    echo "Committing changes..."
    git commit -m "Add complete Q-Learning implementation and project structure

- Full Q-Learning agent with Q-Table and discretization
- Reward shaping wrappers (CartPole, MountainCar, LunarLander)
- Complete utils: logging, plotting, metrics, seed management
- 6 experiment configurations ready to run
- Comprehensive tests (all passing)
- Documentation: README, SETUP, PROJECT_STATUS, COMMANDS
- Virtual environment (.venv) ignored in .gitignore"
    echo ""
fi

# Show commits ready to push
echo "Commits ready to push:"
git log origin/main..HEAD --oneline
echo ""

# Push to GitHub
echo "Pushing to GitHub..."
echo "You will be asked for your SSH passphrase..."
echo ""
git push origin main

echo ""
echo "======================================"
echo "✓ Done!"
echo "======================================"
