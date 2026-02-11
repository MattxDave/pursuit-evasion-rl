#!/bin/bash
# Setup script for pushing to GitHub

echo "🚀 Pursuit-Evasion RL - GitHub Setup"
echo "====================================="
echo ""

# Check if git is configured
if ! git config user.name > /dev/null 2>&1; then
    echo "⚠️  Git user not configured. Please run:"
    echo "   git config --global user.name \"Your Name\""
    echo "   git config --global user.email \"your.email@example.com\""
    echo ""
    exit 1
fi

echo "✓ Git configured as: $(git config user.name) <$(git config user.email)>"
echo ""

# Prompt for repository name
read -p "Enter your GitHub username: " github_username
read -p "Enter repository name [pursuit-evasion-rl]: " repo_name
repo_name=${repo_name:-pursuit-evasion-rl}

echo ""
echo "📋 Next steps:"
echo ""
echo "1. Create a new repository on GitHub:"
echo "   → Go to: https://github.com/new"
echo "   → Repository name: $repo_name"
echo "   → Description: Multi-agent pursuit-evasion with hybrid RL"
echo "   → Visibility: Choose public or private"
echo "   → DO NOT initialize with README, .gitignore, or license"
echo ""
echo "2. After creating the repository on GitHub, run these commands:"
echo ""
echo "   cd /Users/Matthew/Uarc\(PE\)/pursuit_hybrid_expanded_20251214"
echo "   git remote add origin https://github.com/$github_username/$repo_name.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "✨ Your repository is ready to push!"
echo ""
echo "Optional: To use SSH instead of HTTPS:"
echo "   git remote set-url origin git@github.com:$github_username/$repo_name.git"
echo ""
