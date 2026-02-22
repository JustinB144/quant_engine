#!/bin/bash
# ============================================================
# Git Repository Setup Script for quant_engine
# Run this from your quant_engine root directory:
#   cd "/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine"
#   bash scripts/setup_git_repo.sh
# ============================================================

set -e

echo "=== Setting up Git repository for quant_engine ==="

# 1. Remove any broken .git from previous attempts (if exists)
if [ -d ".git" ]; then
    echo "Removing existing .git directory..."
    rm -rf .git
fi

# 2. Initialize fresh repo with 'main' as default branch
echo "Initializing Git repository..."
git init -b main

# 3. Stage all files (the .gitignore will handle exclusions)
echo "Staging files..."
git add -A

# 4. Show what will be committed
echo ""
echo "=== Files to be committed ==="
git status --short | head -40
TOTAL=$(git status --short | wc -l | tr -d ' ')
echo "... ($TOTAL files total)"
echo ""

# 5. Create initial commit
echo "Creating initial commit..."
git commit -m "Initial commit: quant_engine trading system

Complete quantitative trading framework including:
- Data layer: WRDS CRSP + IBKR providers with local parquet cache
- Data cache: 155 daily + 100 30min + 48 15min parquet files with meta.json
- Trained models: ensemble pkl files with regime-specific variants
- Features: intraday, macro, options, LOB, HARX spillovers
- Models: GBR/ElasticNet/RF ensemble with HMM regime blending
- Backtest: walk-forward engine with transaction cost modeling
- Risk: portfolio optimization, drawdown management, stress testing
- Kalshi: event contract pipeline with real-time pricing
- Autopilot: autonomous strategy discovery and paper trading
- Dashboard: Plotly Dash monitoring UI
- Tests: comprehensive test suite"

echo ""
echo "=== Repository initialized successfully ==="
echo ""
git log --oneline
echo ""

# 6. Instructions for connecting to GitHub
echo "=== Next Steps ==="
echo ""
echo "To share this with a collaborator via GitHub:"
echo ""
echo "  1. Create a NEW PRIVATE repo on GitHub (https://github.com/new)"
echo "     - Name it 'quant_engine'"
echo "     - Do NOT initialize with README, .gitignore, or license"
echo "     - Make it PRIVATE (this has trading strategies + credentials patterns)"
echo ""
echo "  2. Connect your local repo to GitHub:"
echo "     git remote add origin https://github.com/YOUR_USERNAME/quant_engine.git"
echo "     git push -u origin main"
echo ""
echo "  3. Invite your collaborator:"
echo "     Go to Settings > Collaborators > Add people"
echo ""
echo "  4. Your collaborator clones it:"
echo "     git clone https://github.com/YOUR_USERNAME/quant_engine.git"
echo "     cp .env.example .env   # then fill in their own credentials"
echo ""
echo "  NOTE: The data cache (~207MB), trained models (~7MB), and results"
echo "  are all included in the repo so your collaborator gets everything"
echo "  needed to run the system immediately."
echo ""
echo "=== Done ==="
