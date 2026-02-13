#!/bin/bash
# Installation script for IDX Stock Analyzer

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         IDX Stock Analyzer - Installation Script             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "📋 uv version: $(uv --version)"

# Sync dependencies
echo ""
echo "📦 Syncing dependencies with uv..."
uv sync

echo ""
echo "✅ Installation complete!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 You can now use: uv run idx-analyzer"
echo ""
echo "📖 Quick start:"
echo "   uv run idx-analyzer BBCA"
echo "   uv run idx-analyzer TLKM --period 1y"
echo "   uv run idx-analyzer --help"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Test the installation
echo "🧪 Testing installation..."
uv run idx-analyzer --version
