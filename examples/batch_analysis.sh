#!/bin/bash
# =============================================================================
# Batch Analysis Example
# =============================================================================
# This script demonstrates how to analyze multiple stocks and save results
# to a file for further analysis.
# 
# Usage:
#   chmod +x batch_analysis.sh
#   ./batch_analysis.sh
# =============================================================================

# Define your stock portfolio
PORTFOLIO=(
    "BBCA"  # Bank Central Asia
    "BBRI"  # Bank Rakyat Indonesia
    "BMRI"  # Bank Mandiri
    "TLKM"  # Telkom Indonesia
    "ASII"  # Astra International
    "UNVR"  # Unilever Indonesia
)

# Output file
OUTPUT_FILE="portfolio_analysis_$(date +%Y%m%d_%H%M%S).txt"

echo "========================================"
echo "IDX Stock Portfolio Analysis"
echo "Generated: $(date)"
echo "========================================"
echo ""

# Analyze each stock
for stock in "${PORTFOLIO[@]}"; do
    echo "Analyzing $stock..."
    echo "----------------------------------------"
    uv run idx-analyzer "$stock" --quiet
    echo ""
done | tee "$OUTPUT_FILE"

echo ""
echo "========================================"
echo "Analysis complete!"
echo "Results saved to: $OUTPUT_FILE"
echo "========================================"
