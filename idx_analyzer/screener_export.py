"""
Export utilities for screener results.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


def export_screener_results(
    df: pd.DataFrame,
    format_type: str,
    output_path: Optional[str] = None,
) -> str:
    """
    Export screener results to file.

    Args:
        df: DataFrame with screener results
        format_type: 'csv' or 'json'
        output_path: Custom output path (optional)

    Returns:
        Path to exported file
    """
    if df.empty:
        print("No results to export.")
        return ""

    # Generate default filename
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screener_results_{timestamp}.{format_type}"
        output_dir = Path("exports") / "screeners"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / filename)

    if format_type == "csv":
        df.to_csv(output_path, index=False)
    elif format_type == "json":
        # Convert to dict for better JSON structure
        results_dict = {
            "generated_at": datetime.now().isoformat(),
            "total_results": len(df),
            "stocks": df.to_dict("records"),
        }
        with open(output_path, "w") as f:
            json.dump(results_dict, f, indent=2, default=str)
    else:
        raise ValueError(f"Unsupported format: {format_type}")

    print(f"Screener results exported to: {output_path}")
    return output_path
