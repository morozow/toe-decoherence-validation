"""
Standard reporting utilities for all ToE experiments.

Provides save_experiment_results() and format_verdict() to generate
consistent output files across all 18 experiments:
  - RESULTS.txt   — human-readable summary with verdict
  - data/parameters.json — machine-readable parameters and metrics
  - data/*.csv    — numerical data arrays

Reference: sec13; Requirements 18.2, 18.3, 18.4
"""

import csv
import json
import os
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple, Union

import numpy as np


# Valid verdict strings
_VALID_VERDICTS = {"PASS", "FAIL", "INCONCLUSIVE", "CONCEPTUAL"}


def format_verdict(test_name: str, passed: bool, details: str = "") -> str:
    """
    Format a PASS/FAIL verdict line for RESULTS.txt.

    Parameters
    ----------
    test_name : str
        Short name of the test or check (e.g. "ghost-freedom").
    passed : bool
        Whether the test passed.
    details : str, optional
        Extra information (metrics, thresholds, etc.).

    Returns
    -------
    str
        Formatted line, e.g. ``"PASS  ghost-freedom — α₃ ≥ 0 satisfied"``.
    """
    tag = "PASS" if passed else "FAIL"
    line = f"{tag}  {test_name}"
    if details:
        line += f" — {details}"
    return line


def save_experiment_results(
    exp_dir: str,
    summary: str,
    verdict: str,
    params: dict,
    csv_data: Optional[Dict[str, Tuple]] = None,
    key_result: str = "",
    manuscript_ref: str = "",
) -> None:
    """
    Save the standard set of output files for an experiment.

    Creates (inside *exp_dir*):
      - ``RESULTS.txt``          — human-readable report with verdict
      - ``data/parameters.json`` — full parameter set, verdict, metrics
      - ``data/<name>.csv``      — one file per entry in *csv_data*

    Directories ``data/`` and ``plots/`` are created if they do not exist.

    Parameters
    ----------
    exp_dir : str
        Root directory of the experiment (e.g. ``experiments/exp03_…``).
    summary : str
        Multi-line human-readable summary for RESULTS.txt.
    verdict : str
        One of ``"PASS"``, ``"FAIL"``, ``"INCONCLUSIVE"``, ``"CONCEPTUAL"``.
    params : dict
        Parameters dict to persist.  May contain nested dicts and
        numpy scalars (they are converted to plain Python types).
        A ``"metrics"`` sub-dict is encouraged but not required.
    csv_data : dict, optional
        Mapping ``{filename: (header_list, 2-d array)}``.
        Each entry produces ``data/<filename>.csv``.
    key_result : str, optional
        One-line key result string for the summary table.
    manuscript_ref : str, optional
        Manuscript section references (e.g. ``"sec03, eq:consistency"``).

    Raises
    ------
    ValueError
        If *verdict* is not one of the four allowed strings.
    """
    verdict_upper = verdict.strip().upper()
    if verdict_upper not in _VALID_VERDICTS:
        raise ValueError(
            f"Invalid verdict '{verdict}'. Must be one of {_VALID_VERDICTS}."
        )

    # --- ensure directories ---
    data_dir = os.path.join(exp_dir, "data")
    plots_dir = os.path.join(exp_dir, "plots")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # --- RESULTS.txt ---
    _write_results_txt(exp_dir, summary, verdict_upper, key_result, manuscript_ref)

    # --- data/parameters.json ---
    _write_parameters_json(
        data_dir, exp_dir, verdict_upper, params, key_result, manuscript_ref,
    )

    # --- data/*.csv ---
    if csv_data:
        _write_csv_files(data_dir, csv_data)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _write_results_txt(
    exp_dir: str,
    summary: str,
    verdict: str,
    key_result: str,
    manuscript_ref: str,
) -> None:
    """Write the human-readable RESULTS.txt."""
    path = os.path.join(exp_dir, "RESULTS.txt")
    lines = []
    lines.append(f"VERDICT: {verdict}")
    if key_result:
        lines.append(f"KEY RESULT: {key_result}")
    if manuscript_ref:
        lines.append(f"MANUSCRIPT REF: {manuscript_ref}")
    lines.append("")
    lines.append(summary.rstrip())
    lines.append("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def _numpy_safe(obj):
    """Recursively convert numpy types to plain Python for JSON serialisation."""
    if isinstance(obj, dict):
        return {k: _numpy_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_numpy_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _write_parameters_json(
    data_dir: str,
    exp_dir: str,
    verdict: str,
    params: dict,
    key_result: str,
    manuscript_ref: str,
) -> None:
    """Write data/parameters.json in the ExperimentResult format."""
    exp_name = os.path.basename(os.path.normpath(exp_dir))

    # Separate metrics from params if present
    metrics = params.pop("metrics", {})
    category = params.pop("category", "")

    doc = {
        "experiment": exp_name,
        "category": category,
        "verdict": verdict,
        "key_result": key_result,
        "manuscript_ref": manuscript_ref,
        "parameters": _numpy_safe(params),
        "metrics": _numpy_safe(metrics),
    }

    path = os.path.join(data_dir, "parameters.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)


def _write_csv_files(data_dir: str, csv_data: dict) -> None:
    """Write each entry in *csv_data* as a CSV file under *data_dir*."""
    for filename, (header, array) in csv_data.items():
        if not filename.endswith(".csv"):
            filename += ".csv"
        path = os.path.join(data_dir, filename)

        arr = np.asarray(array)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            if arr.ndim == 1:
                for val in arr:
                    writer.writerow([val])
            else:
                for row in arr:
                    writer.writerow(row)
