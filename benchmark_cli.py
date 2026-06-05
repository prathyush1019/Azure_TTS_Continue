# benchmark_cli.py
"""Azure TTS latency benchmark (CLI)
Runs the test cases defined in ``input.json`` (10 phrases per word‑count group 1‑10)
and writes detailed latency reports.

Outputs:
- ``latency_report/<group>.json`` – per‑group results (avg/min/max + all runs)
- ``latency_report_summary.json`` – summary across all groups (min/avg/max per group)
- ``latency_report/latency_report.json`` – same as the original Cartesia version (optional)
"""

import os
import json
import time
import threading
from latency_tracker import LatencyTracker
from tts_service import AzureStreamer
from config import SAMPLE_RATE

# ---------------------------------------------------------------------------
# Helper to run a single phrase through AzureStreamer and collect latency
# ---------------------------------------------------------------------------
def run_phrase(tts: AzureStreamer, phrase: str) -> tuple[float, list[bytes]]:
    """Synthesize *phrase* using AzureStreamer and return latency (ms) and audio bytes.
    This bypasses the streaming receiver thread and uses the SDK's audio_data directly.
    """
    tracker = LatencyTracker(phrase_text=phrase)
    # Mark the start of the request
    tracker.mark("Start")
    tracker.mark("text_send_start")
    # Direct synthesis returns audio bytes
    audio_bytes = tts.synthesize(phrase)
    tracker.mark("first_chunk_received")
    tracker.mark("First Audio Chunk")
    latency_ms = tracker.get_metrics()["speech_generation_latency_ms"]
    return latency_ms, [audio_bytes]

# ---------------------------------------------------------------------------
# Main benchmarking routine
# ---------------------------------------------------------------------------
def main():
    input_file = "input.json"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found")
        return

    with open(input_file, "r") as f:
        data = json.load(f)
    test_cases = data.get("test_cases", {})
    if not test_cases:
        print("No test_cases found in input.json")
        return

    # Sort groups numerically ("1_word", "2_words", ...)
    groups = sorted(
        test_cases.keys(),
        key=lambda k: int(k.split('_')[0]) if k.split('_')[0].isdigit() else 999,
    )

    # Prepare output directory
    out_dir = "latency_report"
    os.makedirs(out_dir, exist_ok=True)

    # Store full report (including per‑run details) – optional but kept for parity
    full_report = {"test_cases": {}}
    summary_report = {}

    with AzureStreamer() as tts:
        for group in groups:
            phrases = test_cases[group]
            latencies: list[float] = []
            runs: list[dict] = []
            for idx, phrase in enumerate(phrases):
                latency, _ = run_phrase(tts, phrase)
                if latency is not None:
                    latencies.append(latency)
                    runs.append({"phrase": phrase, "latency_ms": round(latency, 2), "status": "success"})
                else:
                    runs.append({"phrase": phrase, "latency_ms": None, "status": "failed"})
                # Small pause to avoid hitting rate limits
                time.sleep(0.05)

            # Save per‑group JSON file
            group_path = os.path.join(out_dir, f"{group}.json")
            group_data: dict = {"runs": runs}
            if latencies:
                group_data.update({
                    "avg_ms": round(sum(latencies) / len(latencies), 2),
                    "min_ms": round(min(latencies), 2),
                    "max_ms": round(max(latencies), 2),
                })
                # Populate summary for the second report
                summary_report[group] = {
                    "avg_ms": group_data["avg_ms"],
                    "min_ms": group_data["min_ms"],
                    "max_ms": group_data["max_ms"],
                }
            else:
                summary_report[group] = {"avg_ms": None, "min_ms": None, "max_ms": None}

            with open(group_path, "w", encoding="utf-8") as gf:
                json.dump(group_data, gf, indent=2)

            # Also add to the full report structure
            full_report["test_cases"][group] = group_data

    # Write the overall summary file
    summary_path = os.path.join(out_dir, "latency_report_summary.json")
    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(summary_report, sf, indent=2)

    # Write the raw full report (mirrors original Cartesia version)
    full_path = os.path.join(out_dir, "latency_report.json")
    with open(full_path, "w", encoding="utf-8") as ff:
        json.dump(full_report, ff, indent=2)

    print(f"Benchmark completed. Reports written to ./{out_dir}/")

if __name__ == "__main__":
    main()
