"""
VM resource monitor — run this BEFORE starting sentiment_analysis.py.
Logs CPU, RAM, GPU, and disk I/O every INTERVAL seconds to vm_metrics.csv.
Stop with Ctrl+C; a summary table is printed on exit.

Usage:
    python vm_monitor.py               # default 2-second interval
    python vm_monitor.py --interval 1  # 1-second interval
    python vm_monitor.py --output my_run.csv
"""

import argparse
import csv
import os
import signal
import subprocess
import sys
import time
from datetime import datetime

try:
    import psutil
except ImportError:
    sys.exit("psutil not found. Run: pip install psutil")

# ── CONFIG ────────────────────────────────────────────────────────────────────
DEFAULT_INTERVAL  = 2      # seconds between samples
DEFAULT_OUTPUT    = "vm_metrics.csv"
GPU_QUERY_FIELDS  = [
    "index",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "temperature.gpu",
    "power.draw",
]

# ── GPU ───────────────────────────────────────────────────────────────────────
def nvidia_smi_available() -> bool:
    try:
        subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def query_gpus() -> list[dict]:
    """Return a list of dicts, one per GPU."""
    query = ",".join(GPU_QUERY_FIELDS)
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == len(GPU_QUERY_FIELDS):
                gpus.append(dict(zip(GPU_QUERY_FIELDS, parts)))
        return gpus
    except Exception:
        return []


# ── SAMPLING ──────────────────────────────────────────────────────────────────
def sample(has_gpu: bool, prev_disk) -> tuple[dict, object]:
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    cpu  = psutil.cpu_percent(interval=None)
    per_core = psutil.cpu_percent(interval=None, percpu=True)
    ram  = psutil.virtual_memory()
    disk_now = psutil.disk_io_counters()

    # Disk I/O rates (bytes/s since last sample — 0 on first call)
    disk_read_rate  = 0.0
    disk_write_rate = 0.0
    if prev_disk is not None:
        dt = DEFAULT_INTERVAL  # approximate; caller tracks real elapsed
        disk_read_rate  = (disk_now.read_bytes  - prev_disk.read_bytes)  / dt / 1024 / 1024
        disk_write_rate = (disk_now.write_bytes - prev_disk.write_bytes) / dt / 1024 / 1024

    row = {
        "timestamp":        ts,
        "cpu_pct":          cpu,
        "cpu_cores":        ",".join(str(c) for c in per_core),
        "ram_used_gb":      round(ram.used / 1024**3, 2),
        "ram_total_gb":     round(ram.total / 1024**3, 2),
        "ram_pct":          ram.percent,
        "disk_read_mbps":   round(disk_read_rate, 2),
        "disk_write_mbps":  round(disk_write_rate, 2),
    }

    # GPU columns (up to 4 GPUs; pad with empty if fewer)
    if has_gpu:
        gpus = query_gpus()
        for i in range(4):
            pfx = f"gpu{i}_"
            if i < len(gpus):
                g = gpus[i]
                row[pfx + "util_pct"]   = g.get("utilization.gpu",    "")
                row[pfx + "mem_pct"]    = g.get("utilization.memory",  "")
                row[pfx + "mem_used_mb"]= g.get("memory.used",         "")
                row[pfx + "mem_total_mb"]= g.get("memory.total",       "")
                row[pfx + "temp_c"]     = g.get("temperature.gpu",     "")
                row[pfx + "power_w"]    = g.get("power.draw",          "")
            else:
                for sfx in ["util_pct","mem_pct","mem_used_mb","mem_total_mb","temp_c","power_w"]:
                    row[pfx + sfx] = ""

    return row, disk_now


# ── SUMMARY ───────────────────────────────────────────────────────────────────
def print_summary(rows: list[dict], has_gpu: bool, elapsed: float):
    if not rows:
        return

    def col_floats(key):
        return [float(r[key]) for r in rows if r.get(key) not in ("", None)]

    def stats(vals):
        if not vals:
            return "n/a"
        return f"avg={sum(vals)/len(vals):.1f}  peak={max(vals):.1f}"

    print("\n" + "=" * 60)
    print("  VM Monitor — Run Summary")
    print("=" * 60)
    print(f"  Duration  : {elapsed:.1f}s   Samples: {len(rows)}")
    print(f"  CPU %     : {stats(col_floats('cpu_pct'))}")
    print(f"  RAM used  : {stats(col_floats('ram_used_gb'))} GB")
    print(f"  RAM %     : {stats(col_floats('ram_pct'))}")

    if has_gpu:
        for i in range(4):
            util = col_floats(f"gpu{i}_util_pct")
            if util:
                mem  = col_floats(f"gpu{i}_mem_used_mb")
                temp = col_floats(f"gpu{i}_temp_c")
                pwr  = col_floats(f"gpu{i}_power_w")
                print(f"  GPU {i} util : {stats(util)} %")
                print(f"  GPU {i} VRAM : {stats(mem)} MB")
                print(f"  GPU {i} temp : {stats(temp)} °C")
                print(f"  GPU {i} power: {stats(pwr)} W")
    print("=" * 60)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="VM resource monitor")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL,
                        help="Sampling interval in seconds (default: 2)")
    parser.add_argument("--output",   default=DEFAULT_OUTPUT,
                        help=f"Output CSV path (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    has_gpu = nvidia_smi_available()

    print("=" * 60)
    print("  VM Resource Monitor")
    print("=" * 60)
    print(f"  Interval : {args.interval}s")
    print(f"  Output   : {args.output}")
    print(f"  GPU      : {'detected via nvidia-smi' if has_gpu else 'not detected'}")
    print(f"  CPU cores: {psutil.cpu_count()}")
    ram = psutil.virtual_memory()
    print(f"  RAM total: {ram.total / 1024**3:.1f} GB")
    print("=" * 60)
    print("Logging started. Press Ctrl+C to stop.\n")

    rows      = []
    prev_disk = None
    start     = time.time()
    fieldnames = None

    # Warm up cpu_percent (first call always returns 0)
    psutil.cpu_percent(interval=None)
    psutil.cpu_percent(interval=None, percpu=True)

    def handle_stop(sig, frame):
        pass  # just break the loop cleanly

    signal.signal(signal.SIGTERM, handle_stop)

    try:
        with open(args.output, "w", newline="") as csvfile:
            writer = None

            while True:
                loop_start = time.time()
                row, prev_disk = sample(has_gpu, prev_disk)
                rows.append(row)

                # Write header on first row
                if writer is None:
                    fieldnames = list(row.keys())
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()

                writer.writerow(row)
                csvfile.flush()

                # Print a live one-liner
                gpu_str = ""
                if has_gpu and f"gpu0_util_pct" in row and row["gpu0_util_pct"] != "":
                    gpu_str = f"  GPU0 {row['gpu0_util_pct']}% VRAM {int(float(row['gpu0_mem_used_mb'])/1024*10)/10:.1f}GB"
                print(
                    f"[{row['timestamp']}]  "
                    f"CPU {row['cpu_pct']:5.1f}%  "
                    f"RAM {row['ram_used_gb']:.1f}/{row['ram_total_gb']:.0f}GB ({row['ram_pct']:.0f}%)"
                    f"{gpu_str}",
                    flush=True,
                )

                # Sleep for the remainder of the interval
                elapsed_loop = time.time() - loop_start
                sleep_time = max(0, args.interval - elapsed_loop)
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass

    total_elapsed = time.time() - start
    print(f"\nStopped. {len(rows)} samples written to {args.output}")
    print_summary(rows, has_gpu, total_elapsed)


if __name__ == "__main__":
    main()
