"""
VM resource monitor — zero external dependencies, Linux only.
Reads CPU/RAM from /proc, GPU from nvidia-smi, disk from /proc/diskstats.

Usage:
    python3 vm_monitor.py                   # 2s interval, writes vm_metrics.csv
    python3 vm_monitor.py --interval 1      # 1s interval
    python3 vm_monitor.py --output run1.csv
"""

import argparse
import csv
import os
import signal
import subprocess
import sys
import time
from datetime import datetime

# ── CPU (/proc/stat) ──────────────────────────────────────────────────────────
def read_cpu_stat() -> list[list[int]]:
    """Return raw cpu field lists from /proc/stat (overall + per-core)."""
    lines = []
    with open("/proc/stat") as f:
        for line in f:
            if line.startswith("cpu"):
                parts = line.split()
                lines.append([int(x) for x in parts[1:8]])  # user nice sys idle iowait irq softirq
    return lines  # index 0 = aggregate, 1.. = per core


_prev_cpu = None

def cpu_percent() -> tuple[float, list[float]]:
    """Return (overall_pct, [per_core_pct]) since last call."""
    global _prev_cpu
    curr = read_cpu_stat()

    if _prev_cpu is None:
        _prev_cpu = curr
        return 0.0, [0.0] * (len(curr) - 1)

    def pct(prev, cur):
        prev_idle = prev[3] + prev[4]          # idle + iowait
        curr_idle = cur[3]  + cur[4]
        prev_total = sum(prev)
        curr_total = sum(cur)
        d_total = curr_total - prev_total
        d_idle  = curr_idle  - prev_idle
        if d_total == 0:
            return 0.0
        return round((1 - d_idle / d_total) * 100, 1)

    overall   = pct(_prev_cpu[0], curr[0])
    per_core  = [pct(_prev_cpu[i], curr[i]) for i in range(1, len(curr))]
    _prev_cpu = curr
    return overall, per_core


# ── RAM (/proc/meminfo) ───────────────────────────────────────────────────────
def read_meminfo() -> dict[str, int]:
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                info[parts[0].rstrip(":")] = int(parts[1])  # value in kB
    return info


def ram_stats() -> tuple[float, float, float]:
    """Return (used_gb, total_gb, pct)."""
    m          = read_meminfo()
    total_kb   = m.get("MemTotal", 0)
    avail_kb   = m.get("MemAvailable", m.get("MemFree", 0))
    used_kb    = total_kb - avail_kb
    total_gb   = round(total_kb  / 1024**2, 2)
    used_gb    = round(used_kb   / 1024**2, 2)
    pct        = round(used_kb / total_kb * 100, 1) if total_kb else 0.0
    return used_gb, total_gb, pct


# ── DISK (/proc/diskstats) ────────────────────────────────────────────────────
_prev_disk = None
_prev_disk_time = None

def disk_io_rates() -> tuple[float, float]:
    """Return (read_mbps, write_mbps) since last call."""
    global _prev_disk, _prev_disk_time

    stats = {}
    with open("/proc/diskstats") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 14:
                dev = parts[2]
                # Only physical disks (sda, vda, nvme0n1, etc.), skip partitions
                if dev[-1].isdigit() and not dev.startswith("nvme"):
                    continue
                if dev.startswith("loop") or dev.startswith("dm-"):
                    continue
                read_sectors  = int(parts[5])
                write_sectors = int(parts[9])
                stats[dev] = (read_sectors, write_sectors)

    now = time.time()

    if _prev_disk is None or not stats:
        _prev_disk      = stats
        _prev_disk_time = now
        return 0.0, 0.0

    dt = now - _prev_disk_time
    if dt <= 0:
        return 0.0, 0.0

    total_read  = sum(stats[d][0] for d in stats)
    total_write = sum(stats[d][1] for d in stats)
    prev_read   = sum(_prev_disk.get(d, (0, 0))[0] for d in stats)
    prev_write  = sum(_prev_disk.get(d, (0, 0))[1] for d in stats)

    sector_bytes = 512
    read_mbps  = round((total_read  - prev_read)  * sector_bytes / dt / 1024**2, 2)
    write_mbps = round((total_write - prev_write) * sector_bytes / dt / 1024**2, 2)

    _prev_disk      = stats
    _prev_disk_time = now
    return read_mbps, write_mbps


# ── GPU (nvidia-smi) ──────────────────────────────────────────────────────────
GPU_FIELDS = [
    "index", "utilization.gpu", "utilization.memory",
    "memory.used", "memory.total", "temperature.gpu", "power.draw",
]

def nvidia_available() -> bool:
    try:
        subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def query_gpus() -> list[dict]:
    try:
        result = subprocess.run(
            ["nvidia-smi",
             f"--query-gpu={','.join(GPU_FIELDS)}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == len(GPU_FIELDS):
                gpus.append(dict(zip(GPU_FIELDS, parts)))
        return gpus
    except Exception:
        return []


# ── SAMPLE ────────────────────────────────────────────────────────────────────
def sample(has_gpu: bool) -> dict:
    ts                   = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    cpu_overall, per_core = cpu_percent()
    ram_used, ram_total, ram_pct = ram_stats()
    disk_read, disk_write = disk_io_rates()

    row = {
        "timestamp":       ts,
        "cpu_pct":         cpu_overall,
        "cpu_cores":       ",".join(str(c) for c in per_core),
        "ram_used_gb":     ram_used,
        "ram_total_gb":    ram_total,
        "ram_pct":         ram_pct,
        "disk_read_mbps":  disk_read,
        "disk_write_mbps": disk_write,
    }

    if has_gpu:
        gpus = query_gpus()
        for i in range(4):
            pfx = f"gpu{i}_"
            if i < len(gpus):
                g = gpus[i]
                row[pfx + "util_pct"]    = g.get("utilization.gpu",    "")
                row[pfx + "mem_pct"]     = g.get("utilization.memory",  "")
                row[pfx + "mem_used_mb"] = g.get("memory.used",         "")
                row[pfx + "mem_total_mb"]= g.get("memory.total",        "")
                row[pfx + "temp_c"]      = g.get("temperature.gpu",     "")
                row[pfx + "power_w"]     = g.get("power.draw",          "")
            else:
                for sfx in ["util_pct","mem_pct","mem_used_mb","mem_total_mb","temp_c","power_w"]:
                    row[pfx + sfx] = ""

    return row


# ── SUMMARY ───────────────────────────────────────────────────────────────────
def print_summary(rows: list[dict], has_gpu: bool, elapsed: float):
    if not rows:
        return

    def floats(key):
        return [float(r[key]) for r in rows if r.get(key) not in ("", None, "")]

    def stats(vals, unit=""):
        if not vals:
            return "n/a"
        return f"avg {sum(vals)/len(vals):.1f}{unit}  peak {max(vals):.1f}{unit}"

    print("\n" + "=" * 60)
    print("  VM Monitor — Run Summary")
    print("=" * 60)
    print(f"  Duration   : {elapsed:.1f}s    Samples : {len(rows)}")
    print(f"  CPU        : {stats(floats('cpu_pct'), '%')}")
    print(f"  RAM used   : {stats(floats('ram_used_gb'), ' GB')}")
    print(f"  RAM %      : {stats(floats('ram_pct'), '%')}")
    print(f"  Disk read  : {stats(floats('disk_read_mbps'), ' MB/s')}")
    print(f"  Disk write : {stats(floats('disk_write_mbps'), ' MB/s')}")

    if has_gpu:
        for i in range(4):
            util = floats(f"gpu{i}_util_pct")
            if not util:
                continue
            mem  = floats(f"gpu{i}_mem_used_mb")
            temp = floats(f"gpu{i}_temp_c")
            pwr  = floats(f"gpu{i}_power_w")
            print(f"  GPU {i} util  : {stats(util, '%')}")
            print(f"  GPU {i} VRAM  : {stats([m/1024 for m in mem], ' GB')}")
            print(f"  GPU {i} temp  : {stats(temp, '°C')}")
            print(f"  GPU {i} power : {stats(pwr, ' W')}")
    print("=" * 60)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--output",   default=None,
                        help="Save to CSV file (omit to print to terminal only)")
    args = parser.parse_args()

    if not sys.platform.startswith("linux"):
        sys.exit("This script reads /proc — Linux only.")

    has_gpu = nvidia_available()

    # Print startup info
    _, ram_total, _ = ram_stats()
    cpu_count = len(read_cpu_stat()) - 1
    print("=" * 60)
    print("  VM Resource Monitor  (no external dependencies)")
    print("=" * 60)
    print(f"  Interval  : {args.interval}s")
    print(f"  Output    : {args.output if args.output else 'terminal only (use --output file.csv to save)'}")
    print(f"  CPU cores : {cpu_count}")
    print(f"  RAM total : {ram_total} GB")
    print(f"  GPU       : {'detected (nvidia-smi)' if has_gpu else 'not detected'}")
    print("=" * 60)
    print("Logging started — press Ctrl+C to stop.\n")

    # Warm-up read (first cpu_percent call always returns 0)
    cpu_percent()
    time.sleep(0.2)

    rows    = []
    start   = time.time()
    writer  = None
    stopped = False

    def handle_stop(sig, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, handle_stop)

    def run_loop(csvfile=None):
        nonlocal writer
        while not stopped:
            loop_start = time.time()
            row = sample(has_gpu)
            rows.append(row)

            if csvfile is not None:
                if writer is None:
                    writer = csv.DictWriter(csvfile, fieldnames=list(row.keys()))
                    writer.writeheader()
                writer.writerow(row)
                csvfile.flush()

            gpu_str = ""
            if has_gpu and row.get("gpu0_util_pct") not in ("", None):
                vram_gb = float(row["gpu0_mem_used_mb"]) / 1024
                gpu_str = f"  GPU {row['gpu0_util_pct']}%  VRAM {vram_gb:.1f}GB"

            print(
                f"[{row['timestamp']}]  "
                f"CPU {row['cpu_pct']:5.1f}%  "
                f"RAM {row['ram_used_gb']:.1f}/{row['ram_total_gb']:.0f}GB "
                f"({row['ram_pct']:.0f}%)"
                f"{gpu_str}",
                flush=True,
            )

            sleep_for = max(0.0, args.interval - (time.time() - loop_start))
            time.sleep(sleep_for)

    try:
        if args.output:
            with open(args.output, "w", newline="") as csvfile:
                run_loop(csvfile)
        else:
            run_loop()
    except KeyboardInterrupt:
        pass

    elapsed = time.time() - start
    saved_msg = f"  Saved to  : {args.output}" if args.output else "  (no file saved)"
    print(f"\nStopped. {len(rows)} samples collected.\n{saved_msg}")
    print_summary(rows, has_gpu, elapsed)


if __name__ == "__main__":
    main()
