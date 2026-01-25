#!/usr/bin/env python3
"""
Golden Ratio (φ) Run Finder - Large Scale Edition

Finds consecutive repeating digit runs in the decimal expansion of φ.
Supports multiple computation methods:
  1. mpmath - Pure Python (slower, but works everywhere)
  2. y-cruncher output files - Scan pre-computed digits (fastest)
  3. Download pre-computed digits from the web

Usage:
    source phi_env/bin/activate

    # Compute with mpmath (slower)
    python goldenratiofinder.py --max-digits 1B

    # Scan a y-cruncher output file (fastest)
    python goldenratiofinder.py --file phi.txt --max-run 100

    # Download and scan (if available online)
    python goldenratiofinder.py --download --max-digits 1T
"""

import sys
import time
import argparse
import json
import os
import mmap
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import threading
import multiprocessing
import http.client
import urllib.parse
import subprocess

# ─────────────────────────────────────────────────────────────────────────────
# 1) IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

try:
    from mpmath import mp as mpmath_mp
    MPMATH_AVAILABLE = True
except ImportError:
    MPMATH_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# 2) CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MAX_RUN = 100
CHUNK_SIZE = 100_000_000  # 100 million digits per chunk for mpmath
FILE_CHUNK_SIZE = 500_000_000  # 500MB chunks for file scanning
PROGRESS_INTERVAL = 100_000_000  # Progress update every 100M digits
LIVE_UPDATE_INTERVAL = 5  # Update live data every 5 seconds
LIVE_DATA_FILE = "live_status.json"  # Local file for website to read
WEBSITE_DATA_DIR = "/Users/david/goldenrationumbers"  # Directory for website files
AUTO_PUBLISH = True  # Automatically push to GitHub when pattern found

# Known ground truth (verified)
KNOWN_RUNS = {
    2: ("33", 7),
    3: ("222", 131),
    4: ("4444", 1218),
    5: ("99999", 6401),
    6: ("555555", 99790),
    7: ("5555555", 771952),
    8: ("55555555", 771952),
    9: ("333333333", 314529196),
}


# ─────────────────────────────────────────────────────────────────────────────
# 3) UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def git_publish(message: str = "Auto-update"):
    """Push live_status.json to GitHub."""
    if not AUTO_PUBLISH:
        return
    try:
        os.chdir(WEBSITE_DATA_DIR)
        subprocess.run(["git", "add", "live_status.json"], capture_output=True)
        subprocess.run(["git", "commit", "-m", message], capture_output=True)
        result = subprocess.run(["git", "push"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"\n  [Published to GitHub]")
        else:
            print(f"\n  [Git push failed: {result.stderr[:50]}]")
    except Exception as e:
        print(f"\n  [Git error: {e}]")


def format_time(seconds: float) -> str:
    """Format seconds into human readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    elif seconds < 86400:
        return f"{seconds/3600:.1f}h"
    else:
        return f"{seconds/86400:.1f}d"


def format_number(n: int) -> str:
    """Format large numbers with units."""
    if n >= 1_000_000_000_000:
        return f"{n/1_000_000_000_000:.2f}T"
    elif n >= 1_000_000_000:
        return f"{n/1_000_000_000:.2f}B"
    elif n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n/1_000:.2f}K"
    return str(n)


def parse_number(s: str) -> int:
    """Parse number with optional K/M/B/T suffix."""
    s = s.upper().replace(",", "").replace("_", "")
    multipliers = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000, "T": 1_000_000_000_000}
    if s[-1] in multipliers:
        return int(float(s[:-1]) * multipliers[s[-1]])
    return int(s)


# ─────────────────────────────────────────────────────────────────────────────
# LIVE STATUS UPDATER
# ─────────────────────────────────────────────────────────────────────────────

class LiveStatusUpdater:
    """Updates live status file ONLY when a pattern is found."""

    def __init__(self, max_digits: int, max_run: int):
        self.max_digits = max_digits
        self.max_run = max_run
        self.start_time = time.time()
        self.results = []  # List of pattern results with timing
        self.current_position = 0
        self.rate = 0
        self.running = False
        self.last_pattern_time = None

    def _format_elapsed(self, seconds: float) -> str:
        """Format elapsed time with milliseconds for short durations."""
        if seconds < 1:
            return f"{seconds*1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.3f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = seconds % 60
            return f"{mins}m {secs:.1f}s"
        elif seconds < 86400:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"
        else:
            days = seconds / 86400
            return f"{days:.2f} days"

    def _format_timestamp(self, ts: float) -> str:
        """Format timestamp with milliseconds."""
        dt = datetime.fromtimestamp(ts)
        return dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{int((ts % 1) * 1000):03d}"

    def start(self):
        """Mark search as started."""
        self.running = True
        self.start_time = time.time()
        self.last_pattern_time = self.start_time
        self._write_status("Search started")
        print(f"Live status updates enabled: {WEBSITE_DATA_DIR}/{LIVE_DATA_FILE}")

    def update(self, position: int, results: dict, rate: float):
        """Update current position (no file write, just tracking)."""
        self.current_position = position
        self.rate = rate

    def pattern_found(self, run_length: int, sequence: str, position: int):
        """Called when a new pattern is found - writes to JSON and pushes to GitHub."""
        now = time.time()
        elapsed_total = now - self.start_time
        elapsed_since_last = now - self.last_pattern_time if self.last_pattern_time else elapsed_total

        pattern_data = {
            "run_length": run_length,
            "sequence": sequence,
            "position": position,
            "position_formatted": f"{position:,}",
            "found_at_timestamp": self._format_timestamp(now),
            "found_at_unix": now,
            "elapsed_total_seconds": elapsed_total,
            "elapsed_total_formatted": self._format_elapsed(elapsed_total),
            "elapsed_since_last_seconds": elapsed_since_last,
            "elapsed_since_last_formatted": self._format_elapsed(elapsed_since_last)
        }

        self.results.append(pattern_data)
        self.last_pattern_time = now

        self._write_status(f"Found run {run_length}: {sequence}")

        # Auto-publish to GitHub
        git_publish(f"Pattern found: {run_length}-digit run '{sequence}' at position {position:,}")

    def _write_status(self, event: str = ""):
        """Write current status to JSON file."""
        now = time.time()
        elapsed = now - self.start_time

        # Known pattern positions for estimation
        known_positions = {
            2: 7, 3: 131, 4: 1218, 5: 6401, 6: 99790,
            7: 771952, 8: 771952, 9: 314529196, 10: None
        }

        # Estimate time to next pattern
        next_pattern_estimate = None
        if self.rate > 0:
            next_run = len(self.results) + 2  # +2 because we start at run length 2
            if next_run in known_positions and known_positions[next_run]:
                remaining = known_positions[next_run] - self.current_position
                if remaining > 0:
                    eta_seconds = remaining / self.rate
                    next_pattern_estimate = {
                        "next_run_length": next_run,
                        "expected_position": known_positions[next_run],
                        "digits_remaining": remaining,
                        "eta_seconds": eta_seconds,
                        "eta_formatted": self._format_elapsed(eta_seconds)
                    }

        # ETA to complete all target digits
        eta_completion = None
        if self.rate > 0 and self.current_position < self.max_digits:
            remaining_digits = self.max_digits - self.current_position
            eta_seconds = remaining_digits / self.rate
            eta_completion = {
                "digits_remaining": remaining_digits,
                "eta_seconds": eta_seconds,
                "eta_formatted": self._format_elapsed(eta_seconds)
            }

        status = {
            "timestamp": self._format_timestamp(now),
            "timestamp_unix": now,
            "event": event,
            "running": self.running,
            "target_digits": self.max_digits,
            "target_digits_formatted": format_number(self.max_digits),
            "current_position": self.current_position,
            "current_position_formatted": format_number(self.current_position),
            "percent_complete": round((self.current_position / self.max_digits * 100), 4) if self.max_digits > 0 else 0,
            "elapsed_seconds": elapsed,
            "elapsed_formatted": self._format_elapsed(elapsed),
            "rate": self.rate,
            "rate_formatted": f"{format_number(int(self.rate))}/s" if self.rate > 0 else "0/s",
            "patterns_found": len(self.results),
            "max_run_searching": self.max_run,
            "longest_run_found": self.results[-1]["run_length"] if self.results else 0,
            "next_pattern_estimate": next_pattern_estimate,
            "eta_completion": eta_completion,
            "results": self.results,
            "start_time": self._format_timestamp(self.start_time),
            "start_time_unix": self.start_time
        }

        filepath = os.path.join(WEBSITE_DATA_DIR, LIVE_DATA_FILE)
        try:
            os.makedirs(WEBSITE_DATA_DIR, exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(status, f, indent=2)
        except Exception as e:
            print(f"\nWarning: Could not write live status: {e}")

    def stop(self):
        """Mark search as stopped and write final status."""
        self.running = False
        self._write_status("Search completed")


# ─────────────────────────────────────────────────────────────────────────────
# 4) RUN FINDER CLASS
# ─────────────────────────────────────────────────────────────────────────────

class RunFinder:
    """Find first occurrence of each run length."""

    def __init__(self, max_run: int = DEFAULT_MAX_RUN):
        self.max_run = max_run
        self.results = {}
        self.remaining = set(range(2, max_run + 1))
        self.run_char = None
        self.run_len = 0
        self.total_pos = 0
        self.start_time = time.time()
        self.longest_found = 1

    def process_chunk(self, chunk: str, verbose: bool = True) -> list:
        """Process a chunk of digits, returns newly found runs."""
        new_finds = []

        for ch in chunk:
            self.total_pos += 1

            if not ch.isdigit():
                self.run_char = None
                self.run_len = 0
                continue

            if ch == self.run_char:
                self.run_len += 1
            else:
                self.run_char = ch
                self.run_len = 1

            if self.run_len in self.remaining:
                n = self.run_len
                start_pos = self.total_pos - n + 1
                seq = self.run_char * n

                self.results[n] = (seq, start_pos)
                self.remaining.remove(n)
                self.longest_found = max(self.longest_found, n)
                new_finds.append((n, seq, start_pos))

                if verbose:
                    elapsed = time.time() - self.start_time
                    print(f"\n{'='*70}")
                    print(f"FOUND RUN OF {n}!")
                    print(f"  Sequence: {seq[:50]}{'...' if len(seq) > 50 else ''}")
                    print(f"  Position: {start_pos:,}")
                    print(f"  Time: {format_time(elapsed)}")
                    print(f"  Remaining: {len(self.remaining)} run lengths to find")
                    print(f"{'='*70}\n")
                    # Auto-publish to GitHub
                    git_publish(f"Found run of {n}: {seq[:20]} @ {start_pos:,}")

            if self.run_len > self.longest_found:
                self.longest_found = self.run_len

        return new_finds

    def save_state(self, filepath: str):
        """Save current state to JSON file."""
        state = {
            "timestamp": datetime.now().isoformat(),
            "position": self.total_pos,
            "elapsed": time.time() - self.start_time,
            "max_run": self.max_run,
            "longest_found": self.longest_found,
            "results": {str(k): {"seq": v[0], "pos": v[1]} for k, v in self.results.items()},
            "remaining": sorted(list(self.remaining)),
            "run_state": {"char": self.run_char, "len": self.run_len}
        }
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self, filepath: str) -> bool:
        """Load state from JSON file."""
        if not os.path.exists(filepath):
            return False
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            self.total_pos = state["position"]
            self.max_run = state["max_run"]
            self.longest_found = state["longest_found"]
            self.results = {int(k): (v["seq"], v["pos"]) for k, v in state["results"].items()}
            self.remaining = set(state["remaining"])
            if "run_state" in state:
                self.run_char = state["run_state"]["char"]
                self.run_len = state["run_state"]["len"]
            print(f"Resumed from position {self.total_pos:,}")
            return True
        except Exception as e:
            print(f"Failed to load state: {e}")
            return False

    def print_results(self):
        """Print final results."""
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"Searched: {format_number(self.total_pos)} digits")
        print(f"Time: {format_time(time.time() - self.start_time)}")
        print(f"Longest run found: {self.longest_found}")
        print()
        print("First occurrence of each run length:")
        print("-" * 50)

        for n in range(2, self.max_run + 1):
            if n in self.results:
                seq, pos = self.results[n]
                display_seq = seq[:20] + "..." if len(seq) > 20 else seq
                print(f"  Run of {n:3d}: {display_seq} @ position {pos:,}")
            elif n <= self.longest_found:
                print(f"  Run of {n:3d}: NOT FOUND")


# ─────────────────────────────────────────────────────────────────────────────
# 5) FILE SCANNER - For y-cruncher output or downloaded digits
# ─────────────────────────────────────────────────────────────────────────────

def scan_digit_file(filepath: str, max_run: int = DEFAULT_MAX_RUN,
                    save_file: str = None, resume: bool = False,
                    max_digits: int = None) -> dict:
    """
    Scan a file containing φ digits for runs.
    Handles large files efficiently using memory mapping.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    file_size = filepath.stat().st_size
    print("=" * 70)
    print("Golden Ratio (φ) Run Finder - File Scanner")
    print("=" * 70)
    print(f"File: {filepath}")
    print(f"Size: {format_number(file_size)} bytes")
    print(f"Looking for: runs of 2 to {max_run} identical digits")
    print("=" * 70)
    print()

    finder = RunFinder(max_run)

    if resume and save_file:
        finder.load_state(save_file)

    start_pos = finder.total_pos
    last_progress = start_pos

    try:
        with open(filepath, 'rb') as f:
            # Memory map the file for efficient access
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                pos = start_pos
                limit = min(len(mm), max_digits) if max_digits else len(mm)

                while pos < limit and finder.remaining:
                    # Read chunk
                    chunk_end = min(pos + FILE_CHUNK_SIZE, limit)
                    chunk = mm[pos:chunk_end].decode('ascii', errors='ignore')

                    finder.process_chunk(chunk)
                    pos = chunk_end

                    # Progress update
                    if pos - last_progress >= PROGRESS_INTERVAL:
                        elapsed = time.time() - finder.start_time
                        rate = finder.total_pos / elapsed if elapsed > 0 else 0
                        pct = 100 * pos / limit

                        sys.stdout.write(
                            f"\rProgress: {format_number(pos)} / {format_number(limit)} "
                            f"({pct:.2f}%) | "
                            f"Rate: {format_number(int(rate))}/s | "
                            f"Found: {len(finder.results)}/{max_run-1} | "
                            f"Longest: {finder.longest_found}    "
                        )
                        sys.stdout.flush()
                        last_progress = pos

                        if save_file:
                            finder.save_state(save_file)

    except KeyboardInterrupt:
        print("\n\nInterrupted!")
        if save_file:
            finder.save_state(save_file)
            print(f"Progress saved to {save_file}")

    finder.print_results()

    if save_file:
        finder.save_state(save_file)

    return finder.results


# ─────────────────────────────────────────────────────────────────────────────
# 6) MPMATH COMPUTATION (Original method)
# ─────────────────────────────────────────────────────────────────────────────

def compute_with_mpmath(max_digits: int, max_run: int = DEFAULT_MAX_RUN,
                        save_file: str = None, resume: bool = False,
                        live_updates: bool = True) -> dict:
    """Compute φ digits using mpmath and scan for runs."""

    if not MPMATH_AVAILABLE:
        print("Error: mpmath not installed. Run: pip install mpmath")
        sys.exit(1)

    print("=" * 70)
    print("Golden Ratio (φ) Run Finder - mpmath Computation")
    print("=" * 70)
    print(f"Target: {format_number(max_digits)} digits")
    print(f"Looking for: runs of 2 to {max_run} identical digits")
    print(f"Backend: mpmath")
    print("=" * 70)
    print()

    finder = RunFinder(max_run)

    if resume and save_file:
        finder.load_state(save_file)

    # Initialize live status updater
    live_updater = None
    if live_updates:
        live_updater = LiveStatusUpdater(max_digits, max_run)
        live_updater.start()

    start_pos = finder.total_pos
    last_progress = start_pos
    last_save = start_pos

    try:
        pos = start_pos

        while pos < max_digits and finder.remaining:
            chunk_size = min(CHUNK_SIZE, max_digits - pos)

            sys.stdout.write(f"\rComputing digits {format_number(pos)} to {format_number(pos + chunk_size)}...")
            sys.stdout.flush()

            # Compute chunk
            needed = pos + chunk_size + 100
            mpmath_mp.dps = needed
            phi = (1 + mpmath_mp.sqrt(5)) / 2
            s = mpmath_mp.nstr(phi, needed, strip_zeros=False)

            chunk = s[pos:pos + chunk_size] if pos < len(s) else ""
            if not chunk:
                print(f"\nReached computation limit at position {pos:,}")
                break

            new_patterns = finder.process_chunk(chunk)
            pos += len(chunk)

            # Calculate rate
            elapsed = time.time() - finder.start_time
            rate = finder.total_pos / elapsed if elapsed > 0 else 0

            # Update live status position tracking
            if live_updater:
                live_updater.update(finder.total_pos, finder.results, rate)
                # Notify about new patterns found
                for run_len, seq, pattern_pos in new_patterns:
                    live_updater.pattern_found(run_len, seq, pattern_pos)

            # Progress update
            if pos - last_progress >= PROGRESS_INTERVAL // 10:
                pct = 100 * pos / max_digits

                sys.stdout.write(
                    f"\rProgress: {format_number(pos)} / {format_number(max_digits)} "
                    f"({pct:.2f}%) | "
                    f"Rate: {format_number(int(rate))}/s | "
                    f"Found: {len(finder.results)}/{max_run-1} | "
                    f"Longest: {finder.longest_found}    "
                )
                sys.stdout.flush()
                last_progress = pos

            # Periodic save
            if save_file and pos - last_save >= 1_000_000_000:
                finder.save_state(save_file)
                last_save = pos

    except KeyboardInterrupt:
        print("\n\nInterrupted!")
        if save_file:
            finder.save_state(save_file)
            print(f"Progress saved to {save_file}")

    # Stop live updates
    if live_updater:
        live_updater.stop()

    finder.print_results()

    if save_file:
        finder.save_state(save_file)

    return finder.results


# ─────────────────────────────────────────────────────────────────────────────
# 7) Y-CRUNCHER SETUP HELPER
# ─────────────────────────────────────────────────────────────────────────────

def print_ycruncher_instructions():
    """Print instructions for using y-cruncher."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     Y-CRUNCHER SETUP INSTRUCTIONS                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  y-cruncher is 40,000x+ faster than mpmath for computing φ digits.           ║
║  However, it only runs on x86/x64 Linux or Windows.                          ║
║                                                                              ║
║  OPTION 1: Use a Cloud VM (Recommended for large computations)               ║
║  ─────────────────────────────────────────────────────────────────────────   ║
║  1. Spin up an x86 Linux VM (AWS, GCP, Azure, etc.)                          ║
║  2. Download y-cruncher:                                                     ║
║     wget https://github.com/Mysticial/y-cruncher/releases/latest/...         ║
║  3. Run: ./y-cruncher                                                        ║
║  4. Select: 1 (Compute Custom Constant)                                      ║
║  5. Select: 4 (Golden Ratio)                                                 ║
║  6. Enter digits (e.g., 25000000000000 for 25T)                              ║
║  7. Download the output .txt file                                            ║
║  8. Run: python goldenratiofinder.py --file phi.txt                          ║
║                                                                              ║
║  OPTION 2: Docker with x86 emulation (Slower but local)                      ║
║  ─────────────────────────────────────────────────────────────────────────   ║
║  1. Install Docker Desktop                                                   ║
║  2. Enable Rosetta for x86 emulation                                         ║
║  3. docker run --platform linux/amd64 -it ubuntu bash                        ║
║  4. Inside container: apt update && apt install wget                         ║
║  5. Download and run y-cruncher                                              ║
║                                                                              ║
║  STORAGE REQUIREMENTS:                                                       ║
║  ─────────────────────────────────────────────────────────────────────────   ║
║  • 1 Trillion digits:   ~1 TB storage                                        ║
║  • 10 Trillion digits:  ~10 TB storage                                       ║
║  • 25 Trillion digits:  ~25 TB storage                                       ║
║                                                                              ║
║  TIME ESTIMATES (on fast x86 hardware):                                      ║
║  ─────────────────────────────────────────────────────────────────────────   ║
║  • 1 Trillion digits:   ~1-2 hours                                           ║
║  • 10 Trillion digits:  ~12-24 hours                                         ║
║  • 25 Trillion digits:  ~2-4 days                                            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


# ─────────────────────────────────────────────────────────────────────────────
# 8) VERIFICATION TEST
# ─────────────────────────────────────────────────────────────────────────────

def run_test():
    """Quick verification test."""
    if not MPMATH_AVAILABLE:
        print("Error: mpmath required for test. Run: pip install mpmath")
        sys.exit(1)

    print("Running verification test...")

    mpmath_mp.dps = 100_100
    phi = (1 + mpmath_mp.sqrt(5)) / 2
    digits = mpmath_mp.nstr(phi, 100_100, strip_zeros=False)[:100_000]

    finder = RunFinder(max_run=10)
    finder.process_chunk(digits, verbose=False)

    print("\nVerifying against known values:")
    all_ok = True

    for n, (expected_seq, expected_pos) in KNOWN_RUNS.items():
        if n > 10:
            continue
        if n in finder.results:
            seq, pos = finder.results[n]
            if seq == expected_seq and pos == expected_pos:
                print(f"  Run of {n}: ✓ {seq} @ {pos}")
            else:
                print(f"  Run of {n}: ✗ Expected {expected_seq}@{expected_pos}, got {seq}@{pos}")
                all_ok = False
        else:
            if expected_pos and expected_pos <= 100_000:
                print(f"  Run of {n}: ✗ Not found (expected {expected_seq}@{expected_pos})")
                all_ok = False
            else:
                print(f"  Run of {n}: - Not expected in first 100K digits")

    if all_ok:
        print("\n✓ All verifications passed!")
    else:
        print("\n✗ Some verifications failed!")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# 9) MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Find repeating digit runs in the Golden Ratio (φ)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python goldenratiofinder.py --test                    # Verify algorithm
  python goldenratiofinder.py --max-digits 1B           # Compute 1B digits with mpmath
  python goldenratiofinder.py --file phi.txt            # Scan pre-computed file
  python goldenratiofinder.py --ycruncher               # Show y-cruncher setup instructions
  python goldenratiofinder.py --resume --save prog.json # Resume from saved state
"""
    )

    parser.add_argument("--file", type=str, help="Path to digit file (from y-cruncher)")
    parser.add_argument("--max-digits", type=str, default="1B",
                        help="Maximum digits to compute/scan (e.g., 1M, 1B, 25T)")
    parser.add_argument("--max-run", type=int, default=DEFAULT_MAX_RUN,
                        help=f"Maximum run length to find (default: {DEFAULT_MAX_RUN})")
    parser.add_argument("--save", type=str, default="phi_runs.json",
                        help="File to save progress (default: phi_runs.json)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from saved progress")
    parser.add_argument("--test", action="store_true",
                        help="Run verification test")
    parser.add_argument("--ycruncher", action="store_true",
                        help="Show y-cruncher setup instructions")

    args = parser.parse_args()

    if args.test:
        run_test()
        return

    if args.ycruncher:
        print_ycruncher_instructions()
        return

    max_digits = parse_number(args.max_digits)

    if args.file:
        # Scan existing file
        scan_digit_file(
            args.file,
            max_run=args.max_run,
            save_file=args.save,
            resume=args.resume,
            max_digits=max_digits
        )
    else:
        # Compute with mpmath
        compute_with_mpmath(
            max_digits=max_digits,
            max_run=args.max_run,
            save_file=args.save,
            resume=args.resume
        )


if __name__ == "__main__":
    main()
