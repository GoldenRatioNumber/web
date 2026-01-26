#!/usr/bin/env python3
"""
Golden Ratio (φ) Diagonal Pattern Finder

Finds specific "diagonal" patterns where digit N repeats N times:
  1, 22, 333, 4444, 55555, 666666, 7777777, 88888888, 999999999

Usage:
    python diagonal_finder.py --max-digits 500M
"""

import sys
import time
import argparse
import json
import os
from datetime import datetime
import subprocess

try:
    from mpmath import mp as mpmath_mp
    MPMATH_AVAILABLE = True
except ImportError:
    MPMATH_AVAILABLE = False
    print("Error: mpmath not installed. Run: pip install mpmath")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

CHUNK_SIZE = 100_000_000  # 100 million digits per chunk
PROGRESS_INTERVAL = 10_000_000  # Progress update every 10M digits
WEBSITE_DATA_DIR = "/Users/david/goldenrationumbers"
RUNS_INDEX_FILE = "diagonal_runs_index.json"
AUTO_PUBLISH = True

# The diagonal patterns to find: digit N repeated N times
DIAGONAL_PATTERNS = {
    1: "1",
    2: "22",
    3: "333",
    4: "4444",
    5: "55555",
    6: "666666",
    7: "7777777",
    8: "88888888",
    9: "999999999"
}

# ─────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def git_publish(files: list, message: str = "Auto-update"):
    """Push specified files to GitHub."""
    if not AUTO_PUBLISH:
        return
    try:
        os.chdir(WEBSITE_DATA_DIR)
        for f in files:
            subprocess.run(["git", "add", f], capture_output=True)
        subprocess.run(["git", "commit", "-m", message], capture_output=True)
        result = subprocess.run(["git", "push"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"\n  [Published to GitHub]")
        else:
            print(f"\n  [Git push failed: {result.stderr[:50]}]")
    except Exception as e:
        print(f"\n  [Git error: {e}]")


def update_runs_index(run_filename: str, is_latest: bool = True):
    """Update the diagonal_runs_index.json file."""
    index_path = os.path.join(WEBSITE_DATA_DIR, RUNS_INDEX_FILE)

    if os.path.exists(index_path):
        with open(index_path, 'r') as f:
            index = json.load(f)
    else:
        index = {"runs": [], "latest": None}

    if run_filename not in index["runs"]:
        index["runs"].insert(0, run_filename)

    if is_latest:
        index["latest"] = run_filename

    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)

    return index_path


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

class DiagonalStatusUpdater:
    """Updates live status file for diagonal pattern search."""

    def __init__(self, max_digits: int):
        self.max_digits = max_digits
        self.start_time = time.time()
        self.results = []
        self.current_position = 0
        self.rate = 0
        self.running = False
        self.last_pattern_time = None
        self.patterns_to_find = set(DIAGONAL_PATTERNS.keys())
        self.run_filename = f"diagonal_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        self.run_filepath = os.path.join(WEBSITE_DATA_DIR, self.run_filename)

    def _format_elapsed(self, seconds: float) -> str:
        """Format elapsed time with milliseconds."""
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
        self.results = []

        self._write_status("Search started")
        update_runs_index(self.run_filename, is_latest=True)
        git_publish([self.run_filename, RUNS_INDEX_FILE], "Diagonal pattern search started")
        print(f"Live status: {self.run_filepath}")

    def update(self, position: int, rate: float):
        """Update current position."""
        self.current_position = position
        self.rate = rate

    def pattern_found(self, digit: int, pattern: str, position: int):
        """Called when a diagonal pattern is found."""
        now = time.time()
        elapsed_total = now - self.start_time
        elapsed_since_last = now - self.last_pattern_time if self.last_pattern_time else elapsed_total

        pattern_data = {
            "digit": digit,
            "pattern": pattern,
            "length": len(pattern),
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
        self.patterns_to_find.discard(digit)

        self._write_status(f"Found {pattern} at position {position:,}")
        git_publish([self.run_filename, RUNS_INDEX_FILE],
                    f"Diagonal: Found {pattern} at decimal place {position:,}")

    def _write_status(self, event: str = ""):
        """Write current status to JSON file."""
        now = time.time()
        elapsed = now - self.start_time

        # ETA calculation
        eta_completion = None
        if self.rate > 0 and self.current_position < self.max_digits:
            remaining = self.max_digits - self.current_position
            eta_seconds = remaining / self.rate
            eta_completion = {
                "digits_remaining": remaining,
                "eta_seconds": eta_seconds,
                "eta_formatted": self._format_elapsed(eta_seconds)
            }

        status = {
            "run_file": self.run_filename,
            "search_type": "diagonal",
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
            "patterns_remaining": len(self.patterns_to_find),
            "eta_completion": eta_completion,
            "results": self.results,
            "start_time": self._format_timestamp(self.start_time),
            "start_time_unix": self.start_time
        }

        try:
            os.makedirs(WEBSITE_DATA_DIR, exist_ok=True)
            with open(self.run_filepath, 'w') as f:
                json.dump(status, f, indent=2)
        except Exception as e:
            print(f"\nWarning: Could not write status: {e}")

    def stop(self, event: str = "Search completed"):
        """Mark search as stopped."""
        self.running = False
        self._write_status(event)
        update_runs_index(self.run_filename, is_latest=True)
        git_publish([self.run_filename, RUNS_INDEX_FILE], f"Diagonal search ended: {event}")
        print(f"\n  [Final status pushed to GitHub]")


# ─────────────────────────────────────────────────────────────────────────────
# DIAGONAL PATTERN FINDER
# ─────────────────────────────────────────────────────────────────────────────

class DiagonalFinder:
    """Find diagonal patterns: 1, 22, 333, 4444, etc."""

    def __init__(self):
        self.results = {}  # digit -> position
        self.remaining = set(DIAGONAL_PATTERNS.keys())
        self.total_pos = 0
        self.start_time = time.time()

        # Track current run for each digit
        self.run_counts = {d: 0 for d in range(10)}
        self.run_starts = {d: 0 for d in range(10)}

    def process_chunk(self, chunk: str, status_updater=None, verbose: bool = True) -> list:
        """Process a chunk of digits."""
        new_finds = []

        for ch in chunk:
            self.total_pos += 1

            if not ch.isdigit():
                # Reset all run counts
                for d in range(10):
                    self.run_counts[d] = 0
                continue

            digit = int(ch)

            # For the current digit, increment its run count
            if self.run_counts[digit] == 0:
                self.run_starts[digit] = self.total_pos

            self.run_counts[digit] += 1

            # Reset other digits' run counts
            for d in range(10):
                if d != digit:
                    self.run_counts[d] = 0

            # Check if we found a diagonal pattern
            # Pattern for digit N is N repeated N times
            if digit in self.remaining:
                target_length = digit  # e.g., digit 3 needs length 3
                if self.run_counts[digit] == target_length:
                    # Found it! Position is where the pattern starts (subtract 2 for "1.")
                    start_pos = self.run_starts[digit] - 2
                    pattern = DIAGONAL_PATTERNS[digit]

                    self.results[digit] = start_pos
                    self.remaining.remove(digit)
                    new_finds.append((digit, pattern, start_pos))

                    if verbose:
                        elapsed = time.time() - self.start_time
                        print(f"\n{'='*70}")
                        print(f"FOUND DIAGONAL PATTERN: {pattern}")
                        print(f"  Digit {digit} repeated {digit} times")
                        print(f"  Decimal Place: {start_pos:,}")
                        print(f"  Time: {format_time(elapsed)}")
                        print(f"  Remaining: {len(self.remaining)} patterns to find")
                        print(f"{'='*70}\n")

                    if status_updater:
                        status_updater.pattern_found(digit, pattern, start_pos)

        return new_finds

    def print_results(self):
        """Print final results."""
        print("\n" + "="*70)
        print("DIAGONAL PATTERN SEARCH RESULTS")
        print("="*70)
        print(f"{'Pattern':<12} {'Decimal Place':<20} {'Status'}")
        print("-"*70)

        for digit in range(1, 10):
            pattern = DIAGONAL_PATTERNS[digit]
            if digit in self.results:
                pos = self.results[digit]
                print(f"{pattern:<12} {pos:>15,}      FOUND")
            else:
                print(f"{pattern:<12} {'--':>15}      NOT FOUND")

        print("="*70)
        print(f"Found {len(self.results)}/9 diagonal patterns")
        print(f"Total digits searched: {self.total_pos:,}")
        print(f"Total time: {format_time(time.time() - self.start_time)}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def find_diagonal_patterns(max_digits: int = 500_000_000):
    """Main function to find diagonal patterns."""

    print("="*70)
    print("Golden Ratio (φ) Diagonal Pattern Finder")
    print("="*70)
    print(f"Target: {format_number(max_digits)} digits")
    print("Looking for: 1, 22, 333, 4444, 55555, 666666, 7777777, 88888888, 999999999")
    print("="*70 + "\n")

    finder = DiagonalFinder()
    status_updater = DiagonalStatusUpdater(max_digits)
    status_updater.start()

    pos = 0
    last_progress = 0

    try:
        while pos < max_digits and finder.remaining:
            chunk_size = min(CHUNK_SIZE, max_digits - pos)

            sys.stdout.write(f"\rComputing digits {format_number(pos)} to {format_number(pos + chunk_size)}...")
            sys.stdout.flush()

            # Compute phi digits
            needed = pos + chunk_size + 100
            mpmath_mp.dps = needed
            phi = (1 + mpmath_mp.sqrt(5)) / 2
            s = mpmath_mp.nstr(phi, needed, strip_zeros=False)

            chunk = s[pos:pos + chunk_size] if pos < len(s) else ""
            if not chunk:
                print(f"\nReached computation limit at position {pos:,}")
                break

            finder.process_chunk(chunk, status_updater)
            pos += len(chunk)

            # Calculate rate
            elapsed = time.time() - finder.start_time
            rate = finder.total_pos / elapsed if elapsed > 0 else 0
            status_updater.update(finder.total_pos, rate)

            # Progress update
            if pos - last_progress >= PROGRESS_INTERVAL:
                pct = 100 * pos / max_digits
                sys.stdout.write(
                    f"\rProgress: {format_number(pos)} / {format_number(max_digits)} "
                    f"({pct:.2f}%) | "
                    f"Rate: {format_number(int(rate))}/s | "
                    f"Found: {len(finder.results)}/9    "
                )
                sys.stdout.flush()
                last_progress = pos

    except KeyboardInterrupt:
        print("\n\nInterrupted by user!")
        status_updater.stop("Search interrupted by user")
        finder.print_results()
        return finder.results

    status_updater.stop("Search completed successfully")
    finder.print_results()
    return finder.results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find diagonal patterns in phi")
    parser.add_argument("--max-digits", type=str, default="500M",
                        help="Maximum digits to search (e.g., 500M, 1B)")
    args = parser.parse_args()

    max_digits = parse_number(args.max_digits)
    find_diagonal_patterns(max_digits)
