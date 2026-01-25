#!/usr/bin/env python3
import curses
import sys
import time
from decimal import Decimal, getcontext

# ─────────────────────────────────────────────────────────────────────────────
# 1) GLOBALS: How many φ‐characters to precompute, and run‐color maps
# ─────────────────────────────────────────────────────────────────────────────

PHI_PREFIX_LENGTH = 100_000  # precompute “1.” + this many characters

# Known “ground truth” for the first 20 000 characters (including “1.”):
GROUND_TRUTH = {
    2: ("33",    7),
    3: ("222",  131),
    4: ("4444",1218),
    5: ("99999",6401),
    6: ("N/A",   None),
    7: ("N/A",   None),
    8: ("N/A",   None),
    9: ("N/A",   None),
}

# digit '0'..'9' → curses color_pair index (for runs ≥ 2)
RUN_COLOR_MAP = {
    '0': 2,  # red
    '1': 2,  # red + bold
    '2': 3,  # green
    '3': 3,  # green + bold
    '4': 4,  # yellow
    '5': 4,  # yellow + bold
    '6': 5,  # blue
    '7': 5,  # blue + bold
    '8': 6,  # magenta
    '9': 6,  # magenta + bold
}

# run‐length n → highlight color_pair index (with A_REVERSE for white background)
HIGHLIGHT_PAIR = {
    2: 2,   # red-on-white
    3: 4,   # yellow-on-white
    4: 3,   # green-on-white
    5: 7,   # cyan-on-white
    6: 8,   # blue-on-white
    7: 9,   # magenta-on-white
    8: 10,  # red-on-white (bright)
    9: 11,  # yellow-on-white (bright)
}


# ─────────────────────────────────────────────────────────────────────────────
# 2) PRECOMPUTE φ PREFIX
# ─────────────────────────────────────────────────────────────────────────────

def get_phi_prefix(N: int) -> str:
    """
    Compute φ = (1 + sqrt(5)) / 2 to at least N characters (including "1.").
    Returns exactly N characters via one high‐precision Decimal call.
    """
    getcontext().prec = N + 50
    phi = (Decimal(1) + Decimal(5).sqrt()) / Decimal(2)
    s = format(phi, 'f')
    while len(s) < N:
        getcontext().prec += 50
        phi = (Decimal(1) + Decimal(5).sqrt()) / Decimal(2)
        s = format(phi, 'f')
    return s[:N]


# ─────────────────────────────────────────────────────────────────────────────
# 3) FIND FIRST RUNS (DEBUG CHECK)
# ─────────────────────────────────────────────────────────────────────────────

def find_first_runs(phi_str: str, max_n: int = 9) -> dict:
    """
    Given phi_str (including '1.' prefix), find the first contiguous run of length n
    for n=2..max_n. Non‐digit breaks a run. Returns {n: (sequence, position)}.
    """
    first_runs = {}
    L = len(phi_str)

    for n in range(2, max_n + 1):
        seq = "N/A"
        pos = None
        run_char = None
        run_len = 0

        for i, ch in enumerate(phi_str):
            if not ch.isdigit():
                run_char = None
                run_len = 0
                continue

            if ch == run_char:
                run_len += 1
            else:
                run_char = ch
                run_len = 1

            if run_len == n:
                seq = run_char * n
                # Start position = (i+1) - (n - 1)
                pos = (i + 1) - (n - 1)
                break

        first_runs[n] = (seq, pos)

    return first_runs


def run_debug_check():
    """
    Build the first 20 000-character prefix, verify that runs for n=2..9 match GROUND_TRUTH.
    """
    N = 20_000
    sys.stdout.write(f"Debug-checking first {N:,} characters of φ…\n")
    sys.stdout.flush()

    prefix = get_phi_prefix(N)
    found = find_first_runs(prefix, 9)

    mismatches = []
    for n in range(2, 10):
        exp_seq, exp_pos = GROUND_TRUTH[n]
        fnd_seq, fnd_pos = found[n]
        if exp_seq != fnd_seq or exp_pos != fnd_pos:
            mismatches.append((n, exp_seq, exp_pos, fnd_seq, fnd_pos))

    if mismatches:
        sys.stdout.write("\nDEBUG CHECK FAILED! Discrepancies:\n\n")
        for (n, esp, epos, fsp, fpos) in mismatches:
            sys.stdout.write(f"  n = {n}:\n")
            sys.stdout.write(f"    Expected: seq = {repr(esp)}, pos = {epos}\n")
            sys.stdout.write(f"    Found:    seq = {repr(fsp)}, pos = {fpos}\n\n")
        sys.stdout.write("Exiting due to logic errors.\n")
        sys.stdout.flush()
        sys.exit(1)

    sys.stdout.write("DEBUG CHECK PASSED: runs for n=2..9 are correct in first 20,000 characters.\n\n")
    sys.stdout.flush()
    time.sleep(1)


# ─────────────────────────────────────────────────────────────────────────────
# 4) REAL‐TIME CURSES UI
# ─────────────────────────────────────────────────────────────────────────────

def print_legend(header_win, width):
    """
    Draw the legend on rows 1..8 of header_win: “ n = <digit×n>” highlighted
    via reverse video + color_pair so it simulates a white‐background “rainbow” highlight.
    """
    legend_title = "Legend: run-length → highlight (white-bg, rainbow text)"
    header_win.addstr(1, 2, legend_title[: width - 4], curses.A_BOLD)

    for n in range(2, 10):
        sample = str(n) * n
        label = f"  {n} = {sample}"
        safe_label = label[: width - 4]

        attr = curses.A_REVERSE
        if n == 2:
            attr |= curses.color_pair(2)
        elif n == 3:
            attr |= curses.color_pair(4)
        elif n == 4:
            attr |= curses.color_pair(3)
        elif n == 5:
            attr |= curses.color_pair(7)
        elif n == 6:
            attr |= curses.color_pair(8)
        elif n == 7:
            attr |= curses.color_pair(9)
        elif n == 8:
            attr |= curses.color_pair(10)
        elif n == 9:
            attr |= curses.color_pair(11)

        header_win.addstr(n, 2, safe_label, attr)


def curses_main(stdscr):
    """
    Main curses loop. Uses a fixed header (rows 0..9) for title, legend,
    and “Len n: …” lines, and a scrolling digit window (rows 10+).
    Streams from a static prefix first, then switches to live generation.
    Pauses 1 second whenever it finds a new first‐run of length n.
    """
    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()

    # 4A) Initialize color pairs
    curses.init_pair(1, curses.COLOR_WHITE,   -1)  # singletons (white)
    curses.init_pair(2, curses.COLOR_RED,     -1)  # red
    curses.init_pair(3, curses.COLOR_GREEN,   -1)  # green
    curses.init_pair(4, curses.COLOR_YELLOW,  -1)  # yellow
    curses.init_pair(5, curses.COLOR_BLUE,    -1)  # blue
    curses.init_pair(6, curses.COLOR_MAGENTA, -1)  # magenta
    curses.init_pair(7, curses.COLOR_CYAN,    -1)  # cyan (for run‐length 5)
    curses.init_pair(8, curses.COLOR_BLUE,    -1)  # blue (for run‐length 6)
    curses.init_pair(9, curses.COLOR_MAGENTA, -1)  # magenta (for run‐length 7)
    curses.init_pair(10, curses.COLOR_RED,    -1)  # red (for run‐length 8)
    curses.init_pair(11, curses.COLOR_YELLOW, -1)  # yellow (for run‐length 9)

    # 4B) Prepare windows
    h, w = stdscr.getmaxyx()
    if h < 20 or w < 80:
        stdscr.addstr(
            0, 0,
            "Terminal too small. Resize to at least 80×20 and retry.",
            curses.A_BOLD
        )
        stdscr.refresh()
        stdscr.getch()
        return

    HEADER_HEIGHT = 10
    header_win = curses.newwin(HEADER_HEIGHT, w, 0, 0)
    digit_win  = curses.newwin(h - HEADER_HEIGHT, w, HEADER_HEIGHT, 0)
    digit_win.scrollok(True)
    digit_win.idlok(True)

    # 4C) Draw static header
    title = "★ φ‐Run Finder (fractional digits, lengths 2…9) ★"
    x_title = max(0, (w // 2) - (len(title) // 2))
    header_win.addstr(0, x_title, title, curses.A_BOLD | curses.color_pair(1))

    print_legend(header_win, w)

    for n in range(2, 10):
        header_win.addstr(n, 2, f"Len {n}: N/A", curses.color_pair(1))

    header_win.refresh()

    # 4D) Stream the STATIC prefix, then switch to LIVE generation
    total_pos = 0     # 1-based index into φ including '1' and '.'
    run_char = None
    run_len = 0
    remaining = set(range(2, 10))
    start_time = time.time()

    # 4D.1) Precomputed static prefix
    static_prefix = get_phi_prefix(PHI_PREFIX_LENGTH)
    for ch in static_prefix:
        total_pos += 1

        # If not a digit, break run but don't print
        if not ch.isdigit():
            run_char = None
            run_len = 0
            continue

        # Are we continuing a run or starting a new one?
        if run_char is None:
            run_char = ch
            run_len = 1
        elif ch == run_char:
            run_len += 1
        else:
            run_char = ch
            run_len = 1

        # If we’ve hit a first-run for length run_len ∈ remaining:
        if run_len in remaining:
            n = run_len
            start_pos = total_pos - n + 1
            elapsed = time.time() - start_time
            seq = run_char * n

            info = f"Len {n}: {seq} @ {start_pos}   [{elapsed:7.3f}s]"
            safe_info = info[: w - 4]

            hl_attr = curses.A_REVERSE | curses.color_pair(HIGHLIGHT_PAIR[n])
            header_win.move(n, 2)
            header_win.clrtoeol()
            header_win.addstr(n, 2, safe_info, hl_attr)
            header_win.refresh()

            # Beep and pause 1 second so you can see it
            sys.stdout.write("\a")
            sys.stdout.flush()
            time.sleep(1)

            remaining.remove(n)
            # Do NOT exit on n == 9; keep streaming

        # Print the digit in the scrolling window:
        if run_len >= 2:
            pair_idx = RUN_COLOR_MAP.get(run_char, 1)
            pair = curses.color_pair(pair_idx)
            if run_char in ('1', '3', '5', '7', '9'):
                pair |= curses.A_BOLD
            digit_win.addstr(ch, pair)
            digit_win.refresh()
        else:
            digit_win.addstr(ch, curses.color_pair(1))
            digit_win.refresh()

    # 4D.2) Switch to live generator after static prefix ends
    def phi_digit_generator():
        getcontext().prec = PHI_PREFIX_LENGTH + 10
        printed = 0
        while True:
            getcontext().prec = printed + 10
            phi = (Decimal(1) + Decimal(5).sqrt()) / Decimal(2)
            s = format(phi, "f")
            while printed < len(s):
                yield s[printed]
                printed += 1

    live_gen = phi_digit_generator()
    try:
        while True:
            ch = next(live_gen)
            total_pos += 1

            if not ch.isdigit():
                run_char = None
                run_len = 0
                continue

            if run_char is None:
                run_char = ch
                run_len = 1
            elif ch == run_char:
                run_len += 1
            else:
                run_char = ch
                run_len = 1

            # After the prefix, remaining is likely empty, so no more header updates.
            # But if you did want to find “first occurrences” beyond 9, you could
            # keep the same logic. For now, simply continue coloring runs.
            if run_len >= 2:
                pair_idx = RUN_COLOR_MAP.get(run_char, 1)
                pair = curses.color_pair(pair_idx)
                if run_char in ('1', '3', '5', '7', '9'):
                    pair |= curses.A_BOLD
                digit_win.addstr(ch, pair)
                digit_win.refresh()
            else:
                digit_win.addstr(ch, curses.color_pair(1))
                digit_win.refresh()

    except KeyboardInterrupt:
        return


def main():
    run_debug_check()
    curses.wrapper(curses_main)


if __name__ == "__main__":
    main()

