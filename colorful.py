#!/usr/bin/env python3
# colorful.py - Interactive φ Run Finder with curses UI
# pip3 install mpmath --break-system-packages
# python3 colorful.py
import curses
import sys
import time
from mpmath import mp

# ─────────────────────────────────────────────────────────────────────────────
# GLOBALS
# ─────────────────────────────────────────────────────────────────────────────

DIGITS = 333_333_333  # Calculate 333 million digits to find all 9 patterns

# digit '0'..'9' → curses color_pair index (for runs ≥ 2)
RUN_COLOR_MAP = {
    '0': 2,  '1': 2,  '2': 3,  '3': 3,  '4': 4,
    '5': 4,  '6': 5,  '7': 5,  '8': 6,  '9': 6,
}

# run‐length n → highlight color_pair index (with A_REVERSE for white background)
HIGHLIGHT_PAIR = {2:2, 3:4, 4:3, 5:7, 6:8, 7:9, 8:10, 9:11}


# ─────────────────────────────────────────────────────────────────────────────
# LEGEND
# ─────────────────────────────────────────────────────────────────────────────

# Known patterns in φ
KNOWN_PATTERNS = {
    2: ("33", 7),
    3: ("222", 131),
    4: ("4444", 1218),
    5: ("99999", 6401),
    6: ("555555", 99790),
    7: ("5555555", 771952),
    8: ("55555555", 771952),
    9: ("333333333", 314529196),
}

def print_legend(header_win, width):
    """Draw the legend on row 1 - show length 2-9 with single digit samples."""
    header_win.addstr(1, 2, "Len: ", curses.A_BOLD)
    col = 7
    for n in range(2, 10):
        digit = KNOWN_PATTERNS[n][0][0]  # Single digit from known pattern
        label = f"{n}={digit}"
        attr = curses.A_REVERSE | curses.color_pair(HIGHLIGHT_PAIR[n])
        if col + len(label) + 2 < width - 2:
            header_win.addstr(1, col, label, attr)
            col += len(label) + 1


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CURSES UI
# ─────────────────────────────────────────────────────────────────────────────

def curses_main(stdscr):
    """
    Main curses loop. Shows every digit scrolling in real-time with colors.
    Pauses 1 second when a new pattern is found, then continues.
    """
    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()

    # Initialize color pairs
    curses.init_pair(1, curses.COLOR_WHITE,   -1)
    curses.init_pair(2, curses.COLOR_RED,     -1)
    curses.init_pair(3, curses.COLOR_GREEN,   -1)
    curses.init_pair(4, curses.COLOR_YELLOW,  -1)
    curses.init_pair(5, curses.COLOR_BLUE,    -1)
    curses.init_pair(6, curses.COLOR_MAGENTA, -1)
    curses.init_pair(7, curses.COLOR_CYAN,    -1)
    curses.init_pair(8, curses.COLOR_BLUE,    -1)
    curses.init_pair(9, curses.COLOR_MAGENTA, -1)
    curses.init_pair(10, curses.COLOR_RED,    -1)
    curses.init_pair(11, curses.COLOR_YELLOW, -1)

    h, w = stdscr.getmaxyx()
    if h < 20 or w < 80:
        stdscr.addstr(0, 0, "Terminal too small. Resize to at least 80×20.", curses.A_BOLD)
        stdscr.refresh()
        stdscr.getch()
        return

    HEADER_HEIGHT = 10
    header_win = curses.newwin(HEADER_HEIGHT, w, 0, 0)
    digit_win = curses.newwin(h - HEADER_HEIGHT, w, HEADER_HEIGHT, 0)
    digit_win.scrollok(True)
    digit_win.idlok(True)

    # Draw title
    title = "★ φ‐Run Finder (333M digits, lengths 2…9) ★"
    x_title = max(0, (w // 2) - (len(title) // 2))
    header_win.addstr(0, x_title, title, curses.A_BOLD | curses.color_pair(1))

    # Draw compact legend on row 1
    print_legend(header_win, w)

    # Initial "Len n: N/A" for each pattern on rows 2-9
    for n in range(2, 10):
        header_win.addstr(n, 2, f"Len {n}: searching...", curses.color_pair(1))
    header_win.refresh()

    # Show computing message
    digit_win.addstr("Computing φ to 333,333,333 digits...\n", curses.color_pair(4))
    digit_win.refresh()

    # Compute phi with mpmath
    compute_start = time.time()
    mp.dps = DIGITS + 100
    phi_str = mp.nstr((1 + mp.sqrt(5)) / 2, DIGITS + 2, strip_zeros=False)
    compute_time = time.time() - compute_start

    digit_win.addstr(f"Computed {len(phi_str)-2:,} digits in {compute_time:.1f}s\n\n", curses.color_pair(3))
    digit_win.refresh()
    time.sleep(1)

    # Stream digits
    total_pos = 0
    run_char = None
    run_len = 0
    remaining = set(range(2, 10))
    start_time = time.time()

    for ch in phi_str:
        total_pos += 1

        # Skip non-digits (like '1' and '.')
        if not ch.isdigit():
            run_char = None
            run_len = 0
            continue

        # Track runs
        if run_char is None:
            run_char = ch
            run_len = 1
        elif ch == run_char:
            run_len += 1
        else:
            run_char = ch
            run_len = 1

        # Check if we found a new pattern
        if run_len in remaining:
            n = run_len
            start_pos = total_pos - n + 1
            elapsed = time.time() - start_time
            seq = run_char * n

            # Format time
            if elapsed >= 3600:
                time_str = f"{int(elapsed//3600)}h {int((elapsed%3600)//60)}m {elapsed%60:.0f}s"
            elif elapsed >= 60:
                time_str = f"{int(elapsed//60)}m {elapsed%60:.1f}s"
            else:
                time_str = f"{elapsed:.3f}s"

            info = f"Len {n}: {seq} @ {start_pos:,}   [{time_str}]"
            safe_info = info[: w - 4]

            hl_attr = curses.A_REVERSE | curses.color_pair(HIGHLIGHT_PAIR[n])
            header_win.move(n, 2)
            header_win.clrtoeol()
            header_win.addstr(n, 2, safe_info, hl_attr)
            header_win.refresh()

            # Beep and pause 1 second
            curses.beep()
            time.sleep(1)

            remaining.remove(n)

        # Print the digit with color
        if run_len >= 2:
            pair_idx = RUN_COLOR_MAP.get(run_char, 1)
            pair = curses.color_pair(pair_idx)
            if run_char in ('1', '3', '5', '7', '9'):
                pair |= curses.A_BOLD
            try:
                digit_win.addstr(ch, pair)
            except curses.error:
                pass
        else:
            try:
                digit_win.addstr(ch, curses.color_pair(1))
            except curses.error:
                pass

        digit_win.refresh()

    # Done - wait for keypress
    total_time = time.time() - start_time
    if total_time >= 3600:
        time_str = f"{int(total_time//3600)}h {int((total_time%3600)//60)}m"
    else:
        time_str = f"{int(total_time//60)}m {total_time%60:.0f}s"

    digit_win.addstr(f"\n\n═══ COMPLETE ═══ All 9 patterns found in {time_str}\n", curses.color_pair(3) | curses.A_BOLD)
    digit_win.addstr("Press any key to exit.", curses.color_pair(1))
    digit_win.refresh()
    stdscr.nodelay(False)
    stdscr.getch()


def main():
    print("★ φ‐Run Finder ★")
    print(f"Target: {DIGITS:,} digits")
    print("Starting curses interface...\n")
    curses.wrapper(curses_main)


if __name__ == "__main__":
    main()
