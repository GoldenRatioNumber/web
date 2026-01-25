#!/bin/bash
# auto_sync.sh - Automatically push live_status.json to GitHub every 30 seconds
# Run this in a separate terminal while goldenratiofinder.py is running

cd /Users/david/goldenrationumbers

echo "Starting auto-sync to GitHub..."
echo "Press Ctrl+C to stop"
echo ""

while true; do
    # Check if live_status.json has changed
    if git diff --quiet live_status.json 2>/dev/null; then
        echo "$(date '+%H:%M:%S') - No changes"
    else
        echo "$(date '+%H:%M:%S') - Pushing update..."
        git add live_status.json
        git commit -m "Auto-update live status $(date '+%Y-%m-%d %H:%M:%S')" --quiet
        git push --quiet
        echo "$(date '+%H:%M:%S') - Done!"
    fi
    sleep 30
done
