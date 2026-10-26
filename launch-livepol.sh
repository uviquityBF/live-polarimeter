#!/usr/bin/env bash
# Launch the livepol GUI from the project venv. Extra arguments are passed through,
# e.g. ./launch-livepol.sh --simulate  or  ./launch-livepol.sh --exposure 3000

# Run from the repo root so data/, logs/ and the Arena persist.json land here.
cd "$(dirname "$(readlink -f "$0")")" || exit 1

./.venv/Scripts/livepol.exe "$@"
status=$?

# Keep the window open on failure (e.g. camera held by ArenaView) so the error can be read.
if [ $status -ne 0 ]; then
    echo
    echo "livepol exited with code $status."
    read -rp "Press Enter to close..."
fi
exit $status
