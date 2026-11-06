#!/bin/bash
# Double-click this file to start the app on macOS.
# You can also pass a command, e.g.  ./start_mac.command notebook
cd "$(dirname "$0")" || exit 1

for py in python3.13 python3.12 python3.14 python3; do
    if command -v "$py" >/dev/null 2>&1; then
        "$py" run.py "$@"
        status=$?
        if [ $status -ne 0 ]; then
            echo
            read -r -p "Something went wrong (see above). Press Enter to close."
        fi
        exit $status
    fi
done

echo "Python 3 was not found. Install Python 3.13 from https://www.python.org/downloads/ and try again."
read -r -p "Press Enter to close."
exit 1
