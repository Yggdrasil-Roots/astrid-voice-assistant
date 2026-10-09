#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# Ollama runs as a systemd service and should already be up, but just in case
# (e.g. it was manually stopped), wait a few seconds for it to respond.
if ! curl -s -o /dev/null -w "" http://localhost:11434/ 2>/dev/null; then
    echo "Ollama not responding, starting it..."
    systemctl --user start ollama 2>/dev/null || sudo systemctl start ollama 2>/dev/null || true
    for i in $(seq 1 10); do
        curl -s -o /dev/null http://localhost:11434/ 2>/dev/null && break
        sleep 1
    done
fi

# SearXNG (web search) on localhost:8080. tools.py wants the JSON API there.
#
# The Open WebUI stack now publishes its own SearXNG on 127.0.0.1:8080 and it
# answers ?format=json, so Astrid's dedicated astrid-searxng container is no
# longer started -- two containers cannot hold the same port, and the one
# already running serves her fine. Only fall back to her own if nothing
# answers, which is the case on a machine where the new stack is down.
if curl -s -o /dev/null -m 3 "http://localhost:8080/search?q=ping&format=json"; then
    echo "SearXNG already serving on localhost:8080; using it."
elif [ -d searxng ]; then
    echo "Nothing on localhost:8080, starting Astrid's own SearXNG..."
    (cd searxng && docker compose up -d) 2>/dev/null || true
fi

source venv/bin/activate
# Desktop-icon launches (Terminal=false in the .desktop file) get no visible
# stdout/stderr at all, and Python fully buffers both when not attached to a
# tty -- so a crash leaves no trace until the buffer happens to flush on exit,
# which is why two real crashes showed up in the journal as nothing but a
# stray startup log line. -u disables that buffering; logging to a file
# means a crash is actually diagnosable next time. A real terminal still gets
# live output, same as before, for `./start.sh` during manual debugging.
if [ -t 1 ]; then
    exec python3 -u gui.py
else
    LOG="$HOME/.astrid/astrid.log"
    mkdir -p "$(dirname "$LOG")"
    exec python3 -u gui.py >> "$LOG" 2>&1
fi
