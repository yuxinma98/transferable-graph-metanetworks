#!/usr/bin/env bash
# Post a message to Slack via an incoming webhook.
#
# The webhook URL is a secret and is never stored in this repo. It is read from,
# in order:
#   1. $SLACK_WEBHOOK_URL
#   2. $SLACK_WEBHOOK_FILE            (default ~/.config/anydim-notify/slack_webhook)
# Write the file with a single line and chmod 600:
#   mkdir -p ~/.config/anydim-notify
#   printf '%s\n' 'https://hooks.slack.com/services/T.../B.../...' \
#       > ~/.config/anydim-notify/slack_webhook
#   chmod 600 ~/.config/anydim-notify/slack_webhook
#
# Usage:
#   scripts/notify_slack.sh "some *mrkdwn* text"
#   printf 'line1\nline2\n' | scripts/notify_slack.sh
#
# Exit codes: 0 sent, 2 no webhook configured, 1 the POST failed. Callers use the
# distinction to decide whether to retry later (2) or give up on this message (1).
set -uo pipefail

WEBHOOK=${SLACK_WEBHOOK_URL:-}
WEBHOOK_FILE=${SLACK_WEBHOOK_FILE:-$HOME/.config/anydim-notify/slack_webhook}
if [ -z "$WEBHOOK" ] && [ -r "$WEBHOOK_FILE" ]; then
    WEBHOOK=$(tr -d ' \t\n\r' < "$WEBHOOK_FILE")
fi
case "$WEBHOOK" in
    https://hooks.slack.com/*) ;;
    "") echo "notify_slack: no webhook (set SLACK_WEBHOOK_URL or write $WEBHOOK_FILE)" >&2
        exit 2 ;;
    *)  echo "notify_slack: webhook does not look like a Slack incoming-webhook URL" >&2
        exit 2 ;;
esac

if [ $# -gt 0 ]; then TEXT="$*"; else TEXT=$(cat); fi
[ -n "$TEXT" ] || { echo "notify_slack: empty message" >&2; exit 1; }

PYTHON=${PYTHON:-python}
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=python3

# json.dumps rather than hand-rolled quoting: the messages carry backticks,
# quotes and newlines straight out of the summary files.
PAYLOAD=$(printf '%s' "$TEXT" | "$PYTHON" -c \
    'import json,sys; sys.stdout.write(json.dumps({"text": sys.stdin.read(), "mrkdwn": True}))')

OUT=$(mktemp)
trap 'rm -f "$OUT"' EXIT
CODE=$(curl -s -o "$OUT" -w '%{http_code}' --max-time 20 --retry 2 --retry-delay 5 \
    -X POST -H 'Content-type: application/json' --data "$PAYLOAD" "$WEBHOOK")
if [ "$CODE" != "200" ]; then
    echo "notify_slack: HTTP $CODE: $(head -c 300 "$OUT")" >&2
    exit 1
fi
