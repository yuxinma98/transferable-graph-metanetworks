#!/usr/bin/env bash
# Watch the running experiment queues and push a Slack message
#   * once per condition, as each one lands in its queue's SUMMARY.md, and
#   * once at the end, when the whole GPU pool has gone quiet.
#
# Send-once semantics: one marker file per (queue, condition) under $STATE_DIR,
# written only after a successful POST. So restarting this watcher never re-sends,
# and if the webhook is not configured yet the messages simply queue up and flush
# the moment it appears (see scripts/notify_slack.sh for where the URL lives).
#
# On its first start it seeds markers for every condition that has *already*
# finished and sends a single "watching" message with the current state, so you
# don't get a burst of history.
#
# "Everything done" = every queue in QUEUES/JOBQUEUES below is drained with no trainer
# left, *or* still on HOLD. The HOLD files make orchestrate_cnn_first.sh's step-3 resume
# a no-op, so while they exist a drained queue really is finished work. Add a new queue to
# QUEUES the moment it is launched: an unlisted queue is invisible to this check, so the
# watcher would send its final rollup and exit while that queue was still running.
#
# Usage:
#   bash scripts/watch_experiments.sh [poll_seconds]      # default 120
#   screen -dmS notify bash scripts/watch_experiments.sh  # how it is meant to run
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

POLL=${1:-120}
STATE_DIR=${STATE_DIR:-/tmp/anydim_notify}
LOG=$STATE_DIR/watch.log
NOTIFY=${NOTIFY:-./scripts/notify_slack.sh}   # overridable to `cat` for a dry run
mkdir -p "$STATE_DIR"

# queue dir | short name | expected conditions | ps pattern identifying its trainers
#
# The CNN queues all run the same script (train_sizegen_predgen.py), so they are told
# apart on --run_name instead: the forward runner emits `...v1-{sgmn,gmn,mpgmn,eval}-...`
# and the bidirectional one `...v1-bidir-...`, hence `v1-[segm]` vs `v1-bidir-`. A `|`
# alternation is not available here — that character delimits these fields. Matching both
# on the script name would report the (finished) forward queue as active for as long as
# the bidirectional one runs.
#
# The CIFAR-10 and SVHN-GS studies then collide on those same suffixes, so each pattern is
# also anchored at the FRONT of the run name — `predgen-` for CIFAR-10, `svhn-predgen-` for
# SVHN-GS. A trailing-context pattern like the old `[a-z0-9.-]*v1-[segm]` cannot separate
# them: `[a-z0-9.-]*` happily absorbs the `svhn-` token, so the CIFAR-10 queues would read
# as active for the whole SVHN run (and vice versa). `(lr-sweep-)?` and `(sizegen-)?` are
# groups, not alternations, so they stay inside the no-`|` rule above.
QUEUES=(
    "/tmp/predgen_sizegen_v1|CNN Stage 3 (CIFAR-10 acc. prediction, train w16)|10|run_name (lr-sweep-)?predgen-(sizegen-)?v1-[segm]"
    "/tmp/predgen_sizegen_v1_bidir|CNN Stage 3 bidir (CIFAR-10 acc. prediction, train w16)|10|run_name (lr-sweep-)?predgen-(sizegen-)?v1-bidir-"
    "/tmp/predgen_svhn_dup_stage2|CNN Stage 2 dup check (SVHN-GS acc. prediction, train w16)|6|run_name svhn-predgen-dup-"
    "/tmp/predgen_svhn_sizegen_v1|CNN Stage 3b (SVHN-GS acc. prediction, train w16)|10|run_name (lr-sweep-)?svhn-predgen-(sizegen-)?v1-[segm]"
    "/tmp/predgen_svhn_sizegen_v1_bidir|CNN Stage 3f bidir (SVHN-GS acc. prediction, train w16)|6|run_name (lr-sweep-)?svhn-predgen-(sizegen-)?v1-bidir-"
    "/tmp/predgen_svhn_sizegen_v2|CNN Stage 3e re-score to w16–w512 (SVHN-GS acc. prediction)|10|run_name svhn-predgen-sizegen-v2-eval-"
    "/tmp/predgen_svhn_sizegen_v2_bidir|CNN Stage 3f bidir re-score to w16–w512 (SVHN-GS)|6|run_name svhn-predgen-sizegen-v2-bidir-"
    "/tmp/mpsgmn_predgen_svhn|CNN Stage 5 matrix-product ScaleGMN (SVHN-GS, train w16)|4|run_name (lr-sweep-)?svhn-predgen-(sizegen-)?mpsgmn"
    "/tmp/sizegen_v3_bidir|MNIST size-gen v3 bidir (train w24)|8|run_name [a-z0-9.-]*v[34]-.*-bidir"
    "/tmp/fmnist_sizegen_v2_bidir|FMNIST size-gen v2 bidir (train w32)|10|run_name [a-z0-9.-]*fmnist-v2-.*-bidir"
)
# The INR patterns deliberately match the `lr-sweep-*` and `*-eval-*` names as well as the
# `sizegen-*` ones. A condition spends most of its wall time in its 7-LR sweep, so anchoring
# on `sizegen-` alone undercounted "N training" in every Slack message and, worse, could read
# a queue as idle while a sweep was in flight. MNIST needs `v[34]` because its eval step is
# named `sizegen-v4-eval-*` while its training is `sizegen-v3-*`.
#
# The SVHN Stage 2 count is 6, not 4: run_svhn_predgen_dup_stage2.sh appends its four trained
# conditions plus one `### Duplicated-width evaluation — {scalegmn,gmn}` section per model.
# Those two eval passes are a blind spot for liveness — they run
# eval_sizegen_predgen_duplicated.py, which names its wandb run internally and takes no
# --run_name, so running_count() cannot see them and the queue reads "idle" while they work.
# Harmless unless every other queue is simultaneously idle for three polls, which would send
# the final rollup early; relaunch the watcher if that happens.
#
# The SVHN size-gen queues do not all expect one condition per training, because base width =
# train width lets one checkpoint be scored under both arms:
#   Stage 3b       10 = 10 trainings, one condition each
#   Stage 3f       6  = 3 trainings x {muP16, SP} re-scores
#   Stage 3e       10 = one eval-only re-score per 3b checkpoint, no training
#   3f re-score    6  = 3 checkpoints x {muP16, SP}
#   Stage 5        4  = 1 training x {muP16, SP} x {v1 w16-w128, v2 w16-w512}
# Their patterns separate `v1-[segm]` (sgmn/gmn/mpgmn/eval) from `v1-bidir-`, and `v2-eval-`
# from `v2-bidir-`, so the forward and bidirectional arms never read as each other. Stage 5's
# `mpsgmn` cannot be absorbed by Stage 3b's `mpgmn` conditions either: 3b's pattern requires a
# literal `v1-` where an mpsgmn run name has `mpsgmn`.

# Queues whose unit of work is an (arm, width) data-generation job rather than a
# trained condition: there is no SUMMARY.md, so progress is read off the runner's own
# stdout, which prints one "[arm wW] done (HH:MM)" line per job and finally touches
# DONE. Their jobs carry no --run_name, hence the separate liveness counter below.
# Same dir | short name | expected jobs | ps pattern layout as QUEUES.
#
# `--arm` vs `--dataset svhn` separates the studies: the args land on one line in the process
# command line even though the runners wrap them, so CIFAR-10 reads `generate_cnn_zoo.py --arm`
# and SVHN-GS `generate_cnn_zoo.py --dataset svhn`, with no overlap.
#
# The two SVHN stages do share their pattern — same command, only the widths differ, and a
# width class cannot separate {16,32,48,64,96,128} from {192,256,384,512} without a `|`.
# Harmless: they are strictly sequential (run_svhn_zoo_gen_v2.sh aborts unless
# /tmp/predgen_svhn_zoo/DONE exists), and job_events() reads each stage's own gen_*.log, so
# only liveness is shared. The one effect is that the idle stage reads "active" rather than
# "stalled" while the other runs, which errs away from a premature final rollup.
#
# Stage 3a expects 12, not the 11 jobs it queues: finalize() also hard-links mup16's w16 from
# sp's (base width = train width, so they are the same weights), and that step runs
# generate_cnn_zoo.py too, so it leaves a twelfth gen_*.log ending in "Done. Next: ".
# The v2 stage has no link step, hence 8 = 8.
JOBQUEUES=(
    "/tmp/predgen_zoo_v2|CNN Stage 3 v2 zoo (w192–w512, both arms)|8|generate_cnn_zoo.py --arm"
    "/tmp/predgen_svhn_zoo|CNN Stage 3a zoo (SVHN-GS, w16–w128, both arms)|12|generate_cnn_zoo.py --dataset svhn"
    "/tmp/predgen_svhn_zoo_v2|CNN Stage 3 v2 zoo (SVHN-GS, w192–w512, both arms)|8|generate_cnn_zoo.py --dataset svhn"
)

say() { printf '[%s] %s\n' "$(date '+%m-%d %H:%M')" "$*" >> "$LOG"; }

# ------------------------------------------------------------------ primitives

slug() { printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | tr -cs '[:alnum:]' '-' | sed 's/^-//;s/-$//'; }

# non-blank lines of a queue file (missing or empty file -> 0). grep -c exits 1 on
# zero matches, so the count has to be captured rather than chained with ||.
queue_len() {
    local n=0
    [ -f "$1" ] && n=$(grep -c '[^[:space:]]' "$1" 2>/dev/null)
    printf '%s' "${n:-0}"
}

# distinct --run_name values currently running for a ps pattern
running_count() {
    ps -eo cmd 2>/dev/null | grep -E "$1" | grep -v grep \
        | grep -oE -- '--run_name [^ ]+' | sort -u | wc -l
}

# processes matching a pattern, for job queues whose jobs carry no --run_name.
# grep -c exits 1 on zero matches, so the count is captured rather than chained.
proc_count() {
    local n
    n=$(ps -eo cmd 2>/dev/null | grep -E "$1" | grep -v grep | grep -c . 2>/dev/null)
    printf '%s' "${n:-0}"
}

# finished jobs of a JOBQUEUES runner, as "arm wWIDTH".
#
# Read off the per-job generator logs, NOT the runner's `[arm wW] done` line: that line
# goes to the runner's stdout, i.e. into its screen session, and never lands in $LOG_DIR.
# generate_cnn_zoo.py ends a completed job with "Done. Next: ... generate_predgen_splits",
# after the last role, so the presence of that line in gen_{arm}_w{width}.log is the
# per-job completion signal — and it survives a runner restart, which stdout does not.
job_events() {
    grep -lE '^Done\. Next: ' "$1"/gen_*.log 2>/dev/null \
        | sed -E 's|.*/gen_([a-z0-9]+)_w([0-9]+)\.log$|\1 w\2|' | sort -u
}

# the '### ' condition headings of a summary file, in order
headings() { [ -f "$1" ] && grep -E '^### ' "$1" | sed 's/^### //' || true; }

# the block of SUMMARY.md ($1) belonging to condition heading ($2)
block() {
    awk -v want="### $2" '
        $0 == want { inblk = 1; next }
        inblk && /^### / { exit }
        inblk { print }
    ' "$1"
}

# ------------------------------------------------------------------ formatting

# "w16 0.9315 · w32 0.9120 · ..." from the block's markdown table (col 1 + col 2)
width_line() {
    awk -F'|' '
        /^\| *w[0-9]+ / {
            gsub(/^ +| +$/, "", $2); gsub(/^ +| +$/, "", $3)
            sub(/ *\((IN|OUT)\)/, "", $2)
            out = out (out ? " · " : "") $2 " " $3
        }
        END { print out }
    '
}

# one Slack message for a finished condition
condition_message() {
    local qname=$1 label=$2 done_n=$3 total=$4 blk=$5 qlen=$6 nrun=$7
    local lr best widths ood wandb icon ident
    lr=$(printf '%s\n' "$blk"     | grep -m1 -oE 'best LR: \*\*[^*]+\*\*' | sed 's/best LR: \*\*//;s/\*\*//')
    best=$(printf '%s\n' "$blk"   | grep -m1 -E '^- best val' | sed 's/^- //')
    widths=$(printf '%s\n' "$blk" | width_line)
    ood=$(printf '%s\n' "$blk"    | grep -m1 -E '^OOD mean' || true)
    wandb=$(printf '%s\n' "$blk"  | grep -m1 -oE 'https://wandb.ai/[^ )]+' || true)
    # CNN bidirectional arm only: five trainings serve ten conditions, and this audit line is
    # what says the shared-checkpoint premise held. Empty for every other queue.
    # capture the bolded value explicitly: `s/.*\*\*//` would be greedy and eat the closing
    # `**` too, leaving an empty string
    ident=$(printf '%s\n' "$blk"  | grep -m1 -oE 'w16 identity vs the other arm: \*\*[^*]+\*\*' \
                                  | sed -E 's/.*\*\*([^*]+)\*\*.*/\1/')

    icon=':white_check_mark:'
    [ -z "$widths" ] && icon=':warning:'          # no per-width table = the run did not reach eval
    [ "$ident" = MISMATCH ] && icon=':rotating_light:'

    {
        printf '%s *%s* — %s/%s conditions done\n' "$icon" "$qname" "$done_n" "$total"
        printf '*%s*\n' "$label"
        [ -n "$lr" ]   && printf '• best LR `%s`\n' "$lr"
        [ -n "$best" ] && printf '• %s\n' "$best"
        if [ -n "$widths" ]; then printf '• %s\n' "$widths"
        else printf '• :warning: no per-width table in the summary — check the log\n'; fi
        [ -n "$ood" ]   && printf '• %s\n' "$ood"
        case "$ident" in
            MISMATCH) printf '• :rotating_light: w16 identity *MISMATCH* — this condition'\''s other-arm column is not trustworthy\n' ;;
            ''|n/a)   ;;
            *)        printf '• w16 identity vs the other arm: %s\n' "$ident" ;;
        esac
        [ -n "$wandb" ] && printf '• %s\n' "$wandb"
        printf '_queue: %s waiting, %s training_\n' "$qlen" "$nrun"
    }
}

# try to send stdin; mark $1 only on success. Returns 0 sent, 1 keep for later.
send_and_mark() {
    local marker=$1 msg rc=0
    msg=$(cat)
    # the status has to be caught on the pipeline itself: an `if` whose condition
    # fails and which has no `else` exits 0, so `$?` after `fi` is not the sender's.
    printf '%s' "$msg" | $NOTIFY || rc=$?
    if [ "$rc" -eq 0 ]; then
        : > "$marker"
        return 0
    fi
    if [ "$rc" -eq 2 ]; then
        say "webhook not configured yet — holding message for $(basename "$marker")"
    else
        say "POST failed for $(basename "$marker") — will retry next poll"
    fi
    return 1
}

# ------------------------------------------------------------------ activity

# echoes "active", "held" or "idle" for a queue spec
queue_state() {
    local dir=$1 pat=$2
    [ -f "$dir/HOLD" ] && { echo held; return; }
    if [ "$(queue_len "$dir/queue.txt")" -gt 0 ] || [ "$(running_count "$pat")" -gt 0 ]; then
        echo active
    else
        echo idle
    fi
}

# same for a JOBQUEUES spec. "stalled" -- empty queue, nothing running, no DONE
# sentinel -- is reported rather than folded into "idle" because it means the runner
# died mid-queue: not a blocker for the terminal check, but something to surface.
job_queue_state() {
    local dir=$1 pat=$2
    [ -f "$dir/DONE" ] && { echo idle; return; }
    if [ "$(queue_len "$dir/queue.txt")" -gt 0 ] || [ "$(proc_count "$pat")" -gt 0 ]; then
        echo active
    else
        echo stalled
    fi
}

# ------------------------------------------------------------------ seeding

SEEDED=$STATE_DIR/.seeded
SNAPSHOT=$STATE_DIR/.startup_conditions

# Called at the top of every poll, not once at startup: if the webhook is not in
# place yet the seed message cannot be sent, and seeding anyway would silently eat
# the only "it works" confirmation. Until it succeeds nothing is marked and nothing
# is sent, so the first successful poll seeds history and then reports only what is
# genuinely new.
#
# "History" is the snapshot taken when this process started, NOT what exists when
# the webhook finally shows up: a condition that lands in between is a real result
# the user has not seen, and seeding it would silently swallow it.
seed_once() {
    [ -f "$SEEDED" ] && return 0
    say "first start — seeding markers for the $(grep -c . "$SNAPSHOT" 2>/dev/null || echo 0) conditions already finished at startup"
    if ! {
        printf ':satellite: *anydim-metanet watcher started* (%s)\n' "$(date '+%a %b %d %H:%M')"
        printf 'One message per condition from here on, plus a final one when the pool goes quiet.\n'
        for spec in "${QUEUES[@]}"; do
            IFS='|' read -r dir qname total pat <<< "$spec"
            n=$(headings "$dir/SUMMARY.md" | grep -c . || true)
            st=$(queue_state "$dir" "$pat")
            printf '• *%s* — %s/%s done, %s waiting, %s training (%s)\n' \
                "$qname" "${n:-0}" "$total" "$(queue_len "$dir/queue.txt")" \
                "$(running_count "$pat")" "$st"
        done
        for spec in "${JOBQUEUES[@]}"; do
            IFS='|' read -r dir qname total pat <<< "$spec"
            n=$(job_events "$dir" | grep -c . || true)
            printf '• *%s* — %s/%s jobs done, %s waiting, %s generating (%s)\n' \
                "$qname" "${n:-0}" "$total" "$(queue_len "$dir/queue.txt")" \
                "$(proc_count "$pat")" "$(job_queue_state "$dir" "$pat")"
        done
        # `if`, not `[ -f ] && printf`: with pipefail a failing test as the group's last
        # command makes the whole `{ ... } | $NOTIFY` pipeline non-zero, so a message that
        # was in fact posted would be read as "not sent" and re-sent every poll.
        for spec in "${QUEUES[@]}"; do
            IFS='|' read -r dir qname total pat <<< "$spec"
            if [ -f "$dir/HOLD" ]; then
                printf '_%s is on HOLD: %s_\n' "$qname" "$(cat "$dir/HOLD")"
            fi
        done
    } | $NOTIFY; then
        say "startup message not sent (webhook missing?) — will retry next poll"
        return 1
    fi

    while IFS= read -r marker; do
        [ -n "$marker" ] || continue
        : > "$STATE_DIR/$marker"
    done < "$SNAPSHOT"
    : > "$SEEDED"
    say "seeded; reporting only new conditions from here on"
}

# Snapshot of what was already finished when this process started, taken before the
# first send attempt so a late webhook cannot expand it (see seed_once).
if [ ! -f "$SEEDED" ] && [ ! -f "$SNAPSHOT" ]; then
    for spec in "${QUEUES[@]}"; do
        IFS='|' read -r dir qname total pat <<< "$spec"
        while IFS= read -r label; do
            [ -n "$label" ] || continue
            printf '%s__%s\n' "$(slug "$qname")" "$(slug "$label")"
        done < <(headings "$dir/SUMMARY.md")
    done > "$SNAPSHOT"
fi

# ------------------------------------------------------------------ main loop

say "watching (poll ${POLL}s); state in $STATE_DIR"
IDLE_STREAK=0
while true; do
    seed_once
    for spec in "${QUEUES[@]}"; do
        IFS='|' read -r dir qname total pat <<< "$spec"
        SUM=$dir/SUMMARY.md
        [ -f "$SUM" ] || continue
        qslug=$(slug "$qname")
        done_n=0
        while IFS= read -r label; do
            [ -n "$label" ] || continue
            done_n=$((done_n + 1))
            marker=$STATE_DIR/${qslug}__$(slug "$label")
            [ -f "$marker" ] && continue
            say "new condition: $qname / $label"
            condition_message "$qname" "$label" "$done_n" "$total" \
                "$(block "$SUM" "$label")" \
                "$(queue_len "$dir/queue.txt")" "$(running_count "$pat")" \
                | send_and_mark "$marker"
        done < <(headings "$SUM")
    done

    # ---- per-job messages for the data-generation queues
    for spec in "${JOBQUEUES[@]}"; do
        IFS='|' read -r dir qname total pat <<< "$spec"
        qslug=$(slug "$qname")
        done_n=0
        while IFS= read -r label; do
            [ -n "$label" ] || continue
            done_n=$((done_n + 1))
            marker=$STATE_DIR/${qslug}__$(slug "$label")
            [ -f "$marker" ] && continue
            say "new job: $qname / $label"
            {
                printf ':package: *%s* — %s/%s jobs done\n' "$qname" "$done_n" "$total"
                printf '*%s* generated\n' "$label"
                printf '_queue: %s waiting, %s generating_\n' \
                    "$(queue_len "$dir/queue.txt")" "$(proc_count "$pat")"
            } | send_and_mark "$marker"
        done < <(job_events "$dir")
        # the sentinel is the only signal that the splits were written too, which is
        # what actually unblocks the re-score -- so it gets its own message
        if [ -f "$dir/DONE" ] && [ ! -f "$STATE_DIR/${qslug}__done-sentinel" ]; then
            {
                printf ':white_check_mark: *%s* — complete (%s)\n' "$qname" "$(cat "$dir/DONE")"
                printf 'All %s jobs generated and the splits are written.\n' "$total"
            } | send_and_mark "$STATE_DIR/${qslug}__done-sentinel"
        fi
    done

    # ---- terminal check: nothing active anywhere, sustained over 3 polls
    ANY_ACTIVE=0
    STATES=""
    for spec in "${QUEUES[@]}"; do
        IFS='|' read -r dir qname total pat <<< "$spec"
        st=$(queue_state "$dir" "$pat")
        [ "$st" = active ] && ANY_ACTIVE=1
        STATES="$STATES$qname=$st "
    done
    for spec in "${JOBQUEUES[@]}"; do
        IFS='|' read -r dir qname total pat <<< "$spec"
        st=$(job_queue_state "$dir" "$pat")
        [ "$st" = active ] && ANY_ACTIVE=1
        STATES="$STATES$qname=$st "
    done

    if [ "$ANY_ACTIVE" -eq 0 ]; then
        IDLE_STREAK=$((IDLE_STREAK + 1))
        say "no active queue (streak $IDLE_STREAK/3): $STATES"
    else
        IDLE_STREAK=0
    fi

    if [ "$IDLE_STREAK" -ge 3 ] && [ ! -f "$STATE_DIR/.final_sent" ]; then
        say "sending final rollup"
        {
            printf ':checkered_flag: *Everything is done* — the GPU pool is quiet (%s)\n\n' \
                "$(date '+%a %b %d %H:%M')"
            for spec in "${QUEUES[@]}"; do
                IFS='|' read -r dir qname total pat <<< "$spec"
                n=$(headings "$dir/SUMMARY.md" 2>/dev/null | grep -c . || true)
                printf '*%s* — %s/%s conditions\n' "$qname" "${n:-0}" "$total"
                while IFS= read -r label; do
                    [ -n "$label" ] || continue
                    printf '• %s — %s\n' "$label" \
                        "$(block "$dir/SUMMARY.md" "$label" | width_line)"
                done < <(headings "$dir/SUMMARY.md")
                [ -f "$dir/HOLD" ] && printf '_on HOLD: %s_\n' "$(cat "$dir/HOLD")"
                printf '\n'
            done
            for spec in "${JOBQUEUES[@]}"; do
                IFS='|' read -r dir qname total pat <<< "$spec"
                printf '*%s* — %s/%s jobs, %s\n\n' "$qname" \
                    "$(job_events "$dir" | grep -c . || true)" "$total" \
                    "$(job_queue_state "$dir" "$pat")"
            done
            printf 'Summaries:'
            for spec in "${QUEUES[@]}"; do
                IFS='|' read -r dir qname total pat <<< "$spec"
                printf ' `%s/SUMMARY.md`' "$dir"
            done
            printf '\n'
            for spec in "${QUEUES[@]}"; do
                IFS='|' read -r dir qname total pat <<< "$spec"
                if [ -f "$dir/HOLD" ]; then
                    printf 'Still parked: `rm %s/HOLD` and relaunch its runner to resume %s.\n' "$dir" "$qname"
                fi
            done
            : # keep the group's exit status clean for pipefail (see seed_once)
        } | send_and_mark "$STATE_DIR/.final_sent" && { say "final sent; exiting"; exit 0; }
    fi

    sleep "$POLL"
done
