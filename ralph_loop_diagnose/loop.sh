#!/bin/bash

# BasisSimulator — Ghost Artifact Diagnosis Loop
#
# Ralph loop: runs Claude Code in a loop where each iteration is a
# FRESH agent with zero memory. State persists via files + git only.
#
# Usage:
#   ./ralph_loop_diagnose/loop.sh          # Default 15 iterations
#   ./ralph_loop_diagnose/loop.sh 5        # Custom max iterations

set -e

# ─── Configuration ───────────────────────────────────────────────────────────

MAX_ITERATIONS="${1:-15}"
PROJECT_DIR="/Users/daleblack/Documents/dev/CTSimulatorStuff/BasisSimulator.jl"
RALPH_DIR="$PROJECT_DIR/ralph_loop_diagnose"
PRD_FILE="$RALPH_DIR/prd.json"
PROGRESS_FILE="$RALPH_DIR/progress.md"
PROMPT_FILE="$RALPH_DIR/prompt.md"
LOG_FILE="$RALPH_DIR/loop_log.txt"

# Exit markers (agent outputs these to signal state)
COMPLETE_MARKER="RALPH_COMPLETE"
BLOCKED_MARKER="RALPH_BLOCKED"

# Seconds between iterations (rate limiting)
COOLDOWN=5

# Timeout per iteration (60 minutes — GPU CT simulations take time)
MAX_WAIT=3600

# ─── Colors ──────────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

# ─── Helpers ─────────────────────────────────────────────────────────────────

check_files() {
    for file in "$PRD_FILE" "$PROGRESS_FILE" "$PROMPT_FILE"; do
        if [ ! -f "$file" ]; then
            echo -e "${RED}Missing: $file${NC}"
            exit 1
        fi
    done
}

count_remaining() {
    local total=$(grep -c '"id":' "$PRD_FILE" 2>/dev/null || true)
    local done=$(grep -c '"status": "done"' "$PRD_FILE" 2>/dev/null || true)
    echo $(( ${total:-0} - ${done:-0} ))
}

count_done() {
    grep -c '"status": "done"' "$PRD_FILE" 2>/dev/null || true
}

count_total() {
    grep -c '"id":' "$PRD_FILE" 2>/dev/null || true
}

print_status() {
    local done=$(count_done)
    local total=$(count_total)
    local remaining=$((${total:-0} - ${done:-0}))
    echo -e "  ${GREEN}${done:-0} done${NC} / ${YELLOW}${remaining} remaining${NC} / ${total:-0} total"
}

log_msg() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg" >> "$LOG_FILE"
    echo -e "$msg"
}

# Get list of done story IDs
get_done_stories() {
    python3 -c "
import json, sys
with open('$PRD_FILE') as f:
    data = json.load(f)
for s in data.get('stories', []):
    if s.get('status') == 'done':
        print(s['id'])
" 2>/dev/null | sort
}

# Compare done stories before/after
check_story_completions() {
    local before_file="$1"
    local after_stories
    after_stories=$(get_done_stories)

    local new_stories
    new_stories=$(comm -13 "$before_file" <(echo "$after_stories") 2>/dev/null || true)

    if [ -n "$new_stories" ]; then
        echo ""
        echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║  STORY COMPLETED                                             ║${NC}"
        echo "$new_stories" | while read -r story_id; do
            local title
            title=$(python3 -c "
import json
with open('$PRD_FILE') as f:
    data = json.load(f)
for s in data.get('stories', []):
    if s['id'] == '$story_id':
        print(s.get('title', ''))
        break
" 2>/dev/null || echo "")
            printf "${GREEN}║${NC}  %-58s ${GREEN}║${NC}\n" "$story_id: $title"
            log_msg "STORY COMPLETED: $story_id — $title"
        done
        echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
        print_status
        echo ""

        # macOS notification
        local count
        count=$(echo "$new_stories" | wc -l | tr -d ' ')
        osascript -e "display notification \"$new_stories\" with title \"Diagnosis Loop\" subtitle \"$count story(ies) completed\"" 2>/dev/null || true
    fi
}

# Auto-commit on timeout
auto_commit_on_timeout() {
    cd "$PROJECT_DIR"
    local has_changes=false

    if ! git diff --quiet ralph_loop_diagnose/ 2>/dev/null; then
        has_changes=true
    fi
    if ! git diff --cached --quiet ralph_loop_diagnose/ 2>/dev/null; then
        has_changes=true
    fi

    if [ "$has_changes" = true ]; then
        echo -e "${YELLOW}  Auto-committing changes on timeout...${NC}"
        git add ralph_loop_diagnose/ 2>/dev/null || true
        git commit -m "[TIMEOUT] Auto-save diagnosis progress" 2>/dev/null || true
        log_msg "TIMEOUT: Auto-committed changes"
    else
        log_msg "TIMEOUT: No changes to commit"
    fi
}

# ─── Pre-flight ──────────────────────────────────────────────────────────────

cd "$PROJECT_DIR"
check_files

# Create outputs directory
mkdir -p "$RALPH_DIR/outputs"

echo ""
echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC}  ${BOLD}BasisSimulator — Ghost Artifact Diagnosis${NC}               ${BLUE}║${NC}"
echo -e "${BLUE}║${NC}  Max iterations: $MAX_ITERATIONS                                       ${BLUE}║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
print_status
echo ""
log_msg "Loop started. Max iterations: $MAX_ITERATIONS"

# ─── Main Loop ───────────────────────────────────────────────────────────────

iteration=0
while [ $iteration -lt $MAX_ITERATIONS ]; do
    iteration=$((iteration + 1))

    # Check if all stories are done
    remaining=$(count_remaining)
    if [ "$remaining" -eq 0 ]; then
        echo -e "${GREEN}All stories complete. Diagnosis finished.${NC}"
        log_msg "COMPLETE: All stories done"
        exit 0
    fi

    echo -e "${YELLOW}── Iteration $iteration/$MAX_ITERATIONS ──${NC}"
    print_status
    log_msg "Iteration $iteration started. $remaining stories remaining."

    # Snapshot done stories BEFORE this iteration
    DONE_BEFORE="/tmp/ralph_diag_done_before_$$_${iteration}"
    get_done_stories > "$DONE_BEFORE"

    # Run Claude Code with fresh context
    PROMPT_CONTENT=$(cat "$PROMPT_FILE")
    START_TIME=$(date +%s)
    TEMP_OUTPUT="/tmp/ralph_diag_output_$$_${iteration}"

    claude --print --dangerously-skip-permissions "$PROMPT_CONTENT" > "$TEMP_OUTPUT" 2>&1 &
    CLAUDE_PID=$!

    # Wait with timeout
    TIMED_OUT=false
    while ps -p $CLAUDE_PID > /dev/null 2>&1; do
        ELAPSED=$(($(date +%s) - START_TIME))
        printf "\r  Working... %dm%02ds" $((ELAPSED/60)) $((ELAPSED%60))
        if [ $ELAPSED -ge $MAX_WAIT ]; then
            echo -e "\n${RED}  Timeout after ${MAX_WAIT}s. Killing agent.${NC}"
            kill $CLAUDE_PID 2>/dev/null
            wait $CLAUDE_PID 2>/dev/null || true
            TIMED_OUT=true
            auto_commit_on_timeout
            break
        fi
        sleep 3
    done

    DURATION=$(($(date +%s) - START_TIME))
    OUTPUT=$(cat "$TEMP_OUTPUT" 2>/dev/null || echo "")
    rm -f "$TEMP_OUTPUT"

    echo -e "\r  Done in ${DURATION}s                    "
    log_msg "Iteration $iteration finished in ${DURATION}s (timeout=$TIMED_OUT)"

    # Show last 30 lines of output
    echo "$OUTPUT" | tail -30

    # Check for story completions
    check_story_completions "$DONE_BEFORE"
    rm -f "$DONE_BEFORE"

    # Exit conditions
    if echo "$OUTPUT" | grep -q "$COMPLETE_MARKER"; then
        echo ""
        echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║  RALPH_COMPLETE — Ghost artifact diagnosis done!             ║${NC}"
        echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
        log_msg "RALPH_COMPLETE after $iteration iterations"
        exit 0
    fi

    if echo "$OUTPUT" | grep -q "$BLOCKED_MARKER"; then
        echo ""
        echo -e "${RED}Agent is blocked. Check progress.md for details.${NC}"
        log_msg "RALPH_BLOCKED after $iteration iterations"
        exit 1
    fi

    # Cooldown
    if [ $iteration -lt $MAX_ITERATIONS ]; then
        sleep $COOLDOWN
    fi
done

# Max iterations reached
echo ""
echo -e "${RED}Max iterations ($MAX_ITERATIONS) reached. Run again to continue.${NC}"
print_status
log_msg "Max iterations ($MAX_ITERATIONS) reached"
exit 1
