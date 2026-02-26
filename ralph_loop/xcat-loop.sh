#!/bin/bash

# BasisSimulator — XCAT PCCT Artifact Investigation Ralph Loop
#
# Runs Claude Code in a loop. Each iteration is a FRESH agent with zero memory.
# State persists via: xcat-prd.json (stories), xcat-progress.md (log), git (code).
#
# Investigation: PCCT FDK shows cupping/edge artifacts not present in EICT.
# Hypotheses: BHC over-correction, I0_bins mismatch, missing physics effects.
#
# Usage:
#   ./ralph_loop/xcat-loop.sh          # Default 30 iterations
#   ./ralph_loop/xcat-loop.sh 10       # Custom max iterations

set -e

# ─── Configuration ───────────────────────────────────────────────────────────

MAX_ITERATIONS="${1:-30}"
PROJECT_DIR="/Users/daleblack/Documents/dev/CTSimulatorStuff/BasisSimulator.jl"
RALPH_DIR="$PROJECT_DIR/ralph_loop"
PRD_FILE="$RALPH_DIR/xcat-prd.json"
PROGRESS_FILE="$RALPH_DIR/xcat-progress.md"
PROMPT_FILE="$RALPH_DIR/xcat-prompt.md"
CONTEXT_FILE="$RALPH_DIR/xcat-context.md"
GUARDRAILS_FILE="$RALPH_DIR/xcat-guardrails.md"
LOG_FILE="$RALPH_DIR/xcat-loop-log.txt"

# Exit markers (agent outputs these to signal state)
COMPLETE_MARKER="RALPH_COMPLETE"
BLOCKED_MARKER="RALPH_BLOCKED"

# Seconds between iterations (rate limiting)
COOLDOWN=5

# Timeout per iteration (60 minutes — Julia JIT + GPU compilation needs time)
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
    for file in "$PRD_FILE" "$PROGRESS_FILE" "$PROMPT_FILE" "$CONTEXT_FILE" "$GUARDRAILS_FILE"; do
        if [ ! -f "$file" ]; then
            echo -e "${RED}Missing: $file${NC}"
            exit 1
        fi
    done
}

count_open() {
    grep -c '"status": "open"' "$PRD_FILE" 2>/dev/null || echo "0"
}

count_done() {
    grep -c '"status": "done"' "$PRD_FILE" 2>/dev/null || echo "0"
}

count_total() {
    grep -c '"id":' "$PRD_FILE" 2>/dev/null || echo "0"
}

print_status() {
    local done=$(count_done)
    local total=$(count_total)
    local open=$(count_open)
    echo -e "  ${GREEN}${done} done${NC} / ${YELLOW}${open} open${NC} / ${total} total"
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
        osascript -e "display notification \"$new_stories\" with title \"XCAT Investigation\" subtitle \"$count story(ies) completed\"" 2>/dev/null || true
    fi
}

# Auto-commit on timeout
auto_commit_on_timeout() {
    cd "$PROJECT_DIR"
    local has_changes=false

    if [ -n "$(git status --porcelain ralph_loop/ 2>/dev/null)" ]; then
        has_changes=true
    fi
    if [ -n "$(git status --porcelain src/ 2>/dev/null)" ]; then
        has_changes=true
    fi

    if [ "$has_changes" = true ]; then
        echo -e "${YELLOW}  Auto-committing changes on timeout...${NC}"
        git add ralph_loop/xcat-progress.md ralph_loop/xcat-prd.json 2>/dev/null || true
        git add ralph_loop/scripts/ 2>/dev/null || true
        git commit -m "[TIMEOUT] Auto-save XCAT investigation progress" 2>/dev/null || true
        log_msg "TIMEOUT: Auto-committed changes"
    else
        log_msg "TIMEOUT: No changes to commit"
    fi
}

# ─── Pre-flight ──────────────────────────────────────────────────────────────

cd "$PROJECT_DIR"
check_files

echo ""
echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC}  ${BOLD}BasisSimulator — XCAT PCCT Artifact Investigation${NC}        ${BLUE}║${NC}"
echo -e "${BLUE}║${NC}  Max iterations: $MAX_ITERATIONS                                       ${BLUE}║${NC}"
echo -e "${BLUE}║${NC}  Target: Fix PCCT cupping + slice mismatch               ${BLUE}║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
print_status
echo ""
log_msg "XCAT investigation loop started. Max iterations: $MAX_ITERATIONS"

# ─── Main Loop ───────────────────────────────────────────────────────────────

iteration=0
while [ $iteration -lt $MAX_ITERATIONS ]; do
    iteration=$((iteration + 1))

    # Check if all stories are done
    open=$(count_open)
    if [ "$open" -eq 0 ]; then
        echo -e "${GREEN}All stories complete. Loop finished.${NC}"
        log_msg "COMPLETE: All stories done"
        exit 0
    fi

    echo -e "${YELLOW}── Iteration $iteration/$MAX_ITERATIONS ──${NC}"
    print_status
    log_msg "Iteration $iteration started. $open stories open."

    # Snapshot done stories BEFORE this iteration
    DONE_BEFORE="/tmp/ralph_xcat_done_before_$$_${iteration}"
    get_done_stories > "$DONE_BEFORE"

    # Extract highest-priority open unblocked story and inject into prompt
    NEXT_STORY=$(python3 -c "
import json, sys
with open('$PRD_FILE') as f:
    data = json.load(f)
done_ids = {s['id'] for s in data['stories'] if s.get('status') == 'done'}
candidates = []
for s in data['stories']:
    if s.get('status') != 'open':
        continue
    blocked_by = s.get('blockedBy', [])
    if all(b in done_ids for b in blocked_by):
        candidates.append(s)
candidates.sort(key=lambda x: (x.get('priority', 99), x['id']))
if candidates:
    print(json.dumps(candidates[0], indent=2))
" 2>/dev/null)

    if [ -n "$NEXT_STORY" ]; then
        STORY_ID=$(echo "$NEXT_STORY" | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
        STORY_TITLE=$(echo "$NEXT_STORY" | python3 -c "import json,sys; print(json.load(sys.stdin)['title'])")
        echo -e "  ${CYAN}Next story: $STORY_ID — $STORY_TITLE${NC}"
        log_msg "Selected story: $STORY_ID — $STORY_TITLE"
        STORY_BLOCK="## YOUR NEXT STORY (pre-selected)
\`\`\`json
$NEXT_STORY
\`\`\`

---

"
    else
        STORY_BLOCK=""
        echo -e "  ${YELLOW}No unblocked open story found${NC}"
        log_msg "WARNING: No unblocked story found"
    fi

    # Build full prompt: injected story + prompt template
    PROMPT_CONTENT="${STORY_BLOCK}$(cat "$PROMPT_FILE")"
    START_TIME=$(date +%s)
    TEMP_OUTPUT="/tmp/ralph_xcat_output_$$_${iteration}"

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
        echo -e "${GREEN}║  RALPH_COMPLETE — PCCT artifacts identified and fixed!       ║${NC}"
        echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
        log_msg "RALPH_COMPLETE after $iteration iterations"
        exit 0
    fi

    if echo "$OUTPUT" | grep -q "$BLOCKED_MARKER"; then
        echo ""
        echo -e "${RED}Agent is blocked. Check xcat-progress.md for details.${NC}"
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
