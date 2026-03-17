#!/bin/bash

# BasisSimulator.jl 10x Speed Discovery Loop
#
# Research-only loop. Produces knowledge artifacts, not code.
# Each iteration: fresh agent reads state → researches/critiques/refines → commits → exits.
# Output: speed-spec.md (the optimization roadmap).
#
# Usage:
#   ./ralph_loops/speed-loop.sh          # Default 30 iterations
#   ./ralph_loops/speed-loop.sh 15       # Custom max iterations

set -e

# --- Configuration -----------------------------------------------------------

MAX_ITERATIONS="${1:-30}"
PROJECT_DIR="/Users/daleblack/Documents/dev/CTSimulatorStuff/BasisSimulator.jl"
RALPH_DIR="$PROJECT_DIR/ralph_loops"
PROMPT_FILE="$RALPH_DIR/speed-prompt.md"
STATE_FILE="$RALPH_DIR/speed-state.md"
SPEC_FILE="$RALPH_DIR/speed-spec.md"
PROGRESS_FILE="$RALPH_DIR/speed-progress.md"
PRD_FILE="$RALPH_DIR/speed-prd.json"

COMPLETE_MARKER="SPEED_COMPLETE"
BLOCKED_MARKER="SPEED_BLOCKED"
COOLDOWN=5
TIMEOUT=1800

# --- Colors -------------------------------------------------------------------

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

# --- Helpers ------------------------------------------------------------------

check_files() {
    for file in "$PROMPT_FILE" "$STATE_FILE" "$SPEC_FILE" "$PROGRESS_FILE" "$PRD_FILE"; do
        if [ ! -f "$file" ]; then
            echo -e "${RED}Missing: $file${NC}"
            exit 1
        fi
    done
}

count_phases() {
    local status="$1"
    grep -c "\"$status\"" "$PRD_FILE" 2>/dev/null || echo "0"
}

get_current_phase() {
    grep "Current phase:" "$STATE_FILE" 2>/dev/null | head -1 | sed 's/.*: //' || echo "unknown"
}

# --- Pre-flight ---------------------------------------------------------------

cd "$PROJECT_DIR"
check_files

OPEN=$(count_phases "open")
DONE=$(count_phases "done")
PHASE=$(get_current_phase)

echo ""
echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}  ${BOLD}BasisSimulator.jl — 10x Speed Discovery Loop${NC}               ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}  Profile → Research → Critique → Optimize Spec               ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}  Goal: 10x speedup with identical physics                    ${CYAN}║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  Max iterations: $MAX_ITERATIONS | Timeout: ${TIMEOUT}s | Cooldown: ${COOLDOWN}s"
echo -e "  Phase status: ${GREEN}$DONE done${NC} / ${YELLOW}$OPEN open${NC}"
echo -e "  Current: $PHASE"
echo ""

# --- Main Loop ----------------------------------------------------------------

iteration=0
while [ $iteration -lt $MAX_ITERATIONS ]; do
    iteration=$((iteration + 1))

    OPEN=$(count_phases "open")
    DONE=$(count_phases "done")
    PHASE=$(get_current_phase)

    if [ "$OPEN" -eq 0 ]; then
        echo -e "${GREEN}══════════════════════════════════════════════════════════════${NC}"
        echo -e "${GREEN}  All research complete! Speed spec is ready at:${NC}"
        echo -e "${GREEN}  $SPEC_FILE${NC}"
        echo -e "${GREEN}══════════════════════════════════════════════════════════════${NC}"
        exit 0
    fi

    echo -e "${YELLOW}══════════════════════════════════════════════════════════════${NC}"
    echo -e "  ${BOLD}Iteration $iteration / $MAX_ITERATIONS${NC}"
    echo -e "${YELLOW}══════════════════════════════════════════════════════════════${NC}"
    echo -e "  Phase: $PHASE"
    echo -e "  Status: ${GREEN}$DONE done${NC} / ${YELLOW}$OPEN open${NC}"
    echo ""

    PROMPT_CONTENT=$(cat "$PROMPT_FILE")
    START_TIME=$(date +%s)
    TEMP_OUTPUT="/tmp/speed_discovery_ralph_$$"

    echo -e "  Launching research agent..."
    claude --print --dangerously-skip-permissions "$PROMPT_CONTENT" > "$TEMP_OUTPUT" 2>&1 &
    CLAUDE_PID=$!

    while ps -p $CLAUDE_PID > /dev/null 2>&1; do
        ELAPSED=$(($(date +%s) - START_TIME))
        printf "\r  Working... %dm%02ds  " $((ELAPSED/60)) $((ELAPSED%60))
        if [ $ELAPSED -ge $TIMEOUT ]; then
            echo -e "\n  ${RED}Timeout after ${TIMEOUT}s. Killing agent.${NC}"
            kill $CLAUDE_PID 2>/dev/null || true
            wait $CLAUDE_PID 2>/dev/null || true
            echo -e "--- TIMEOUT SAVE (iteration $iteration) ---"
            if [ -n "$(git status --porcelain ralph_loops/speed-* 2>/dev/null)" ]; then
                echo -e "  Saving uncommitted research..."
                git add ralph_loops/speed-*.md ralph_loops/speed-*.json 2>/dev/null || true
                git commit -m "TIMEOUT-SAVE: Speed discovery iteration $iteration" 2>/dev/null || true
            fi
            echo -e "--- END TIMEOUT SAVE ---"
            break
        fi
        sleep 3
    done

    DURATION=$(($(date +%s) - START_TIME))
    OUTPUT=$(cat "$TEMP_OUTPUT" 2>/dev/null || echo "")
    rm -f "$TEMP_OUTPUT"

    echo -e "\r  Done in ${DURATION}s                    "
    echo ""
    echo -e "--- Agent Output (last 15 lines) ---"
    echo "$OUTPUT" | tail -15
    echo -e "--- End Agent Output ---"
    echo ""

    DONE=$(count_phases "done")
    OPEN=$(count_phases "open")
    echo -e "  Status: ${GREEN}$DONE done${NC} / ${YELLOW}$OPEN open${NC}"

    if echo "$OUTPUT" | grep -q "$COMPLETE_MARKER"; then
        echo ""
        echo -e "${GREEN}══════════════════════════════════════════════════════════════${NC}"
        echo -e "${GREEN}  All research complete! Speed spec is ready at:${NC}"
        echo -e "${GREEN}  $SPEC_FILE${NC}"
        echo -e "${GREEN}══════════════════════════════════════════════════════════════${NC}"
        exit 0
    fi

    if echo "$OUTPUT" | grep -q "$BLOCKED_MARKER"; then
        echo ""
        echo -e "${RED}  Research blocked. Check speed-state.md for details.${NC}"
        exit 1
    fi

    if [ $iteration -lt $MAX_ITERATIONS ]; then
        echo -e "  Cooldown ${COOLDOWN}s..."
        sleep $COOLDOWN
    fi
done

echo ""
echo -e "${YELLOW}Max iterations ($MAX_ITERATIONS) reached. Run again to continue.${NC}"
echo -e "  ${GREEN}$DONE done${NC} / ${YELLOW}$OPEN open${NC}"
echo -e "  Spec: $SPEC_FILE"
exit 0
