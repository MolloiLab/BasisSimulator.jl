#!/bin/bash

# BasisSimulator.jl 10x Speed Build Loop
#
# Implements the 4 optimization stories from speed-spec.md.
# ALL work on branch 'speed/fused-projection'. Never touches main.
#
# Usage:
#   ./ralph_loops/speed-build-loop.sh          # Default 20 iterations
#   ./ralph_loops/speed-build-loop.sh 10       # Custom max iterations

set -e

# --- Configuration -----------------------------------------------------------

MAX_ITERATIONS="${1:-20}"
PROJECT_DIR="/Users/daleblack/Documents/dev/CTSimulatorStuff/BasisSimulator.jl"
RALPH_DIR="$PROJECT_DIR/ralph_loops"
PROMPT_FILE="$RALPH_DIR/speed-build-prompt.md"
PRD_FILE="$RALPH_DIR/speed-build-prd.json"
PROGRESS_FILE="$RALPH_DIR/speed-build-progress.md"
BRANCH="speed/fused-projection"

COMPLETE_MARKER="SPEED_BUILD_COMPLETE"
BLOCKED_MARKER="SPEED_BUILD_BLOCKED"
COOLDOWN=5
TIMEOUT=3600  # 60 minutes per iteration (building is slower than research)

# --- Colors -------------------------------------------------------------------

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

# --- Helpers ------------------------------------------------------------------

check_files() {
    for file in "$PROMPT_FILE" "$PRD_FILE" "$PROGRESS_FILE"; do
        if [ ! -f "$file" ]; then
            echo -e "${RED}Missing: $file${NC}"
            exit 1
        fi
    done
}

count_done() {
    grep -c '"done"' "$PRD_FILE" 2>/dev/null || echo "0"
}

count_open() {
    # v2: stories are created with status "ready" by the discovery loop
    grep -c '"ready"' "$PRD_FILE" 2>/dev/null || echo "0"
}

ensure_branch() {
    cd "$PROJECT_DIR"
    CURRENT=$(git branch --show-current)
    if [ "$CURRENT" != "$BRANCH" ]; then
        if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
            echo -e "  Switching to branch ${CYAN}$BRANCH${NC}"
            git checkout "$BRANCH"
        else
            echo -e "  Creating branch ${CYAN}$BRANCH${NC} from main"
            git checkout main
            git checkout -b "$BRANCH"
        fi
    fi
    echo -e "  Branch: ${GREEN}$(git branch --show-current)${NC}"
}

# --- Pre-flight ---------------------------------------------------------------

cd "$PROJECT_DIR"
check_files
ensure_branch

DONE=$(count_done)
OPEN=$(count_open)

echo ""
echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}  ${BOLD}BasisSimulator.jl — 10x Speed Build Loop (v2 GPU-First)${NC}     ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}  Branch: speed/fused-projection (main is SAFE)                ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}  GPU benchmarks required | Stories from discovery loop        ${CYAN}║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  Stories: ${GREEN}$DONE done${NC} / ${YELLOW}$OPEN open${NC}"
echo -e "  Max iterations: $MAX_ITERATIONS | Timeout: ${TIMEOUT}s"
echo ""

# --- Main Loop ----------------------------------------------------------------

iteration=0
while [ $iteration -lt $MAX_ITERATIONS ]; do
    iteration=$((iteration + 1))

    # Verify branch before every iteration
    cd "$PROJECT_DIR"
    CURRENT=$(git branch --show-current)
    if [ "$CURRENT" != "$BRANCH" ]; then
        echo -e "${RED}  SAFETY: Not on $BRANCH (on $CURRENT). Switching back.${NC}"
        git checkout "$BRANCH"
    fi

    DONE=$(count_done)
    OPEN=$(count_open)

    if [ "$OPEN" -eq 0 ]; then
        echo -e "${GREEN}══════════════════════════════════════════════════════════════${NC}"
        echo -e "${GREEN}  All stories complete! Branch: $BRANCH${NC}"
        echo -e "${GREEN}══════════════════════════════════════════════════════════════${NC}"
        exit 0
    fi

    echo -e "${YELLOW}══════════════════════════════════════════════════════════════${NC}"
    echo -e "  ${BOLD}Iteration $iteration / $MAX_ITERATIONS${NC}"
    echo -e "${YELLOW}══════════════════════════════════════════════════════════════${NC}"
    echo -e "  Stories: ${GREEN}$DONE done${NC} / ${YELLOW}$OPEN open${NC}"
    echo -e "  Branch: $(git branch --show-current)"
    echo ""

    PROMPT_CONTENT=$(cat "$PROMPT_FILE")
    START_TIME=$(date +%s)
    TEMP_OUTPUT="/tmp/speed_build_ralph_$$"

    echo -e "  Launching build agent..."
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
            cd "$PROJECT_DIR"
            # Verify still on branch before committing
            if [ "$(git branch --show-current)" = "$BRANCH" ]; then
                if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
                    echo -e "  Saving uncommitted work on $BRANCH..."
                    git add -A
                    git commit -m "TIMEOUT-SAVE: Speed build iteration $iteration" 2>/dev/null || true
                fi
            else
                echo -e "${RED}  WARNING: Not on $BRANCH after timeout. NOT saving.${NC}"
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
    echo -e "--- Agent Output (last 20 lines) ---"
    echo "$OUTPUT" | tail -20
    echo -e "--- End Agent Output ---"
    echo ""

    # Verify branch after agent ran
    cd "$PROJECT_DIR"
    CURRENT=$(git branch --show-current)
    if [ "$CURRENT" != "$BRANCH" ]; then
        echo -e "${RED}  ⚠ Agent switched to $CURRENT! Switching back to $BRANCH.${NC}"
        git checkout "$BRANCH"
    fi

    DONE=$(count_done)
    OPEN=$(count_open)
    echo -e "  Stories: ${GREEN}$DONE done${NC} / ${YELLOW}$OPEN open${NC}"

    if echo "$OUTPUT" | grep -q "$COMPLETE_MARKER"; then
        echo ""
        echo -e "${GREEN}══════════════════════════════════════════════════════════════${NC}"
        echo -e "${GREEN}  All stories complete! Branch: $BRANCH${NC}"
        echo -e "${GREEN}  Ready for review and merge to main.${NC}"
        echo -e "${GREEN}══════════════════════════════════════════════════════════════${NC}"
        exit 0
    fi

    if echo "$OUTPUT" | grep -q "$BLOCKED_MARKER"; then
        echo -e "${RED}  Build blocked. Check speed-build-progress.md.${NC}"
        exit 1
    fi

    if [ $iteration -lt $MAX_ITERATIONS ]; then
        echo -e "  Cooldown ${COOLDOWN}s..."
        sleep $COOLDOWN
    fi
done

echo ""
echo -e "${YELLOW}Max iterations ($MAX_ITERATIONS) reached.${NC}"
echo -e "  ${GREEN}$DONE done${NC} / ${YELLOW}$OPEN open${NC}"
echo -e "  Branch: $BRANCH"
exit 0
