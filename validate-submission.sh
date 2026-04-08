#!/usr/bin/env bash
#
# validate-submission.sh — OpenEnv Submission Validator (Optimized for Mac/Live Logs)
#
# Checks that your HF Space is live, Docker image builds, and openenv validate passes.
#

set -uo pipefail

# --- Configuration ---
DOCKER_BUILD_TIMEOUT=600 # 10 minutes

# Colors for pretty output
if [ -t 1 ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BOLD='\033[1m'
  NC='\033[0m'
else
  RED='' GREEN='' YELLOW='' BOLD='' NC=''
fi

# Helper: Cleanup temp files
CLEANUP_FILES=()
cleanup() { rm -f "${CLEANUP_FILES[@]+"${CLEANUP_FILES[@]}"}"; }
trap cleanup EXIT

portable_mktemp() {
  local prefix="${1:-validate}"
  mktemp "${TMPDIR:-/tmp}/${prefix}-XXXXXX" 2>/dev/null || mktemp
}

# --- Arguments ---
PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [ -z "$PING_URL" ]; then
  printf "Usage: %s <ping_url> [repo_dir]\n" "$0"
  printf "Example: ./validate-submission.sh https://your-space.hf.space\n"
  exit 1
fi

if ! REPO_DIR="$(cd "$REPO_DIR" 2>/dev/null && pwd)"; then
  printf "Error: directory '%s' not found\n" "${2:-.}"
  exit 1
fi

PING_URL="${PING_URL%/}" # Strip trailing slash
export PING_URL

# --- Logging Helpers ---
log()  { printf "[%s] %b\n" "$(date -u +%H:%M:%S)" "$*"; }
pass() { log "${GREEN}PASSED${NC} -- $1"; }
fail() { log "${RED}FAILED${NC} -- $1"; }
hint() { printf "  ${YELLOW}Hint:${NC} %b\n" "$1"; }
stop_at() {
  printf "\n"
  printf "${RED}${BOLD}Validation stopped at %s.${NC} Fix the above before continuing.\n" "$1"
  exit 1
}

# --- Header ---
printf "\n"
printf "${BOLD}========================================${NC}\n"
printf "${BOLD}  OpenEnv Submission Validator${NC}\n"
printf "${BOLD}========================================${NC}\n"
log "Repo:     $REPO_DIR"
log "Ping URL: $PING_URL"
printf "\n"

# --- Step 1: Pinging HF Space ---
log "${BOLD}Step 1/3: Pinging HF Space${NC} ($PING_URL/reset) ..."

CURL_OUTPUT=$(portable_mktemp "validate-curl")
CLEANUP_FILES+=("$CURL_OUTPUT")
HTTP_CODE=$(curl -s -o "$CURL_OUTPUT" -w "%{http_code}" -X POST \
  -H "Content-Type: application/json" -d '{}' \
  "$PING_URL/reset" --max-time 30 2>"$CURL_OUTPUT" || printf "000")

if [ "$HTTP_CODE" = "200" ]; then
  pass "HF Space is live and responds to /reset"
else
  fail "HF Space /reset returned HTTP $HTTP_CODE (expected 200)"
  hint "If you see 404, make sure you use the Direct URL (e.g., https://name.hf.space) not the Repo URL."
  stop_at "Step 1"
fi

# --- Step 2: Running Docker Build (LIVE OUTPUT) ---
log "${BOLD}Step 2/3: Running docker build${NC} (Streaming progress) ..."

if ! command -v docker &>/dev/null; then
  fail "docker command not found. Please install Docker Desktop."
  stop_at "Step 2"
fi

if [ -f "$REPO_DIR/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR"
else
  fail "No Dockerfile found in $REPO_DIR"
  stop_at "Step 2"
fi

log "  Found Dockerfile. Starting build..."
printf "${YELLOW}--- Docker Build Output Start ---${NC}\n"

# Build with live output to terminal
if docker build -t insurelink-v1-test "$DOCKER_CONTEXT"; then
  printf "${YELLOW}--- Docker Build Output End ---${NC}\n"
  pass "Docker build succeeded"
else
  printf "${YELLOW}--- Docker Build Output End ---${NC}\n"
  fail "Docker build failed"
  hint "Check the errors above in the Docker build log."
  stop_at "Step 2"
fi

# --- Step 3: Running OpenEnv Validate ---
log "${BOLD}Step 3/3: Running openenv validate${NC} ..."

if ! command -v openenv &>/dev/null; then
  fail "openenv command not found"
  hint "Try: pip install openenv-core"
  stop_at "Step 3"
fi

if (cd "$REPO_DIR" && openenv validate); then
  pass "openenv validate passed"
else
  fail "openenv validate failed"
  hint "Check your openenv.yaml and models.py structure."
  stop_at "Step 3"
fi

# --- Footer ---
printf "\n"
printf "${BOLD}========================================${NC}\n"
printf "${GREEN}${BOLD}  All 3/3 checks passed!${NC}\n"
printf "${GREEN}${BOLD}  Your submission is ready to submit.${NC}\n"
printf "${BOLD}========================================${NC}\n"
printf "\n"