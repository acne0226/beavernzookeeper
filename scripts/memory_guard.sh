#!/bin/bash
# ============================================================================
# Memory Guard — Ouroboros 고아 프로세스 자동 정리 스크립트
# ============================================================================
# 용도: evolve_step 실행 중 stall/retry로 인한 고아 프로세스를 감지하고 kill
# 사용법: ./scripts/memory_guard.sh &
# 중단: kill $(cat /tmp/memory_guard.pid) 또는 Ctrl+C
# ============================================================================

set -euo pipefail

# ── 설정 ────────────────────────────────────────────────────────────────────
MEMORY_LIMIT_MB=8000          # 8GB 넘으면 정리 시작
CHECK_INTERVAL=30             # 30초마다 체크
LOG_FILE="/tmp/memory_guard.log"
PID_FILE="/tmp/memory_guard.pid"

# 보호할 프로세스 (이것만 남기고 나머지 claude/node 서브프로세스 kill)
# 메인 claude PID는 실행 시 자동 감지
MAIN_CLAUDE_PID=""
OUROBOROS_MCP_PID=""
GOOGLE_SHEETS_PID=""

# ── 함수 ────────────────────────────────────────────────────────────────────

log() {
    local msg="[$(date '+%H:%M:%S')] $1"
    echo "$msg" | tee -a "$LOG_FILE"
}

detect_protected_pids() {
    # 메인 claude 프로세스 (터미널에서 실행된 것)
    MAIN_CLAUDE_PID=$(ps aux | grep "claude$" | grep -v grep | awk '{print $2}' | head -1)

    # ouroboros mcp serve
    OUROBOROS_MCP_PID=$(ps aux | grep "ouroboros mcp serve" | grep -v grep | awk '{print $2}' | head -1)

    # uv launcher for ouroboros
    UV_LAUNCHER_PID=$(ps aux | grep "uv tool uvx.*ouroboros" | grep -v grep | awk '{print $2}' | head -1)

    # mcp-google-sheets
    GOOGLE_SHEETS_PID=$(ps aux | grep "mcp-google-sheets" | grep -v grep | awk '{print $2}' | head -1)

    log "보호 PID: claude=$MAIN_CLAUDE_PID, ouroboros=$OUROBOROS_MCP_PID, uv=$UV_LAUNCHER_PID, sheets=$GOOGLE_SHEETS_PID"
}

get_total_memory_mb() {
    ps aux | grep -E "claude|ouroboros|node.*mcp" | grep -v grep | \
        awk '{sum+=$6} END {printf "%.0f", sum/1024}'
}

get_subprocess_count() {
    ps aux | grep -E "claude.*--resume|claude.*-p " | grep -v grep | wc -l | tr -d ' '
}

kill_orphan_subprocesses() {
    local killed=0

    # claude 서브프로세스 찾기 (--resume 또는 -p 플래그가 있는 것들 = 서브에이전트)
    local sub_pids=$(ps aux | grep -E "claude.*--resume|claude.*-p " | grep -v grep | awk '{print $2}')

    for pid in $sub_pids; do
        # 보호 PID가 아니면 kill
        if [ "$pid" != "$MAIN_CLAUDE_PID" ]; then
            log "  KILL 서브에이전트 PID=$pid ($(ps -o rss= -p "$pid" 2>/dev/null | awk '{printf "%.0f MB", $1/1024}'))"
            kill -TERM "$pid" 2>/dev/null || true
            killed=$((killed + 1))
        fi
    done

    # 고아 node 프로세스 (MCP 서버 중복분) 찾기
    local node_pids=$(ps aux | grep "node.*mcp\|node.*npx" | grep -v grep | awk '{print $2}')

    for pid in $node_pids; do
        if [ "$pid" != "$GOOGLE_SHEETS_PID" ]; then
            log "  KILL 고아 MCP node PID=$pid ($(ps -o rss= -p "$pid" 2>/dev/null | awk '{printf "%.0f MB", $1/1024}'))"
            kill -TERM "$pid" 2>/dev/null || true
            killed=$((killed + 1))
        fi
    done

    # 고아 python ouroboros 프로세스 찾기
    local py_pids=$(ps aux | grep "ouroboros" | grep -v grep | awk '{print $2}')

    for pid in $py_pids; do
        if [ "$pid" != "$OUROBOROS_MCP_PID" ] && [ "$pid" != "$UV_LAUNCHER_PID" ]; then
            log "  KILL 고아 ouroboros PID=$pid"
            kill -TERM "$pid" 2>/dev/null || true
            killed=$((killed + 1))
        fi
    done

    echo $killed
}

# ── 메인 루프 ───────────────────────────────────────────────────────────────

echo $$ > "$PID_FILE"
log "======================================"
log "Memory Guard 시작 (한도: ${MEMORY_LIMIT_MB}MB, 주기: ${CHECK_INTERVAL}초)"
log "======================================"

detect_protected_pids

trap 'log "Memory Guard 종료"; rm -f "$PID_FILE"; exit 0' INT TERM

cycle=0
while true; do
    cycle=$((cycle + 1))
    total_mb=$(get_total_memory_mb)
    sub_count=$(get_subprocess_count)

    # 매 10회(5분)마다 보호 PID 재감지 (프로세스가 재시작되었을 수 있음)
    if [ $((cycle % 10)) -eq 0 ]; then
        detect_protected_pids
    fi

    if [ "$total_mb" -gt "$MEMORY_LIMIT_MB" ]; then
        log "⚠️  메모리 경고: ${total_mb}MB / ${MEMORY_LIMIT_MB}MB 한도 초과! (서브프로세스: ${sub_count}개)"
        killed=$(kill_orphan_subprocesses)

        if [ "$killed" -gt 0 ]; then
            sleep 3
            new_total=$(get_total_memory_mb)
            log "✅ ${killed}개 프로세스 정리 완료. 메모리: ${total_mb}MB → ${new_total}MB"
        else
            log "  정리할 고아 프로세스 없음. 메인 프로세스 자체가 큰 상태."
        fi
    else
        # 5분(10회)마다만 정상 로그 출력
        if [ $((cycle % 10)) -eq 0 ]; then
            log "✓ 정상: ${total_mb}MB (서브프로세스: ${sub_count}개)"
        fi
    fi

    sleep "$CHECK_INTERVAL"
done
