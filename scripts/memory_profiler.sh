#!/bin/bash
# ============================================================================
# Memory Profiler — Ralph 실행 중 메모리 스파이크 원인 분석기
# ============================================================================
# 5초마다 claude/ouroboros/node 프로세스 메모리를 기록
# 메모리 급증 패턴을 자동 감지하고 원인 프로세스를 식별
# ============================================================================

LOG="/tmp/memory_profile.log"
CSV="/tmp/memory_profile.csv"

echo "timestamp,total_procs,total_mb,claude_main_mb,subagent_count,subagent_mb,mcp_count,mcp_mb,top_process" > "$CSV"
echo "" > "$LOG"

log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG"; }

log "========================================="
log "Memory Profiler 시작 (5초 주기)"
log "========================================="

prev_total=0
spike_count=0
max_total=0

while true; do
    ts=$(date '+%H:%M:%S')

    # 메인 claude 프로세스
    claude_main=$(ps aux | grep "claude$" | grep -v grep | awk '{print $6/1024}' | head -1)
    claude_main=${claude_main:-0}

    # 서브에이전트 (claude --resume 또는 worktree 등)
    subagent_info=$(ps aux | grep "claude" | grep -v "claude$" | grep -v grep | awk '{n++; sum+=$6} END {printf "%d %.0f", n+0, sum/1024+0}')
    subagent_count=$(echo "$subagent_info" | awk '{print $1}')
    subagent_mb=$(echo "$subagent_info" | awk '{print $2}')

    # MCP 서버들 (node + python ouroboros)
    mcp_info=$(ps aux | grep -E "node.*mcp|ouroboros mcp serve|uv.*ouroboros" | grep -v grep | awk '{n++; sum+=$6} END {printf "%d %.0f", n+0, sum/1024+0}')
    mcp_count=$(echo "$mcp_info" | awk '{print $1}')
    mcp_mb=$(echo "$mcp_info" | awk '{print $2}')

    # 전체
    total_info=$(ps aux | grep -E "claude|ouroboros|node.*mcp" | grep -v grep | awk '{n++; sum+=$6} END {printf "%d %.0f", n+0, sum/1024+0}')
    total_procs=$(echo "$total_info" | awk '{print $1}')
    total_mb=$(echo "$total_info" | awk '{print $2}')

    # 가장 큰 프로세스
    top_proc=$(ps aux | grep -E "claude|ouroboros|node.*mcp" | grep -v grep | sort -k6 -rn | head -1 | awk '{printf "%s(PID=%s,%.0fMB)", $11, $2, $6/1024}')

    # CSV 기록
    echo "$ts,$total_procs,$total_mb,${claude_main%.*},$subagent_count,$subagent_mb,$mcp_count,$mcp_mb,$top_proc" >> "$CSV"

    # 스파이크 감지 (이전 대비 500MB 이상 증가)
    delta=$((${total_mb%.*} - ${prev_total%.*}))
    if [ "$delta" -gt 500 ] 2>/dev/null; then
        spike_count=$((spike_count + 1))
        log "⚠️  SPIKE #$spike_count: +${delta}MB (${prev_total%.*}→${total_mb%.*}MB)"
        log "    서브에이전트: ${subagent_count}개 (${subagent_mb}MB)"
        log "    MCP 서버: ${mcp_count}개 (${mcp_mb}MB)"
        log "    새 프로세스:"
        ps aux | grep -E "claude|ouroboros|node.*mcp" | grep -v grep | sort -k6 -rn | head -5 | awk '{printf "      PID=%-6s %6.0f MB  %s\n", $2, $6/1024, $11}' | tee -a "$LOG"
    fi

    # 최대치 갱신
    if [ "${total_mb%.*}" -gt "${max_total%.*}" ] 2>/dev/null; then
        max_total=$total_mb
    fi

    # 위험 수준 경고 (4GB 이상)
    if [ "${total_mb%.*}" -gt 4000 ] 2>/dev/null; then
        log "🔴 위험: ${total_mb}MB — 상세 프로세스 목록:"
        ps aux | grep -E "claude|ouroboros|node.*mcp" | grep -v grep | sort -k6 -rn | awk '{printf "   PID=%-6s %6.0f MB  CPU=%s%%  %s\n", $2, $6/1024, $3, $11}' | tee -a "$LOG"
    fi

    # 30초마다 요약 출력
    sec=$(date '+%S')
    if [ "$((sec % 30))" -lt 5 ]; then
        log "📊 ${total_procs} procs | ${total_mb%.*}MB | claude=${claude_main%.*}MB | sub=${subagent_count}x${subagent_mb}MB | mcp=${mcp_count}x${mcp_mb}MB | max=${max_total%.*}MB | spikes=${spike_count}"
    fi

    prev_total=$total_mb
    sleep 5
done
