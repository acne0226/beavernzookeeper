# Beavernzoopeeker — 랄프톤 서울 중간발표

## Slide 1: 표지
- **프로젝트명**: Beavernzoopeeker
- **팀명**: 비버와 사육사
- **팀원**: 안혜원, 김명준
- **GitHub**: github.com/acne0226/beavernzookeeper

## Slide 2: 문제 정의
- **대상**: VC 심사역 (투자 담당자)
- **문제**: 매일 5~8건 외부 미팅, 30~50통 메일, 50~100개 Notion 딜 레코드를 직접 핸드패킹해야 하는 고통
- **빈도**: 매일, 모든 미팅마다 반복
- **고통**: 미팅 준비에 들어가는 시간이 미팅 자체보다 길 수 있음. 포트폴리오사 메일 데드라인을 놓치면 신뢰 손실.
- **핵심 메시지**: "정답지를 만들려면 엣지 케이스를 일일이 5개씩 핸드패킹해야 한다 — 이걸 AI 혼자 판단하게 하기엔 신뢰도가 부족하다"

## Slide 3: 솔루션 — Beavernzoopeeker
- **한 문장**: 심사역의 미팅 준비, 메일 관리, 업무 질의응답을 자동화하는 개인 AI 비서 데몬
- **핵심 흐름**:
  1. Google Calendar에서 외부 미팅 15분 전 자동 감지
  2. Gmail + Notion + Slack + 웹 검색을 통합해 브리핑 자동 생성
  3. Claude API가 참석자 프로필, 아젠다, 이메일 요약을 작성
  4. Slack DM으로 즉시 전달
- **추가 기능**: `/brief`, `/mail`, `/ask` 슬래시 커맨드로 온디맨드 사용

## Slide 4: 나의 랄프 세팅 / 랄프 역량
- **AI 에이전트**: Claude Code + Ouroboros (Seed v2 기반 Builder-Breaker 검증 루프)
- **Builder-Breaker 방식**: AI 두 개가 경찰과 도둑처럼 대립 — Builder가 구현하면 Breaker가 엣지 케이스로 공격, 무승부가 나올 때까지 반복
- **안전장치**: 최대 20사이클 제한, 진동 감지(같은 에러 3회 반복 시 자동 종료), 에러 심각도 분류(critical vs minor)
- **운영 도구**: memory_guard.sh (고아 프로세스 자동 정리, 8GB 메모리 한도 모니터링)
- **지속 전략**: Specification-first — 코드 전에 Seed로 요구사항을 정의하고, 평가 원칙(정확성 1.0, 데드라인 무결 1.0)으로 품질 기준을 설정

## Slide 5: 현재 진행 상황
- **Feature 1 (미팅 브리핑)**: 구현 완료 — 매일 09:30 전체 브리핑 + 외부 미팅 15분 전 개별 브리핑
- **Feature 2 (메일 관리)**: 구현 완료 — 5분 주기 포트폴리오 메일 스캔, 데드라인/오버듀/미회신 알림
- **Feature 3 (자연어 Q&A)**: 구현 완료 — `/ask`로 캘린더+메일+Notion+Slack 통합 질의응답 (15초 이내)
- **테스트**: 36개 모듈
- **아키텍처**: 모듈별 분리 (calendar / gmail / notion / slack / ai / briefing), 에러 복원력, graceful degradation
