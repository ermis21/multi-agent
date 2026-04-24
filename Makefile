.PHONY: up down build restart logs soul-update dream-run test-up test-down config-agent test eval test-full doctor e2e e2e-one test-dream test-dream-live e2e-dream

# ── Production stack ─────────────────────────────────────────────────────────

up:
	docker compose up -d

build:
	docker compose build --no-cache

down:
	docker compose down

restart:
	docker compose restart phoebe-api

logs:
	docker compose logs -f

logs-api:
	docker compose logs -f phoebe-api

logs-sandbox:
	docker compose logs -f phoebe-sandbox

status:
	docker compose ps

# ── Triggers ─────────────────────────────────────────────────────────────────

soul-update:
	@echo "Triggering manual soul update..."
	@curl -sf -X POST http://localhost:8090/internal/soul-update | python3 -m json.tool

# Manual dream run with live SSE stream + per-edit review.
#   DATE=YYYY-MM-DD  use a UTC calendar day instead of the default rolling 24h window
#   HOURS=N          rolling-window size in hours (default 24; ignored when DATE is set)
#   META=0           skip meta-dreamer
#   REVIEW=0         auto-commit (skip the per-edit review gate)
dream-run:
	@python3 scripts/dream_cli.py \
	  --url http://localhost:8090 \
	  $(if $(DATE),--date $(DATE),) \
	  $(if $(HOURS),--window-hours $(HOURS),) \
	  --meta $(if $(META),$(META),1) \
	  --review $(if $(REVIEW),$(REVIEW),1)

config-agent:
	@echo "Starting config agent... (press Enter on blank line to exit)"
	@SESSION="cli_config_$$(date +%s)"; \
	RESP=$$(curl -sf -X POST http://localhost:8090/config/agent \
	  -H 'Content-Type: application/json' \
	  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"start\"}],\"session_id\":\"$$SESSION\"}" \
	  | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null); \
	echo "$$RESP"; echo; \
	while true; do \
	  read -p "> " MSG; \
	  [ -z "$$MSG" ] && { echo "Exiting."; break; }; \
	  RESP=$$(curl -sf -X POST http://localhost:8090/config/agent \
	    -H 'Content-Type: application/json' \
	    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$$MSG\"}],\"session_id\":\"$$SESSION\"}" \
	    | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null); \
	  echo "$$RESP"; echo; \
	done

health:
	@curl -sf http://localhost:8090/health | python3 -m json.tool

# ── Test / staging stack ──────────────────────────────────────────────────────

test-up:
	docker compose -f docker-compose.test.yml up -d --build

test-down:
	docker compose -f docker-compose.test.yml down --remove-orphans

test-health:
	@curl -sf http://localhost:8091/health | python3 -m json.tool

# ── Tests & eval ─────────────────────────────────────────────────────────────

# Pure-unit tests: no stack dependencies. Runs inside phoebe-api so deps are available.
# Default 'test' target runs these — fast, always safe.
test-fast:
	docker exec phoebe-api pytest /app/test -m "not live"

# Integration tests: require the full stack at PHOEBE_URL (default http://localhost:8090).
test-integration:
	PHOEBE_LIVE=1 docker exec -e PHOEBE_LIVE=1 -e PHOEBE_URL=$${PHOEBE_URL:-http://localhost:8090} phoebe-api pytest /app/test -m live

# Default: fast tests only. Use `make test-integration` for live E2E coverage.
test: test-fast

eval:
	python test/eval.py

test-full: up test-fast test-integration eval

# Focused dream-system suite. Unit tests run inside phoebe-api (fast);
# live diagnostics run inside phoebe-sandbox (needs /project and /state mounts).
test-dream:
	docker exec phoebe-api pytest /app/test -m "not live" -k "dream or phrase_store or model_ranks"

test-dream-live:
	docker exec phoebe-sandbox pytest -m live /project/test/test_dream_diagnostics.py

# ── Discord end-to-end scenarios ─────────────────────────────────────────────
# Drives the worker bot from the config bot. Requires:
#   DISCORD_TEST_CHANNEL_ID, DISCORD_TEST_DRIVER_USER_ID, PHOEBE_ENABLE_TEXT_COMMANDS=1
# set in the phoebe-discord container's environment.

e2e:
	docker exec phoebe-discord python /app/e2e_scenarios.py

e2e-one:
	@if [ -z "$(SCENARIO)" ]; then echo "usage: make e2e-one SCENARIO=<name>"; exit 2; fi
	docker exec phoebe-discord python /app/e2e_scenarios.py --scenario $(SCENARIO)

e2e-dream:
	docker exec phoebe-discord python /app/e2e_scenarios.py --scenario dream_smoke

# ── Diagnostics ──────────────────────────────────────────────────────────────

doctor:
	@echo "Running system diagnostics..."
	@curl -sf http://localhost:8090/internal/diagnostics | python3 -c "\
import json,sys; d=json.load(sys.stdin); \
G='\033[32m'; Y='\033[33m'; R='\033[31m'; N='\033[0m'; \
col={'pass':G,'warn':Y,'fail':R}; \
sc=d.get('sandbox_checks',{}); ac=d.get('api_checks',{}); \
checks={**sc.get('checks',{}), **ac}; \
[print(f'  {col.get(c[\"status\"],\"\")}{c[\"status\"].upper():<6}{N} {name}' + (f' — {c[\"detail\"]}' if c.get(\"detail\") else '')) \
 for name,c in sorted(checks.items())]; \
o=d.get('overall','?'); print(); print(f'Overall: {col.get(o,\"\")}{o.upper()}{N}  |  {d.get(\"summary\",\"\")}')"

# ── Chat shortcut ─────────────────────────────────────────────────────────────

chat:
	@read -p "Message: " MSG; \
	curl -sf -X POST http://localhost:8090/v1/chat/completions \
	  -H 'Content-Type: application/json' \
	  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$$MSG\"}]}" | python3 -m json.tool
