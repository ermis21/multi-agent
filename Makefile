.PHONY: up down build restart logs soul-update test-up test-down config-agent test eval test-full doctor

# ── Production stack ─────────────────────────────────────────────────────────

up:
	docker compose up -d

build:
	docker compose build --no-cache

down:
	docker compose down

restart:
	docker compose restart mab-api

logs:
	docker compose logs -f

logs-api:
	docker compose logs -f mab-api

logs-sandbox:
	docker compose logs -f mab-sandbox

status:
	docker compose ps

# ── Triggers ─────────────────────────────────────────────────────────────────

soul-update:
	@echo "Triggering manual soul update..."
	@curl -sf -X POST http://localhost:8090/internal/soul-update | python3 -m json.tool

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

test:
	pytest test/ -v

eval:
	python test/eval.py

test-full: up test eval

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
