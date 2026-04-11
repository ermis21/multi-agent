.PHONY: up down build restart logs soul-update test-up test-down config-agent

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
	@echo "Starting interactive config agent (type your message)..."
	@read -p "Message: " MSG; \
	curl -sf -X POST http://localhost:8090/config/agent \
	  -H 'Content-Type: application/json' \
	  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$$MSG\"}]}" | python3 -m json.tool

health:
	@curl -sf http://localhost:8090/health | python3 -m json.tool

# ── Test / staging stack ──────────────────────────────────────────────────────

test-up:
	docker compose -f docker-compose.test.yml up -d --build

test-down:
	docker compose -f docker-compose.test.yml down --remove-orphans

test-health:
	@curl -sf http://localhost:8091/health | python3 -m json.tool

# ── Chat shortcut ─────────────────────────────────────────────────────────────

chat:
	@read -p "Message: " MSG; \
	curl -sf -X POST http://localhost:8090/v1/chat/completions \
	  -H 'Content-Type: application/json' \
	  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$$MSG\"}]}" | python3 -m json.tool
