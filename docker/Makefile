# =============================================================================
# SochDB Docker Makefile
# =============================================================================
#
# Usage: make <target>
#
# Copyright 2025 Sushanth (https://github.com/sushanthpy)
# Licensed under the Apache License, Version 2.0

.PHONY: all build build-slim run stop logs clean test health dev web monitoring prod

# Default target
all: build run

# =============================================================================
# Build Targets
# =============================================================================

## Build the standard Docker image
build:
	docker build -t sochdb/sochdb-grpc:latest -f Dockerfile ..

## Build the slim Alpine image
build-slim:
	docker build -f Dockerfile.slim -t sochdb/sochdb-grpc:slim ..

## Build both images
build-all: build build-slim

# =============================================================================
# Run Targets
# =============================================================================

## Start SochDB with Docker Compose
run:
	docker compose up -d

## Start in development mode
dev:
	docker compose --profile dev up -d

## Start with gRPC-Web support
web:
	docker compose --profile web up -d

## Start with monitoring (Prometheus + Grafana)
monitoring:
	docker compose --profile monitoring up -d

## Start everything
full:
	docker compose --profile dev --profile web --profile monitoring up -d

## Start production stack
prod:
	docker compose -f docker-compose.production.yml up -d

## Stop all services
stop:
	docker compose down

## Stop production stack
stop-prod:
	docker compose -f docker-compose.production.yml down

# =============================================================================
# Development
# =============================================================================

## View logs
logs:
	docker compose logs -f sochdb

## View all logs
logs-all:
	docker compose logs -f

## Shell into container
shell:
	docker exec -it sochdb-grpc /bin/sh

## Health check
health:
	@grpc_health_probe -addr=localhost:50051 && echo "✅ SochDB is healthy" || echo "❌ SochDB is unhealthy"

## Test gRPC connection
test:
	@echo "Testing gRPC connection..."
	@grpc_health_probe -addr=localhost:50051

# =============================================================================
# Cleanup
# =============================================================================

## Remove containers and networks
clean:
	docker compose down -v --remove-orphans

## Remove all images
clean-images:
	docker rmi sochdb/sochdb-grpc:latest sochdb/sochdb-grpc:slim 2>/dev/null || true

## Full cleanup
clean-all: clean clean-images

# =============================================================================
# Registry
# =============================================================================

## Push images to Docker Hub
push:
	docker push sochdb/sochdb-grpc:latest
	docker push sochdb/sochdb-grpc:slim

## Tag and push versioned images
release:
	@read -p "Enter version (e.g., 0.4.5): " version; \
	docker tag sochdb/sochdb-grpc:latest sochdb/sochdb-grpc:$$version; \
	docker tag sochdb/sochdb-grpc:slim sochdb/sochdb-grpc:$$version-slim; \
	docker push sochdb/sochdb-grpc:$$version; \
	docker push sochdb/sochdb-grpc:$$version-slim; \
	docker push sochdb/sochdb-grpc:latest; \
	docker push sochdb/sochdb-grpc:slim

# =============================================================================
# Utilities
# =============================================================================

## Show container status
status:
	docker compose ps

## Show container stats
stats:
	docker stats sochdb-grpc

## Backup data volume
backup:
	docker run --rm \
		-v sochdb_docker_sochdb-data:/data:ro \
		-v $(PWD):/backup \
		alpine tar czf /backup/sochdb-backup-$(shell date +%Y%m%d-%H%M%S).tar.gz -C /data .
	@echo "Backup created: sochdb-backup-$(shell date +%Y%m%d-%H%M%S).tar.gz"

## Restore from backup
restore:
	@read -p "Enter backup file: " backup; \
	docker run --rm \
		-v sochdb_docker_sochdb-data:/data \
		-v $(PWD):/backup:ro \
		alpine tar xzf /backup/$$backup -C /data

# =============================================================================
# Help
# =============================================================================

## Show this help
help:
	@echo "SochDB Docker Management"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@grep -E '^## ' Makefile | sed 's/## /  /'
	@echo ""
	@echo "Common workflows:"
	@echo "  make build run      # Build and start SochDB"
	@echo "  make monitoring     # Start with Prometheus + Grafana"
	@echo "  make logs           # View container logs"
	@echo "  make clean          # Stop and remove containers"
