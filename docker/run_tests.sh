#!/bin/bash
# =============================================================================
# SochDB Docker Test Runner
# =============================================================================
#
# Comprehensive test suite for SochDB gRPC server:
# 1. Build Docker image
# 2. Start server
# 3. Run integration tests
# 4. Run performance benchmarks
# 5. Generate report
#
# Usage: ./run_tests.sh [--skip-build] [--skip-benchmarks]
#
# Copyright 2025 Sushanth (https://github.com/sushanthpy)
# Licensed under the Apache License, Version 2.0

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
DOCKER_IMAGE="sochdb/sochdb-grpc:latest"
CONTAINER_NAME="sochdb-test"
GRPC_PORT=50051
SKIP_BUILD=false
SKIP_BENCHMARKS=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --skip-benchmarks)
            SKIP_BENCHMARKS=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--skip-build] [--skip-benchmarks]"
            exit 1
            ;;
    esac
done

# Helper functions
print_header() {
    echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}${1}${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

print_step() {
    echo -e "${BLUE}▶ ${1}${NC}"
}

print_success() {
    echo -e "${GREEN}✓ ${1}${NC}"
}

print_error() {
    echo -e "${RED}✗ ${1}${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ ${1}${NC}"
}

# Cleanup function
cleanup() {
    print_step "Cleaning up..."
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
}

# Trap cleanup on exit
trap cleanup EXIT

# Main test flow
main() {
    print_header "SochDB Docker Test Suite"
    
    # Step 1: Check Docker daemon
    print_step "Checking Docker daemon..."
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker daemon is not running"
        print_warning "Please start Docker Desktop and try again"
        exit 1
    fi
    print_success "Docker daemon is running"
    
    # Step 2: Build Docker image
    if [ "$SKIP_BUILD" = false ]; then
        print_step "Building Docker image..."
        if docker build -t $DOCKER_IMAGE -f Dockerfile .. 2>&1 | tee build.log; then
            print_success "Docker image built successfully"
        else
            print_error "Failed to build Docker image"
            print_warning "Check build.log for details"
            exit 1
        fi
    else
        print_warning "Skipping Docker build"
    fi
    
    # Step 3: Start Docker container
    print_step "Starting SochDB gRPC server..."
    
    # Clean up any existing container
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
    
    # Start container
    docker run -d \
        --name $CONTAINER_NAME \
        -p ${GRPC_PORT}:50051 \
        -e RUST_LOG=info \
        $DOCKER_IMAGE
    
    if [ $? -eq 0 ]; then
        print_success "Container started: $CONTAINER_NAME"
    else
        print_error "Failed to start container"
        exit 1
    fi
    
    # Wait for server to be ready
    print_step "Waiting for server to be ready..."
    MAX_RETRIES=30
    RETRY_COUNT=0
    
    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        if docker logs $CONTAINER_NAME 2>&1 | grep -q "Starting SochDB gRPC server"; then
            print_success "Server is ready"
            break
        fi
        
        RETRY_COUNT=$((RETRY_COUNT + 1))
        if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
            print_error "Server failed to start within timeout"
            echo "Container logs:"
            docker logs $CONTAINER_NAME
            exit 1
        fi
        
        sleep 1
    done
    
    # Show server info
    echo ""
    docker logs $CONTAINER_NAME 2>&1 | head -20
    
    # Step 4: Install Python dependencies
    print_step "Installing Python dependencies..."
    if python3 -m pip install --quiet numpy 2>&1 | tee pip_install.log; then
        print_success "Python dependencies installed"
    else
        print_warning "Failed to install some dependencies (continuing anyway)"
    fi
    
    # Step 5: Run integration tests
    print_step "Running integration tests..."
    if python3 test_integration.py --host localhost --port $GRPC_PORT; then
        print_success "Integration tests passed"
    else
        print_error "Integration tests failed"
        TEST_FAILED=true
    fi
    
    # Step 6: Run performance benchmarks
    if [ "$SKIP_BENCHMARKS" = false ]; then
        print_step "Running performance benchmarks..."
        if python3 test_performance.py --host localhost --port $GRPC_PORT; then
            print_success "Performance benchmarks completed"
        else
            print_error "Performance benchmarks failed"
            BENCH_FAILED=true
        fi
    else
        print_warning "Skipping performance benchmarks"
    fi
    
    # Step 7: Collect container stats
    print_step "Collecting container statistics..."
    docker stats --no-stream $CONTAINER_NAME > container_stats.txt
    cat container_stats.txt
    
    # Step 8: Show container logs
    print_header "Container Logs (last 50 lines)"
    docker logs --tail 50 $CONTAINER_NAME
    
    # Step 9: Generate summary report
    print_header "Test Summary"
    
    if [ -f integration_test_results.json ]; then
        echo -e "${GREEN}Integration Test Results:${NC}"
        cat integration_test_results.json | python3 -m json.tool || cat integration_test_results.json
        echo ""
    fi
    
    if [ -f performance_benchmark_results.json ]; then
        echo -e "${GREEN}Performance Benchmark Results:${NC}"
        cat performance_benchmark_results.json | python3 -m json.tool || cat performance_benchmark_results.json
        echo ""
    fi
    
    # Final status
    if [ "${TEST_FAILED}" = true ] || [ "${BENCH_FAILED}" = true ]; then
        print_error "Some tests failed"
        exit 1
    else
        print_success "All tests passed!"
        echo ""
        echo "Generated files:"
        echo "  - integration_test_results.json"
        echo "  - performance_benchmark_results.json"
        echo "  - container_stats.txt"
        echo "  - build.log"
        exit 0
    fi
}

# Run main
main
