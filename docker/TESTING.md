# SochDB Docker Testing Guide

Comprehensive testing documentation for the SochDB gRPC server Docker deployment.

## 📋 Test Suite Overview

The test suite includes:

1. **Integration Tests** (`test_integration.py`)
   - Server connectivity
   - Key-value operations (PUT, GET, DELETE)
   - Vector index operations
   - Graph operations
   - Namespace isolation
   - Health checks

2. **Performance Benchmarks** (`test_performance.py`)
   - KV write throughput
   - KV read throughput
   - Concurrent client testing
   - Mixed workload benchmarks
   - Latency percentiles (p50, p95, p99)

3. **Automated Test Runner** (`run_tests.sh`)
   - Build Docker image
   - Start server
   - Run all tests
   - Generate reports
   - Clean up resources

## 🚀 Quick Start

### Prerequisites

```bash
# Install Python dependencies
pip install numpy grpcio grpcio-tools

# Install SochDB Python SDK
cd ../sochdb-python-sdk
pip install -e .
```

### Run All Tests

```bash
# Full test suite (build + integration + performance)
./run_tests.sh

# Skip Docker build (use existing image)
./run_tests.sh --skip-build

# Skip performance benchmarks (faster)
./run_tests.sh --skip-benchmarks
```

## 📊 Test Execution

### 1. Integration Tests Only

```bash
# Start server first
docker compose up -d

# Run integration tests
python3 test_integration.py --host localhost --port 50051

# Check results
cat integration_test_results.json
```

### 2. Performance Benchmarks Only

```bash
# Start server first
docker compose up -d

# Run benchmarks
python3 test_performance.py --host localhost --port 50051

# Check results
cat performance_benchmark_results.json
```

### 3. Manual Docker Testing

```bash
# Build image
docker build -t sochdb/sochdb-grpc:latest -f Dockerfile ..

# Run container
docker run -d \
  --name sochdb-test \
  -p 50051:50051 \
  -e RUST_LOG=debug \
  sochdb/sochdb-grpc:latest

# View logs
docker logs -f sochdb-test

# Check health
grpc_health_probe -addr=localhost:50051

# Stop container
docker stop sochdb-test
docker rm sochdb-test
```

## 📈 Performance Benchmarks

### Expected Performance (Baseline)

| Metric | Expected Value | Notes |
|--------|---------------|-------|
| KV Write Throughput | > 50,000 ops/sec | Single client |
| KV Read Throughput | > 100,000 ops/sec | Single client |
| Concurrent Write | > 200,000 ops/sec | 10 threads |
| Search Latency (p50) | < 5 ms | 128-dim vectors |
| Search Latency (p99) | < 20 ms | 128-dim vectors |

### Benchmark Configuration

```python
# Default settings
KV_OPERATIONS = 10_000
CONCURRENT_THREADS = 10
OPS_PER_THREAD = 1_000
VECTOR_DIMENSION = 128
NUM_VECTORS = 1_000
```

### Custom Benchmarks

```python
from test_performance import PerformanceBenchmark

benchmark = PerformanceBenchmark(host="localhost", port=50051)

# Custom KV benchmark
result = benchmark.benchmark_kv_writes(num_ops=50_000)
print(f"Throughput: {result.throughput_ops_sec:,.0f} ops/sec")

# Custom concurrent benchmark
result = benchmark.benchmark_concurrent_writes(
    num_threads=20,
    ops_per_thread=5_000
)
```

## 🔍 Test Results

### Integration Test Output

```
╔════════════════════════════════════════════════════════════════════════════╗
║         SochDB gRPC Server Integration Test Suite                         ║
╚════════════════════════════════════════════════════════════════════════════╝

════════════════════════════════════════
           CONNECTION TEST
════════════════════════════════════════

  ✓ PASS Connect to gRPC server (12.34ms)
  ► Server address: localhost:50051

════════════════════════════════════════
        KEY-VALUE OPERATIONS
════════════════════════════════════════

  ✓ PASS PUT operation (2.15ms)
  ✓ PASS GET operation (1.87ms)
  ✓ PASS DELETE operation (2.03ms)
  ✓ PASS Verify deletion (1.65ms)
```

### Performance Benchmark Output

```
📊 Benchmarking KV Writes (10000 operations)...

  Results for KV Writes:
    Operations:      10,000
    Duration:        0.85 sec
    Throughput:      11,765 ops/sec
    Latency (mean):  0.085 ms
    Latency (p50):   0.078 ms
    Latency (p95):   0.142 ms
    Latency (p99):   0.231 ms
    Latency (max):   2.456 ms
    Success rate:    100.0%
```

### Result Files

| File | Content |
|------|---------|
| `integration_test_results.json` | Integration test results with timestamps |
| `performance_benchmark_results.json` | Performance metrics and latencies |
| `container_stats.txt` | Docker container resource usage |
| `build.log` | Docker build output (if applicable) |

## 🐛 Troubleshooting

### Docker Daemon Not Running

```bash
# Error: Cannot connect to Docker daemon
# Solution: Start Docker Desktop
open -a Docker
```

### Port Already in Use

```bash
# Error: Port 50051 already in use
# Solution: Stop existing container
docker ps | grep 50051
docker stop <container_id>

# Or use different port
docker run -p 50052:50051 sochdb/sochdb-grpc:latest
python3 test_integration.py --port 50052
```

### Server Not Ready

```bash
# Check container status
docker ps

# View logs
docker logs sochdb-test

# Check if server started
docker logs sochdb-test | grep "Starting SochDB"
```

### Test Failures

```bash
# Enable debug logging
docker run -e RUST_LOG=debug sochdb/sochdb-grpc:latest

# Run tests with verbose output
python3 -u test_integration.py

# Check gRPC connectivity
grpc_health_probe -addr=localhost:50051 -v
```

### Memory Issues

```bash
# Increase Docker memory limit
docker run --memory=4g --memory-swap=4g sochdb/sochdb-grpc:latest

# Monitor container resources
docker stats sochdb-test
```

## 📝 Writing Custom Tests

### Custom Integration Test

```python
from test_integration import IntegrationTest

class CustomTest(IntegrationTest):
    def test_custom_feature(self) -> bool:
        """Test custom feature"""
        print_header("CUSTOM FEATURE TEST")
        
        start = time.time()
        try:
            # Your test logic here
            result = self.client.custom_operation()
            duration = (time.time() - start) * 1000
            print_test("Custom operation", True, duration)
            return True
        except Exception as e:
            duration = (time.time() - start) * 1000
            print_test("Custom operation", False, duration)
            print(f"    Error: {e}")
            return False

# Run custom test
test = CustomTest(host="localhost", port=50051)
test.connect()
test.test_custom_feature()
```

### Custom Benchmark

```python
from test_performance import PerformanceBenchmark, BenchmarkResult

class CustomBenchmark(PerformanceBenchmark):
    def benchmark_custom_operation(self, num_ops: int = 1000) -> BenchmarkResult:
        """Benchmark custom operation"""
        client = self._create_client()
        latencies = []
        
        overall_start = time.time()
        
        for i in range(num_ops):
            start = time.time()
            # Your benchmark logic here
            client.custom_operation()
            latencies.append((time.time() - start) * 1000)
        
        overall_duration = time.time() - overall_start
        
        # Calculate statistics
        latencies.sort()
        result = BenchmarkResult(
            name="Custom Operation",
            operations=num_ops,
            duration_sec=overall_duration,
            throughput_ops_sec=num_ops / overall_duration,
            latency_mean_ms=sum(latencies) / len(latencies),
            latency_p50_ms=latencies[int(len(latencies) * 0.50)],
            latency_p95_ms=latencies[int(len(latencies) * 0.95)],
            latency_p99_ms=latencies[int(len(latencies) * 0.99)],
            latency_max_ms=max(latencies),
            success_rate=1.0
        )
        
        self._print_result(result)
        return result
```

## 🔄 Continuous Integration

### GitHub Actions Example

```yaml
name: Docker Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install numpy grpcio grpcio-tools
          pip install -e ./sochdb-python-sdk
      
      - name: Run tests
        run: |
          cd sochdb_docker
          ./run_tests.sh
      
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: test-results
          path: |
            sochdb_docker/integration_test_results.json
            sochdb_docker/performance_benchmark_results.json
```

## 📊 Performance Monitoring

### Prometheus Metrics

Start monitoring stack:

```bash
docker compose --profile monitoring up -d

# Access Prometheus: http://localhost:9090
# Access Grafana: http://localhost:3000
```

### Key Metrics to Monitor

```promql
# Request rate
rate(sochdb_requests_total[5m])

# Latency percentiles
histogram_quantile(0.99, rate(sochdb_request_duration_seconds_bucket[5m]))

# Error rate
rate(sochdb_errors_total[5m])

# Active connections
sochdb_active_connections

# Memory usage
sochdb_memory_usage_bytes
```

## 🎯 Best Practices

1. **Always run integration tests before performance benchmarks**
   - Ensures server is functioning correctly
   - Validates basic operations work

2. **Run benchmarks multiple times**
   - First run warms up the system
   - Take median of 3+ runs for accuracy

3. **Monitor system resources**
   - Check CPU, memory, disk I/O
   - Ensure no resource bottlenecks

4. **Use consistent hardware**
   - Benchmark results vary by hardware
   - Document test environment specs

5. **Test under load**
   - Simulate realistic workloads
   - Test concurrent clients

## 📚 Additional Resources

- [SochDB Documentation](https://sochdb.dev/docs)
- [gRPC Performance Best Practices](https://grpc.io/docs/guides/performance/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Python SDK Documentation](../sochdb-python-sdk/README.md)
