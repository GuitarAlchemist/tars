# TARS v2 - Quick Reference Guide

## 🚀 New Commands (December 2025)

### `tars reflect` - Symbolic Reflection
Scan the knowledge ledger for contradictions and inconsistencies.

```bash
# Basic reflection scan
tars reflect

# Scan + cleanup beliefs below 30% confidence
tars reflect --cleanup 0.3

# Show help
tars reflect --help
```

**Output:**
```
╔═══════════════════════════════════════════════════════════╗
║              TARS Symbolic Reflection                     ║
╚═══════════════════════════════════════════════════════════╝

📊 Ledger Stats:
   - Valid Beliefs: 42
   - Total Beliefs: 50
   - Current Contradictions: 2
   - Unique Subjects: 18
   - Unique Objects: 25

🔍 Running symbolic reflection...

✅ Reflection complete!
   - New contradictions found: 3
   - Total contradictions now: 5

💡 Tip: Use --cleanup <threshold> to auto-retract low-confidence beliefs
```

---

## 📊 Production Observability

### Health Check
```bash
curl http://localhost:9090/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-12-25T21:15:30Z",
  "checks": {
    "memory": "ok",
    "disk": "ok"
  }
}
```

### Metrics (Prometheus Format)
```bash
curl http://localhost:9090/metrics
```

**Sample Output:**
```
# TYPE tars_llm_request_total counter
tars_llm_request_total{status="success"} 42
tars_llm_request_total{status="error"} 3

# TYPE tars_llm_request_duration_ms_avg gauge
tars _llm_request_duration_ms_avg 1250.5
```

---

## 🐳 Docker Infrastructure

### Start All Services
```bash
# Full stack (Postgres, Chroma, Neo4j, Graphiti)
docker-compose -f docker-compose.all.yml up -d

# Monitoring only (Prometheus + Grafana)
docker-compose -f docker-compose.monitoring.yml up -d
```

### Access Services
- **Grafana**: http://localhost:3001 (admin/tars_admin)
- **Prometheus**: http://localhost:9091
- **Neo4j Browser**: http://localhost:7474
- **Chroma**: http://localhost:8000

### Stop Services
```bash
docker-compose -f docker-compose.all.yml down
```

---

## 🧪 Validation & Testing

### Run Production Validation
```bash
pwsh ./validate_production.ps1
```

**Tests:**
- ✅ Full solution build
- ✅ VerifierAgent tests (3 tests)
- ✅ Metrics export functionality
- ✅ Configuration management
- ✅ InfrastructureServer endpoints
- ✅ Docker compose files
- ✅ ReflectionAgent implementation
- ✅ Project file integrity

### Run Specific Tests
```bash
# VerifierAgent contradiction detection
dotnet test tests/Tars.Tests/Tars.Tests.fsproj --filter VerifierAgent

# All tests
dotnet test Tars.sln
```

---

## 💾 Knowledge Management

### Check Ledger Status
```bash
# Via reflect command
tars reflect

# Direct query (if Postgres configured)
psql -h localhost -U tars_user -d tars_kb -c "SELECT COUNT(*) FROM beliefs;"
```

### Clean Up Low-Confidence Beliefs
```bash
# Retract beliefs below 30% confidence
tars reflect --cleanup 0.3

# Retract beliefs below 50% confidence
tars reflect --cleanup 0.5
```

---

## 🎯 Quick Troubleshooting

### TARS Won't Start
```bash
# Check if ports are available
netstat -an | findstr "9090 11434"

# View logs
dotnet run --project src/Tars.Interface.Cli/Tars.Interface.Cli.fsproj -- chat 2>&1 | tee tars.log
```

### Metrics Not Showing
```bash
# 1. Check if server is running
curl http://localhost:9090/health

# 2. Check configuration
cat src/Tars.Interface.Cli/appsettings.json | grep -A 3 "Metrics"

# 3. Check logs for "Metrics Config:"
dotnet run --project src/Tars.Interface.Cli/Tars.Interface.Cli.fsproj -- chat | grep "Metrics"
```

### Docker Services Won't Start
```bash
# Check Docker is running
docker ps

# View logs
docker-compose -f docker-compose.all.yml logs

# Restart specific service
docker-compose -f docker-compose.all.yml restart postgres
```

---

## 📚 Configuration

### appsettings.json
```json
{
  "Metrics": {
    "Enabled": true,
    "Port": 9090
  }
}
```

### Environment Variables
```bash
# Enable metrics
export TARS_METRICS_ENABLED=true

# Change metrics port
export TARS_METRICS_PORT=9091

# Set Postgres connection
export TARS_POSTGRES_CONN="Host=localhost;Database=tars_kb;Username=tars_user;Password=..."
```

---

## 🔧 Development Workflow

### Build & Run
```bash
# Full build
dotnet build Tars.sln

# Run CLI
dotnet run --project src/Tars.Interface.Cli/Tars.Interface.Cli.fsproj -- chat

# Run with specific model
dotnet run --project src/Tars.Interface.Cli/Tars.Interface.Cli.fsproj -- chat --model qwen2.5-coder:32b
```

### Hot Reload (Development)
```bash
# Terminal 1: Infrastructure
docker-compose -f docker-compose.all.yml up

# Terminal 2: TARS with watch mode
dotnet watch --project src/Tars.Interface.Cli/Tars.Interface.Cli.fsproj run -- chat
```

---

## 📈 Monitoring Dashboards

### Grafana Setup
1. Open http://localhost:3001
2. Login: admin/tars_admin
3. Add Prometheus data source: http://prometheus:9090
4. Import dashboard or create new

### Prometheus Queries
```promql
# Request rate (requests per second)
rate(tars_llm_request_total[5m])

# Average request duration
tars_llm_request_duration_ms_avg

# Error rate
rate(tars_llm_request_total{status="error"}[5m])
```

---

## 🎓 Learning Resources

### Architecture Docs
- `docs/1_Vision/architectural_vision.md` - Core philosophy
- `SESSION_SUMMARY_2025-12-25.md` - Latest updates
- `task.md` - Current progress tracker

### Code Examples
- `src/Tars.Knowledge/ReflectionAgent.fs` - Self-correction logic
- `src/Tars.Symbolic/ConstraintScoring.fs` - Contradiction detection
- `src/Tars.Interface.Cli/InfrastructureServer.fs` - HTTP endpoints

---

**Last Updated**: December 25, 2025
**TARS Version**: v2 (Phase 7 Complete, Phase 9 75%)
**Status**: Production-Ready Observability ✅ | Self-Correcting Knowledge ✅
