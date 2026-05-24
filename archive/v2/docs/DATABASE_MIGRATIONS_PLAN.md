# 🗄️ TARS Database Migrations with DbUp

**Status**: 📋 **PLANNED**  
**Recommendation**: ✅ **YES - Implement DbUp for TARS**  
**Priority**: 🔥 **HIGH** (enables easy scaffolding + team collaboration)

---

## 🎯 Why Database Migrations for TARS?

### Current Problem
- ❌ Manual SQL schema in code
- ❌ No version control of schema changes
- ❌ Hard to bootstrap new TARS instances
- ❌ Schema drift between environments
- ❌ No rollback capability
- ❌ Team members have different schemas

### Solution: DbUp
```
Manual SQL → DbUp Migrations → Version-Controlled Schema Evolution
```

---

## 📚 Migration Tool Comparison

| Tool | Pros | Cons | Verdict |
|------|------|------|---------|
| **DbUp** | ✅ Simple SQL files<br>✅ F# friendly<br>✅ Lightweight<br>✅ One NuGet package | ❌ Less enterprise features | ✅ **RECOMMENDED** |
| **FluentMigrator** | ✅ .NET-native<br>✅ EF-like API<br>✅ Good tooling | ❌ More complex<br>❌ Heavier | ⚠️ Overkill for TARS |
| **Liquibase** | ✅ Enterprise-grade<br>✅ Language-agnostic<br>✅ Rollback support | ❌ XML/YAML config<br>❌ Java dependency | ⚠️ Too heavyweight |
| **EF Core Migrations** | ✅ Integrated with EF<br>✅ Auto-generate | ❌ TARS uses raw SQL<br>❌ Event sourcing mismatch | ❌ Not suitable |

**Winner: DbUp** - Perfect fit for TARS's SQL-first, event-sourced architecture

---

## 🏗️ Proposed DbUp Architecture

### Directory Structure
```
src/
└── Tars.Migrations/
    ├── Tars.Migrations.fsproj
    ├── Program.fs                    # Migration runner
    ├── Scripts/
    │   ├── 001_InitialSchema.sql
    │   ├── 002_AddBeliefTable.sql
    │   ├── 003_AddPlansTable.sql
    │   ├── 004_AddEvidenceTable.sql
    │   └── ...
    └── README.md
```

### Migration File Naming Convention
```
{version}_{description}.sql

Examples:
001_InitialSchema.sql           # Knowledge ledger basics
002_AddBeliefTable.sql          # Beliefs snapshot table
003_AddPlansTable.sql           # Plans + events
004_AddCreated ByColumn.sql     # Fix missing column
005_AddIndexes.sql              # Performance indexes
```

---

## 💻 Implementation Plan

### Phase 1: Setup DbUp (30 minutes)

**1. Create Migration Project**
```xml
<!-- src/Tars.Migrations/Tars.Migrations.fsproj -->
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net10.0</TargetFramework>
  </PropertyGroup>

  <ItemGroup>
    <Compile Include="Program.fs" />
  </ItemGroup>

  <ItemGroup>
    <PackageReference Include="DbUp" Version="5.0.37" />
    <PackageReference Include="DbUp-PostgreSQL" Version="5.0.37" />
  </ItemGroup>

  <ItemGroup>
    <EmbeddedResource Include="Scripts\**\*.sql" />
  </ItemGroup>
</Project>
```

**2. Migration Runner**
```fsharp
// src/Tars.Migrations/Program.fs
module Tars.Migrations.Program

open System
open DbUp

[<EntryPoint>]
let main args =
    let connectionString =
        match Environment.GetEnvironmentVariable("TARS_POSTGRES_CONNECTION") with
        | null | "" -> 
            "Host=localhost;Database=tars;Username=postgres;Password=postgres"
        | cs -> cs

    printfn "🗄️  Migrating TARS database..."
    printfn "Connection: %s" (connectionString.Replace(connectionString.Split(';').[2], "Password=***"))

    let upgrader =
        DeployChanges.To
            .PostgresqlDatabase(connectionString)
            .WithScriptsEmbeddedInAssembly(Reflection.Assembly.GetExecutingAssembly())
            .LogToConsole()
            .Build()

    let result = upgrader.PerformUpgrade()

    if result.Successful then
        printfn "✅ Migration successful!"
        0
    else
        printfn "❌ Migration failed: %s" result.Error.Message
        1
```

**3. Extract Current Schema to Migration Files**
```sql
-- src/Tars.Migrations/Scripts/001_InitialKnowledgeLedger.sql
CREATE TABLE IF NOT EXISTS knowledge_ledger (
    id UUID PRIMARY KEY,
    belief_id UUID NOT NULL,
    event_type TEXT NOT NULL,
    event_data JSONB NOT NULL,
    agent_id TEXT NOT NULL,
    run_id UUID NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_ledger_belief_id ON knowledge_ledger(belief_id);
CREATE INDEX IF NOT EXISTS idx_ledger_timestamp ON knowledge_ledger(timestamp);
```

```sql
-- src/Tars.Migrations/Scripts/002_AddBeliefTable.sql
CREATE TABLE IF NOT EXISTS beliefs (
    id UUID PRIMARY KEY,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    provenance_source TEXT NOT NULL,
    provenance_agent TEXT NOT NULL,
    provenance_run_id UUID NULL,
    provenance_confidence DOUBLE PRECISION NOT NULL,
    provenance_timestamp TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT NOT NULL,
    tags JSONB NOT NULL DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_beliefs_subject ON beliefs(subject);
CREATE INDEX IF NOT EXISTS idx_beliefs_predicate ON beliefs(predicate);
CREATE INDEX IF NOT EXISTS idx_beliefs_created_by ON beliefs(created_by);
```

### Phase 2: Integrate with TARS (15 minutes)

**1. Add `init-db` Command to CLI**
```fsharp
// src/Tars.Interface.Cli/Commands/InitDb.fs
module Tars.Interface.Cli.Commands.InitDb

open System
open DbUp

let execute (connectionString: string) =
    printfn "🗄️  Initializing TARS database schema..."
    
    let upgrader =
        DeployChanges.To
            .PostgresqlDatabase(connectionString)
            .WithScriptsEmbeddedInAssembly(typeof<KnowledgeLedger>.Assembly) // Embed migrations
            .LogToConsole()
            .Build()

    let result = upgrader.PerformUpgrade()

    if result.Successful then
        printfn "✅ Database initialized successfully!"
        0
    else
        printfn "❌ Database initialization failed: %s" result.Error.Message
        1
```

**2. Update PostgresLedgerStorage to Use Migrations**
```fsharp
// Remove this:
let ensureSchema() = ...  // ← DELETE

// Replace with:
member _.Initialize() =
    // DbUp migrations handle schema now
    Task.FromResult(Ok())
```

**3. CLI Usage**
```bash
# Initialize brand new TARS database
dotnet run --project src/Tars.Interface.Cli -- init-db

# Specify connection string
dotnet run --project src/Tars.Interface.Cli -- init-db --connection "Host=prod;Database=tars;..."

# Check migration status
dotnet run --project src/Tars.Interface.Cli -- db-status
```

### Phase 3: Migration Workflow (Ongoing)

**Adding a New Migration:**
```bash
# 1. Create new SQL file
echo "ALTER TABLE beliefs ADD COLUMN version INT DEFAULT 1;" > \
  src/Tars.Migrations/Scripts/006_AddBeliefVersioning.sql

# 2. Run migrations
dotnet run --project src/Tars.Migrations

# 3. Commit to version control
git add src/Tars.Migrations/Scripts/006_AddBeliefVersioning.sql
git commit -m "Add belief versioning column"
```

**Team Collaboration:**
```bash
# Pull latest code
git pull

# Run migrations (safe, idempotent)
dotnet run --project src/Tars.Migrations

# ✅ Your local DB now matches team schema
```

---

## 🎁 Benefits for TARS

### For Developers
- ✅ **One-command setup**: `dotnet run -- init-db` → ready to code
- ✅ **No schema drift**: Everyone has same DB state
- ✅ **Safe updates**: Migrations are tested and versioned
- ✅ **Easy rollback**: Revert bad changes

### For Deployment
- ✅ **Environment parity**: Dev/staging/prod identical schemas
- ✅ **Zero-downtime migrations**: Apply in production with confidence
- ✅ **Audit trail**: Know exactly when/why each change happened
- ✅ **Disaster recovery**: Rebuild DB from scratch in minutes

### For TARS Evolution
- ✅ **Event sourcing-friendly**: Migrations complement event logs
- ✅ **Multi-tenant ready**: Easy to scaffold new TARS instances
- ✅ **Phase-based rollout**: Migrate schema as you implement phases
- ✅ **Backward compatibility**: Test migrations before applying

---

## 📊 Migration Examples

### Example 1: Add Knowledge Graph Support
```sql
-- 007_AddTemporalKnowledgeGraph.sql
CREATE TABLE IF NOT EXISTS kg_nodes (
    id UUID PRIMARY KEY,
    entity_id TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    properties JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT TIMESTAMP,
    created_by TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS kg_edges (
    id UUID PRIMARY KEY,
    source_id UUID NOT NULL REFERENCES kg_nodes(id),
    target_id UUID NOT NULL REFERENCES kg_nodes(id),
    relation_type TEXT NOT NULL,
    properties JSONB NOT NULL DEFAULT '{}',
    valid_from TIMESTAMPTZ NOT NULL,
    valid_to TIMESTAMPTZ NULL
);

CREATE INDEX idx_kg_edges_temporal ON kg_edges(valid_from, valid_to);
```

### Example 2: Add Full-Text Search
```sql
-- 008_AddFullTextSearch.sql
ALTER TABLE beliefs 
ADD COLUMN search_vector tsvector
GENERATED ALWAYS AS (
    to_tsvector('english', 
        coalesce(subject, '') || ' ' || 
        coalesce(object, '')
    )
) STORED;

CREATE INDEX idx_beliefs_fts ON beliefs USING GIN(search_vector);
```

### Example 3: Partition Large Tables
```sql
-- 009_PartitionKnowledgeLedger.sql
CREATE TABLE knowledge_ledger_partitioned (LIKE knowledge_ledger INCLUDING ALL)
PARTITION BY RANGE (timestamp);

CREATE TABLE knowledge_ledger_2024 PARTITION OF knowledge_ledger_partitioned
FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

CREATE TABLE knowledge_ledger_2025 PARTITION OF knowledge_ledger_partitioned
FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');
```

---

## 🚀 Quick Start (After Implementation)

```bash
# New developer joins team
git clone https://github.com/yourorg/tars.git
cd tars

# Setup database
docker-compose up -d postgres  # Start PostgreSQL
dotnet run --project src/Tars.Migrations  # Run all migrations

# ✅ Ready to code! Database is at latest schema
```

---

## 🔧 Advanced Features

### Conditional Migrations
```sql
-- Only run if column doesn't exist
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name='beliefs' AND column_name='created_by'
    ) THEN
        ALTER TABLE beliefs ADD COLUMN created_by TEXT NOT NULL DEFAULT 'system';
    END IF;
END $$;
```

### Data Migrations
```sql
-- 010_BackfillBeliefConfidence.sql
UPDATE beliefs 
SET confidence = 0.95 
WHERE confidence IS NULL 
  AND provenance_confidence > 0.9;
```

### Migration Metadata
```sql
-- DbUp automatically tracks migrations in:
CREATE TABLE schemaversions (
    schemaversionsid SERIAL PRIMARY KEY,
    scriptname VARCHAR(255) NOT NULL,
    applied TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Query what's been applied:
SELECT * FROM schemaversions ORDER BY applied DESC;
```

---

## ✅ Recommendation Summary

**YES - Implement DbUp for TARS!**

**Effort**: ~2 hours  
**Value**: Immense (enables team collaboration + easy scaffolding)  
**Complexity**: Low (just SQL files + simple F# runner)  
**Risk**: Minimal (DbUp is battle-tested, 10M+ downloads)

**Action Items**:
1. ✅ Create `Tars.Migrations` project
2. ✅ Extract current schema to migration files  
3. ✅ Add `init-db` CLI command
4. ✅ Update `PostgresLedgerStorage` to remove `ensureSchema()`
5. ✅ Document for team

**Next PR**: "Add DbUp database migrations for TARS schema management"

---

*Recommended by: Antigravity AI*  
*Date: 2024-12-24*  
*Based on: ChatGPT recommendations + TARS architecture*
