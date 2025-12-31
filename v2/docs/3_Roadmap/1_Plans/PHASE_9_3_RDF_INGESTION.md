# Phase 9.3: RDF/Linked Data Ingestion

## 🎯 Objective
Replace HTML-based Wikipedia extraction with **direct RDF triple import** from authoritative Linked Open Data sources. This provides deterministic ingestion of billions of verified facts from Wikidata, DBpedia, GeoNames, and the broader LOD Cloud.

## 🌟 Why RDF > LLM Extraction?

| Aspect | LLM Extraction (Old) | RDF Ingestion (New) |
|--------|---------------------|-------------------|
| **Precision** | ~70% (hallucinations) | ~99% (verified data) |
| **Scale** | ~50 facts/article | Billions of triples |
| **Speed** | 30s/article | 100K triples/sec |
| **Cost** | $0.01/article | $0 (open data) |
| **Provenance** | "From Wikipedia" | "Wikidata v2024-12-01" |
| **Global Linking** | No | Yes (URIs) |

## 📋 Implementation Checklist

### ✅ Prerequisites
- [ ] Add `dotNetRDF` NuGet package (v3.1.0)
- [ ] Create `Tars.LinkedData` project
- [ ] Study RDF/Turtle basics (1 hour)
- [ ] Study SPARQL basics (2 hours)

### 🔧 Phase 1: RDF Parser (2 days)

#### Files to Create
- [ ] `src/Tars.LinkedData/RdfParser.fs` - Parse Turtle/N-Triples/RDF-XML
- [ ] `src/Tars.LinkedData/UriResolver.fs` - Resolve URIs to labels
- [ ] `src/Tars.LinkedData/RdfImporter.fs` - Batch import to ledger
- [ ] `src/Tars.LinkedData/Tars.LinkedData.fsproj` - Project file

#### Tasks
- [ ] Implement Turtle parser using dotNetRDF
- [ ] Convert RDF triples to `Tars.Knowledge.Belief`
- [ ] Handle blank nodes and literals
- [ ] Batch import optimization (1000+ triples/sec)

#### Test Data
```bash
# Download small DBpedia subset
wget http://downloads.dbpedia.org/2016-10/core-i18n/en/infobox_properties_en.ttl.bz2
```

### 🌐 Phase 2: SPARQL Client (1 day)

#### Files to Create
- [ ] `src/Tars.LinkedData/SparqlClient.fs` - Execute queries
- [ ] `src/Tars.LinkedData/QueryTemplates.fs` - Common patterns
- [ ] `src/Tars.Interface.Cli/Commands/QuerySparqlCommand.fs` - CLI

#### Tasks
- [ ] Implement SPARQL query execution
- [ ] Support Wikidata, DBpedia, GeoNames endpoints
- [ ] Parse SPARQL result sets (JSON/XML)
- [ ] Convert results to Beliefs

#### Example Queries
```sparql
# Programming languages from Wikidata
SELECT ?lang ?langLabel WHERE {
  ?lang wdt:P31 wd:Q9143 .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
}

# Cities from DBpedia
SELECT ?city ?population WHERE {
  ?city a dbo:City ;
        dbp:populationTotal ?population .
  FILTER (?population > 1000000)
}
```

### 📊 Phase 3: Data Source Catalog (1 day)

#### LOD Cloud Integration
- [ ] Fetch dataset metadata from lod-cloud.net API
- [ ] Parse DCAT catalogs (data.gov, EU portals)
- [ ] Build dataset registry

#### Files to Create
- [ ] `src/Tars.LinkedData/DatasetCatalog.fs` - Discover datasets
- [ ] `src/Tars.LinkedData/DcatParser.fs` - Parse DCAT metadata

### 🔄 Phase 4: Incremental Sync (2 days)

#### Features
- [ ] Track dataset versions (MD5/SHA hashes)
- [ ] Download only changed triples
- [ ] Scheduled updates (daily/weekly)
- [ ] Conflict resolution (multiple sources)

#### Files to Create
- [ ] `src/Tars.LinkedData/ChangeDetector.fs` - Version tracking
- [ ] `src/Tars.LinkedData/IncrementalSync.fs` - Delta updates
- [ ] `src/Tars.LinkedData/Scheduler.fs` - Periodic sync

### 🧪 Phase 5: Multi-Source Reasoning (2 days)

#### Cross-Source Validation
- [ ] Detect contradictions between Wikidata & DBpedia
- [ ] Confidence scoring based on source authority
- [ ] Provenance-aware conflict resolution

#### Example
```
Wikidata: (Paris, population, 2165423) [confidence: 0.95]
DBpedia:  (Paris, population, 2220445) [confidence: 0.85]

ReflectionAgent → Mark as "Ambiguous - Multiple Sources"
```

## 📦 Project Structure

```
src/Tars.LinkedData/
├── Domain.fs               # RDF-specific types
├── RdfParser.fs            # Parse Turtle/N-Triples
├── UriResolver.fs          # URI → Label resolution
├── SparqlClient.fs         # Query SPARQL endpoints
├── QueryTemplates.fs       # Common SPARQL patterns
├── RdfImporter.fs          # Batch import logic
├── DatasetCatalog.fs       # Discover LOD datasets
├── DcatParser.fs           # Parse DCAT catalogs
├── ChangeDetector.fs       # Track versions
├── IncrementalSync.fs      # Delta updates
├── Scheduler.fs            # Periodic sync
└── Tars.LinkedData.fsproj

src/Tars.Interface.Cli/Commands/
├── IngestRdfCommand.fs     # tars ingest-rdf <file>
├── QuerySparqlCommand.fs   # tars query-sparql <endpoint> <query>
└── SyncCommand.fs          # tars sync <dataset>
```

## 🎯 CLI Commands

### `tars ingest-rdf`
```bash
# Ingest local RDF file
tars ingest-rdf programming_languages.ttl

# Ingest from URL
tars ingest-rdf https://dbpedia.org/data/functional_programming.ttl

# Batch mode (directory)
tars ingest-rdf ./rdf_dumps/*.ttl
```

### `tars query-sparql`
```bash
# Query Wikidata
tars query-sparql wikidata "SELECT ?lang WHERE { ?lang wdt:P31 wd:Q9143 } LIMIT 100"

# Save results
tars query-sparql wikidata languages.sparql --output results.json

# Import results into ledger
tars query-sparql dbpedia cities.sparql --import
```

### `tars sync`
```bash
# Sync with Wikidata (incremental)
tars sync wikidata --incremental

# Full dump import
tars sync dbpedia --full

# Schedule daily sync
tars sync wikidata --schedule daily
```

### `tars catalog`
```bash
# List available datasets
tars catalog list

# Search for specific domains
tars catalog search "life sciences"

# Show dataset info
tars catalog info dbpedia
```

## 🚀 Quick Win: DBpedia Programming Languages (30 mins)

### Step-by-Step Demo

```bash
# 1. Create LinkedData project
dotnet new classlib -n Tars.LinkedData -lang F# -o src/Tars.LinkedData
cd src/Tars.LinkedData
dotnet add package dotNetRDF

# 2. Download DBpedia sample
wget http://downloads.dbpedia.org/2016-10/core-i18n/en/programming_language_en.ttl.bz2
bunzip2 programming_language_en.ttl.bz2

# 3. Build & run ingestion
dotnet build Tars.sln
tars ingest-rdf programming_language_en.ttl

# 4. Verify import
tars reflect  # Should show ~500 new beliefs

# 5. Query
tars query "What functional programming languages are there?"
```

## 📊 Success Metrics

- [ ] Parse 100K+ triples/sec
- [ ] Import Wikidata programming languages (~5K beliefs)
- [ ] Import DBpedia cities (~10K beliefs)
- [ ] Detect 0 contradictions (verified data)
- [ ] Query live SPARQL endpoints
- [ ] Incremental sync working (track changes)

## 🎓 Learning Resources

### RDF Basics
- W3C RDF Primer: https://www.w3.org/TR/rdf11-primer/
- Turtle Syntax: https://www.w3.org/TR/turtle/

### SPARQL Tutorials
- W3C SPARQL: https://www.w3.org/TR/sparql11-query/
- Wikidata Query Service: https://query.wikidata.org/

### dotNetRDF Documentation
- GitHub: https://github.com/dotnetrdf/dotnetrdf
- User Guide: https://github.com/dotnetrdf/dotnetrdf/wiki

## 🔮 Future Extensions

### Phase 10: Semantic Reasoning
- [ ] OWL2 ontology support
- [ ] RDFS inferencing
- [ ] SHACL validation

### Phase 11: Knowledge Graph Visualization
- [ ] 3D force-directed graph of beliefs
- [ ] Real-time updates from SPARQL
- [ ] Interactive exploration

### Phase 12: Federated Queries
- [ ] Query multiple endpoints simultaneously
- [ ] Join results across sources
- [ ] Distributed knowledge graph

---

## 🎉 Expected Outcome

After Phase 9.3, TARS will:
- ✅ Ingest billions of verified triples from LOD Cloud
- ✅ Query live SPARQL endpoints (Wikidata, DBpedia)
- ✅ Track provenance for all facts
- ✅ Detect contradictions across multiple sources
- ✅ Sync incrementally with upstream datasets
- ✅ Discover new datasets automatically

**This transforms TARS from a chatbot into a grounded, multi-source knowledge reasoner.** 🌐🧠

---

**Status**: Planning Complete ✅ | Ready for Implementation
**Effort**: ~7-10 days for full implementation
**Priority**: **HIGH** - Replaces brittle LLM extraction with robust RDF parsing
