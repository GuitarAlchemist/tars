# TARS v2 - RDF/Linked Data Ingestion Plan

## 🌐 Vision: Ground TARS in the Linked Open Data Cloud

Instead of extracting beliefs from unstructured text via LLM, we can **directly ingest structured RDF triples** from authoritative sources. This provides:

- **Deterministic parsing** - No LLM hallucination
- **Billion-scale knowledge** - Wikidata has 12B+ triples
- **Global linking** - Connect to standard URIs
- **Provenance** - Track source datasets

## 📊 Architecture

```
LOD Cloud / Wikidata / DBpedia
           ↓
[SPARQL Query / RDF Dump Download]
           ↓
[RDF Parser] → (Subject, Predicate, Object) URIs
           ↓
[URI Resolver] → Convert URIs to readable labels
           ↓
[VerifierAgent.Verify] → Check consistency (optional)
           ↓
[KnowledgeLedger.Assert] → Store with provenance
           ↓
[ReflectionAgent.Reflect] → Detect contradictions across sources
```

## 🎯 Primary Data Sources

### 1. **Wikidata** (Tier 1 - Most Comprehensive)
- **Scale**: 100M+ entities, 12B+ triples
- **Format**: RDF dumps (Turtle, N-Triples)
- **SPARQL**: https://query.wikidata.org/sparql
- **Dumps**: https://dumps.wikimedia.org/wikidatawiki/entities/

**Sample Query**:
```sparql
# Get all programming languages and their paradigms
SELECT ?lang ?langLabel ?paradigm ?paradigmLabel
WHERE {
  ?lang wdt:P31 wd:Q9143 .  # instance of: programming language
  ?lang wdt:P3966 ?paradigm . # programming paradigm
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
}
LIMIT 1000
```

### 2. **DBpedia** (Tier 1 - Wikipedia as Linked Data)
- **Scale**: 8M+ entities, 3B+ triples
- **Format**: RDF dumps (Turtle, N-Triples)
- **SPARQL**: https://dbpedia.org/sparql
- **Dumps**: https://databus.dbpedia.org/dbpedia/collections/latest-core

**Sample Query**:
```sparql
# Get all cities with population > 1M
SELECT ?city ?cityLabel ?population
WHERE {
  ?city a dbo:City ;
        dbp:populationTotal ?population ;
        rdfs:label ?cityLabel .
  FILTER (?population > 1000000)
  FILTER (lang(?cityLabel) = 'en')
}
LIMIT 1000
```

### 3. **GeoNames** (Tier 2 - Geographical Data)
- **Scale**: 25M+ place names
- **Format**: RDF dumps
- **Download**: http://download.geonames.org/all-geonames-rdf.zip

### 4. **LOD Cloud Catalog** (Discovery)
- **Website**: https://lod-cloud.net/
- **API**: Programmatic access to dataset metadata
- **Use**: Discover domain-specific datasets dynamically

## 🏗️ Implementation Plan

### Phase 1: RDF Parser & Importer (1-2 days)
**Goal**: Parse Turtle/N-Triples and import into KnowledgeLedger

**Components**:
1. `RdfParser.fs` - Parse RDF formats (use dotNetRDF library)
2. `UriResolver.fs` - Resolve URIs to human-readable labels
3. `RdfImporter.fs` - Batch import triples
4. `tars ingest-rdf <file.ttl>` - CLI command

**Libraries**:
```xml
<PackageReference Include="dotNetRDF" Version="3.1.0" />
```

### Phase 2: SPARQL Query Integration (1 day)
**Goal**: Query live SPARQL endpoints (Wikidata, DBpedia)

**Components**:
1. `SparqlClient.fs` - Execute SPARQL queries
2. `QueryTemplates.fs` - Pre-built queries for common patterns
3. `tars query-sparql <endpoint> <query>` - CLI command

**Example**:
```bash
# Query Wikidata for programming languages
tars query-sparql wikidata "SELECT ?lang WHERE { ?lang wdt:P31 wd:Q9143 } LIMIT 100"

# Import results into ledger
tars query-sparql wikidata languages.sparql --import
```

### Phase 3: Incremental Sync (2 days)
**Goal**: Keep ledger synchronized with upstream sources

**Components**:
1. `ChangeDetector.fs` - Track dataset versions
2. `IncrementalSync.fs` - Download only new/changed triples
3. `Scheduler.fs` - Periodic sync (daily/weekly)

### Phase 4: Multi-Source Reasoning (2-3 days)
**Goal**: Detect contradictions across different data sources

**Scenarios**:
- Wikidata says "Paris population: 2.1M"
- DBpedia says "Paris population: 2.2M"
- ReflectionAgent detects discrepancy, records both with provenance

## 📦 Data Formats

### Turtle (.ttl) - Human-Readable
```turtle
@prefix ex: <http://example.org/> .
@prefix foaf: <http://xmlns.com/foaf/0.1/> .

ex:alice a foaf:Person ;
    foaf:name "Alice" ;
    foaf:knows ex:bob .

ex:bob a foaf:Person ;
    foaf:name "Bob" .
```

### N-Triples (.nt) - Machine-Optimized
```ntriples
<http://example.org/alice> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person> .
<http://example.org/alice> <http://xmlns.com/foaf/0.1/name> "Alice" .
<http://example.org/alice> <http://xmlns.com/foaf/0.1/knows> <http://example.org/bob> .
```

## 🎯 Quick Win: DBpedia Programming Languages

**Demonstration**: Import all programming languages from DBpedia

```fsharp
// Query DBpedia for programming languages
let query = """
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?lang ?label ?paradigm
WHERE {
  ?lang a dbo:ProgrammingLanguage ;
        rdfs:label ?label ;
        dbo:programmingParadigm ?paradigm .
  FILTER (lang(?label) = 'en')
}
LIMIT 500
"""

// Convert SPARQL results to Beliefs
// (F#, has paradigm, functional)
// (Python, has paradigm, object-oriented)
// (Rust, has paradigm, systems)
```

**Result**: ~500 verified beliefs about programming languages in seconds!

## 🔧 Code Structure

```
src/Tars.LinkedData/
├── RdfParser.fs           - Parse Turtle/N-Triples
├── UriResolver.fs         - Resolve URIs to labels
├── SparqlClient.fs        - Query SPARQL endpoints  
├── QueryTemplates.fs      - Common query patterns
├── RdfImporter.fs         - Batch import logic
├── IncrementalSync.fs     - Track changes
└── Tars.LinkedData.fsproj

src/Tars.Interface.Cli/Commands/
├── IngestRdfCommand.fs    - tars ingest-rdf
└── QuerySparqlCommand.fs  - tars query-sparql
```

## 📊 Expected Impact

| Metric | Before (Wikipedia) | After (RDF/LOD) |
|--------|-------------------|-----------------|
| **Precision** | ~70% (LLM errors) | ~99% (verified triples) |
| **Scale** | ~50 facts/article | Billions of triples |
| **Speed** | 30s/article (LLM) | 100K triples/sec (parsing) |
| **Cost** | $0.01/article (LLM) | $0 (open data) |
| **Provenance** | "From Wikipedia" | "From Wikidata v2024-12-01" |

## 🎓 Learning Curve

**RDF Basics** (1 hour):
- Subject-Predicate-Object structure
- URIs vs Literals
- Common vocabularies (FOAF, Dublin Core, SKOS)

**SPARQL Basics** (2 hours):
- SELECT queries
- FILTER and BIND
- Property paths
- Federated queries

**dotNetRDF Library** (1 hour):
- Load RDF files
- Parse triples
- Execute SPARQL

**Total**: ~4 hours to productive RDF ingestion

## 🚀 Next Steps

1. **Install dotNetRDF** - Add NuGet package
2. **Create RdfParser** - Basic Turtle parsing
3. **Test with DBpedia dump** - Download small subset (1MB)
4. **Import to ledger** - Verify ReflectionAgent works with RDF beliefs
5. **Query Wikidata** - Live SPARQL integration
6. **Scale up** - Incremental sync with full dumps

## 🎯 Immediate Demo (30 mins)

```bash
# 1. Download DBpedia programming languages (small subset)
wget http://downloads.dbpedia.org/2016-10/core-i18n/en/programming_language_en.ttl.bz2
bunzip2 programming_language_en.ttl.bz2

# 2. Ingest into TARS
tars ingest-rdf programming_language_en.ttl

# 3. Run reflection
tars reflect

# 4. Query ledger
tars query "show me functional programming languages"
```

**Result**: TARS now has deep, verified knowledge of all programming languages from DBpedia!

---

**This is the future of TARS**: Grounded in the **world's largest knowledge graph** (Wikidata), connected to the **global Linked Data cloud**, with **provenance tracking** and **multi-source reasoning**. 🌐🧠
