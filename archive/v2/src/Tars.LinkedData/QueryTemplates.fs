namespace Tars.LinkedData

module QueryTemplates =

    /// Common SPARQL Endpoints
    module Endpoints =
        let Wikidata = "https://query.wikidata.org/sparql"
        let DBpedia = "https://dbpedia.org/sparql"
        let UniProt = "https://sparql.uniprot.org/sparql"

    /// Pre-defined SPARQL queries
    module Queries =
        
        /// Get programming languages from Wikidata (limit 20)
        let getProgrammingLanguages = """
            SELECT ?lang ?langLabel ?paradigmLabel WHERE {
              ?lang wdt:P31/wdt:P279* wd:Q9143 .
              OPTIONAL { ?lang wdt:P3966 ?paradigm . }
              SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
            }
            LIMIT 20
        """

        /// Get programming languages from DBpedia (limit 20)
        let getDbpediaLanguages = """
            PREFIX dbo: <http://dbpedia.org/ontology/>
            PREFIX dbr: <http://dbpedia.org/resource/>
            
            SELECT ?lang ?name ?designer WHERE {
              ?lang a dbo:ProgrammingLanguage ;
                    rdfs:label ?name .
              OPTIONAL { ?lang dbo:designer ?d . ?d rdfs:label ?designer }
              FILTER (lang(?name) = 'en')
              FILTER (lang(?designer) = 'en')
            }
            LIMIT 20
        """

        /// Find an entity by name (Wikidata)
        let findEntityByName (name: string) = 
            sprintf """
            SELECT ?item ?itemLabel ?description WHERE {
              ?item rdfs:label "%s"@en .
              OPTIONAL { ?item schema:description ?description . FILTER(LANG(?description) = "en") }
              SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
            }
            LIMIT 5
            """ name

        /// Get all properties of a specific entity (Wikidata)
        let getEntityProperties (entityId: string) =
            sprintf """
            SELECT ?propUrl ?propLabel ?valUrl ?valLabel WHERE {
              wd:%s ?propUrl ?valUrl .
              ?property ?ref ?propUrl .
              ?property rdf:type wikibase:Property .
              ?property rdfs:label ?propLabel .
              FILTER (lang(?propLabel) = 'en' )
              
              BIND(?valUrl AS ?val) . 
              SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
            }
            LIMIT 50
            """ entityId
