namespace Tars.LinkedData

/// Catalog of well-known Linked Open Data (LOD) datasets and endpoints
module DatasetCatalog =

    type Dataset = {
        Id: string
        Name: string
        Description: string
        Endpoint: string option
        DownloadUrl: string option
        Format: string
        Domain: string
    }

    let CommonDatasets = [
        { Id = "wikidata"; Name = "Wikidata"; Description = "The free knowledge base that anyone can edit."; 
          Endpoint = Some "https://query.wikidata.org/sparql"; DownloadUrl = None; Format = "RDF/SPARQL"; Domain = "General" }
        
        { Id = "dbpedia"; Name = "DBpedia"; Description = "Structured version of Wikipedia info boxes."; 
          Endpoint = Some "https://dbpedia.org/sparql"; DownloadUrl = Some "https://www.dbpedia.org/resources/data-set/"; Format = "RDF/SPARQL"; Domain = "General" }
        
        { Id = "geonames"; Name = "GeoNames"; Description = "Geographical database containing over 25 million geographical names."; 
          Endpoint = None; DownloadUrl = Some "https://download.geonames.org/all-countries.zip"; Format = "TSV/RDF"; Domain = "Geography" }
          
        { Id = "uniprot"; Name = "UniProt"; Description = "Comprehensive resource for protein sequence and annotation data."; 
          Endpoint = Some "https://sparql.uniprot.org/sparql"; DownloadUrl = None; Format = "RDF/SPARQL"; Domain = "Life Sciences" }
          
        { Id = "wordnet"; Name = "WordNet"; Description = "Lexical database for English."; 
          Endpoint = Some "http://wordnet-rdf.princeton.edu/sparql"; DownloadUrl = None; Format = "RDF"; Domain = "Linguistics" }
    ]

    let list () = CommonDatasets

    let findById (id: string) = 
        CommonDatasets |> List.tryFind (fun d -> d.Id.ToLowerInvariant() = id.ToLowerInvariant())

    let search (query: string) =
        let q = query.ToLowerInvariant()
        CommonDatasets |> List.filter (fun d -> 
            d.Name.ToLowerInvariant().Contains(q) || 
            d.Description.ToLowerInvariant().Contains(q) ||
            d.Domain.ToLowerInvariant().Contains(q))