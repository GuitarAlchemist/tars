open System
open WebResearchEngine

[<EntryPoint>]
let main argv =
    printfn "🔍 TARS Web Research Engine Demo"

    // Create sample research queries
    let queries = [
        {
            Query = "F# functional programming best practices"
            Domain = Some "programming"
            MaxResults = 5
            RequiredConfidence = 0.7
        }
        {
            Query = "machine learning algorithms 2024"
            Domain = Some "ai"
            MaxResults = 3
            RequiredConfidence = 0.8
        }
    ]

    // Execute research
    async {
        let! results = conductResearch queries

        printfn "\n📊 Research Results Summary:"
        for result in results do
            printfn $"\n🔍 Query: {result.Query.Query}"
            printfn $"   📈 Sources: {result.TotalSources}"
            printfn $"   🎯 Confidence: {result.AverageConfidence:F2}"
            printfn $"   ⏱️ Time: {result.ExecutionTime.TotalMilliseconds:F0}ms"

            for searchResult in result.Results |> List.take 2 do
                printfn $"   📄 {searchResult.Title}"
                printfn $"      {searchResult.Url}"

        // Validate research quality
        let qualityReport = validateResearchQuality results
        printfn $"\n📋 Quality Report:"
        printfn $"   Total Queries: {qualityReport.TotalQueries}"
        printfn $"   Total Sources: {qualityReport.TotalSources}"
        printfn $"   Average Confidence: {qualityReport.AverageConfidence:F2}"
        printfn $"   Quality Score: {qualityReport.QualityScore:F2}"
        printfn $"   High Quality: {qualityReport.IsHighQuality}"

        printfn "\n✅ Web Research Engine Demo Complete!"
    } |> Async.RunSynchronously

    0
