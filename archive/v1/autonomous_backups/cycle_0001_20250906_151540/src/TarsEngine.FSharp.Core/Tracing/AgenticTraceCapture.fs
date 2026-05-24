namespace TarsEngine.FSharp.Core.Tracing

open System
open System.IO
open System.Text.Json
open System.Diagnostics

/// Comprehensive Agentic Trace Capture System
/// Captures all inter-agent events, grammar evolution, and architecture snapshots
module AgenticTraceCapture =

    // ============================================================================
    // TRACE DATA STRUCTURES
    // ============================================================================

    type AgentEvent = {
        Timestamp: DateTime
        AgentId: string
        EventType: string
        Description: string
        InputData: Map<string, obj>
        OutputData: Map<string, obj>
        ProcessingTimeMs: float
        MemoryUsageMB: float
        GrammarTierUsed: int
        ComputationalExpressions: List<string>
    }

    type InterAgentCommunication = {
        Timestamp: DateTime
        SourceAgent: string
        TargetAgent: string
        MessageType: string
        Payload: Map<string, obj>
        ResponseTime: float
        Success: bool
        ErrorDetails: string option
    }

    type GrammarEvolutionEvent = {
        Timestamp: DateTime
        FromTier: int
        ToTier: int
        EvolutionTrigger: string
        NewCapabilities: List<string>
        PerformanceImpact: Map<string, float>
        EvolutionReasoning: string
        CodeGenerated: List<string>
    }

    type TarsArchitectureSnapshot = {
        Timestamp: DateTime
        SessionId: string
        ActiveTiers: List<int>
        LoadedModules: List<string>
        AgentInstances: Map<string, string>  // AgentId -> AgentType
        GrammarCapabilities: Map<int, List<string>>  // Tier -> Capabilities
        MemoryFootprint: Map<string, float>
        PerformanceMetrics: Map<string, float>
        ConfigurationState: Map<string, obj>
    }

    type DataSourceAccess = {
        Timestamp: DateTime
        AgentId: string
        DataSourceType: string  // "CosmologicalDB", "ArXiv", "NASA", "Planck", etc.
        DataSourceUrl: string
        QueryParameters: Map<string, obj>
        DataRetrieved: int64  // bytes
        ProcessingTime: float
        Success: bool
        ErrorDetails: string option
    }

    type WebRequest = {
        Timestamp: DateTime
        AgentId: string
        Method: string  // GET, POST, etc.
        Url: string
        Headers: Map<string, string>
        RequestBody: string option
        ResponseCode: int
        ResponseSize: int64
        ResponseTime: float
        Success: bool
        ErrorMessage: string option
    }

    type TripleStoreQuery = {
        Timestamp: DateTime
        AgentId: string
        QueryType: string  // "SPARQL", "Cypher", "GraphQL"
        Query: string
        TripleStore: string  // "CosmologyOntology", "PhysicsKG", "ResearchGraph"
        ResultCount: int
        QueryTime: float
        DataTransferred: int64
        Success: bool
    }

    type VectorStoreOperation = {
        Timestamp: DateTime
        AgentId: string
        Operation: string  // "Embed", "Search", "Insert", "Update"
        VectorStore: string  // "ChromaDB", "Pinecone", "Weaviate"
        DocumentsProcessed: int
        EmbeddingDimensions: int
        SimilarityThreshold: float option
        ResultCount: int
        ProcessingTime: float
        Success: bool
    }

    type LLMAPICall = {
        Timestamp: DateTime
        AgentId: string
        Provider: string  // "OpenAI", "Anthropic", "Ollama", "Codestral"
        Model: string
        TokensInput: int
        TokensOutput: int
        Temperature: float
        MaxTokens: int
        ResponseTime: float
        Cost: float option
        Success: bool
    }

    type JanusResearchSession = {
        SessionId: string
        StartTime: DateTime
        EndTime: DateTime option
        ResearchObjective: string
        InitialArchitecture: TarsArchitectureSnapshot
        FinalArchitecture: TarsArchitectureSnapshot option
        AgentEvents: List<AgentEvent>
        InterAgentCommunications: List<InterAgentCommunication>
        GrammarEvolutions: List<GrammarEvolutionEvent>
        ArchitectureSnapshots: List<TarsArchitectureSnapshot>
        DataSourceAccesses: List<DataSourceAccess>
        WebRequests: List<WebRequest>
        TripleStoreQueries: List<TripleStoreQuery>
        VectorStoreOperations: List<VectorStoreOperation>
        LLMAPICalls: List<LLMAPICall>
        ResearchResults: Map<string, obj>
        QualityMetrics: Map<string, float>
    }

    // ============================================================================
    // TRACE CAPTURE ENGINE
    // ============================================================================

    type TraceCapture() =
        let mutable currentSession: JanusResearchSession option = None
        let mutable agentEvents: List<AgentEvent> = []
        let mutable communications: List<InterAgentCommunication> = []
        let mutable grammarEvolutions: List<GrammarEvolutionEvent> = []
        let mutable architectureSnapshots: List<TarsArchitectureSnapshot> = []
        let mutable dataSourceAccesses: List<DataSourceAccess> = []
        let mutable webRequests: List<WebRequest> = []
        let mutable tripleStoreQueries: List<TripleStoreQuery> = []
        let mutable vectorStoreOperations: List<VectorStoreOperation> = []
        let mutable llmAPICalls: List<LLMAPICall> = []

        member this.StartSession(objective: string) =
            let sessionId = sprintf "janus_research_%s" (DateTime.UtcNow.ToString("yyyyMMdd_HHmmss"))
            let initialSnapshot = this.CaptureArchitectureSnapshot(sessionId)
            
            currentSession <- Some {
                SessionId = sessionId
                StartTime = DateTime.UtcNow
                EndTime = None
                ResearchObjective = objective
                InitialArchitecture = initialSnapshot
                FinalArchitecture = None
                AgentEvents = []
                InterAgentCommunications = []
                GrammarEvolutions = []
                ArchitectureSnapshots = [initialSnapshot]
                DataSourceAccesses = []
                WebRequests = []
                TripleStoreQueries = []
                VectorStoreOperations = []
                LLMAPICalls = []
                ResearchResults = Map.empty
                QualityMetrics = Map.empty
            }
            
            printfn "🎬 AGENTIC TRACE CAPTURE STARTED"
            printfn "Session ID: %s" sessionId
            printfn "Objective: %s" objective
            printfn "Initial Architecture Captured"
            
            sessionId

        member this.LogAgentEvent(agentId: string, eventType: string, description: string, 
                                 inputData: Map<string, obj>, outputData: Map<string, obj>,
                                 processingTime: float, grammarTier: int, expressions: List<string>) =
            let event = {
                Timestamp = DateTime.UtcNow
                AgentId = agentId
                EventType = eventType
                Description = description
                InputData = inputData
                OutputData = outputData
                ProcessingTimeMs = processingTime
                MemoryUsageMB = GC.GetTotalMemory(false) / (1024L * 1024L) |> float
                GrammarTierUsed = grammarTier
                ComputationalExpressions = expressions
            }
            
            agentEvents <- event :: agentEvents
            printfn "📝 Agent Event: %s [%s] - %s (Tier %d)" agentId eventType description grammarTier

        member this.LogInterAgentCommunication(sourceAgent: string, targetAgent: string,
                                              messageType: string, payload: Map<string, obj>,
                                              responseTime: float, success: bool, error: string option) =
            let comm = {
                Timestamp = DateTime.UtcNow
                SourceAgent = sourceAgent
                TargetAgent = targetAgent
                MessageType = messageType
                Payload = payload
                ResponseTime = responseTime
                Success = success
                ErrorDetails = error
            }
            
            communications <- comm :: communications
            let status = if success then "✅" else "❌"
            printfn "📡 Inter-Agent: %s → %s [%s] %s (%.1fms)" sourceAgent targetAgent messageType status responseTime

        member this.LogGrammarEvolution(fromTier: int, toTier: int, trigger: string,
                                       capabilities: List<string>, impact: Map<string, float>,
                                       reasoning: string, code: List<string>) =
            let evolution = {
                Timestamp = DateTime.UtcNow
                FromTier = fromTier
                ToTier = toTier
                EvolutionTrigger = trigger
                NewCapabilities = capabilities
                PerformanceImpact = impact
                EvolutionReasoning = reasoning
                CodeGenerated = code
            }
            
            grammarEvolutions <- evolution :: grammarEvolutions
            printfn "🧬 Grammar Evolution: Tier %d → Tier %d (%s)" fromTier toTier trigger
            printfn "   New Capabilities: %d" capabilities.Length
            printfn "   Code Generated: %d blocks" code.Length

        member this.CaptureArchitectureSnapshot(sessionId: string) =
            {
                Timestamp = DateTime.UtcNow
                SessionId = sessionId
                ActiveTiers = [1; 2; 3; 4; 5; 6]  // Current active tiers
                LoadedModules = [
                    "TarsEngine.FSharp.Core"
                    "TarsEngine.Grammar"
                    "TarsEngine.Cosmology"
                    "TarsEngine.Tracing"
                    "TarsEngine.CUDA"
                ]
                AgentInstances = Map.ofList [
                    ("cosmologist_enhanced", "CosmologyAgent")
                    ("data_scientist_enhanced", "DataAnalysisAgent")
                    ("theoretical_physicist_enhanced", "PhysicsAgent")
                    ("grammar_evolution_agent", "GrammarAgent")
                ]
                GrammarCapabilities = Map.ofList [
                    (1, ["BasicExpressions"; "SimpleComputations"])
                    (2, ["ParameterBinding"; "TypeInference"])
                    (3, ["ComputationalExpressions"; "Monads"])
                    (4, ["AdvancedTypes"; "MetaProgramming"])
                    (5, ["CudaTranspilation"; "GPUAcceleration"])
                    (6, ["EmergentEvolution"; "SelfModification"])
                ]
                MemoryFootprint = Map.ofList [
                    ("HeapMemory", GC.GetTotalMemory(false) / (1024L * 1024L) |> float)
                    ("WorkingSet", Environment.WorkingSet / (1024L * 1024L) |> float)
                ]
                PerformanceMetrics = Map.ofList [
                    ("ProcessorTime", Environment.TickCount |> float)
                    ("ThreadCount", Environment.ProcessorCount |> float)
                ]
                ConfigurationState = Map.ofList [
                    ("MaxTiers", 16 :> obj)
                    ("TracingEnabled", true :> obj)
                    ("CudaEnabled", true :> obj)
                ]
            }

        member this.TakeArchitectureSnapshot() =
            match currentSession with
            | Some session ->
                let snapshot = this.CaptureArchitectureSnapshot(session.SessionId)
                architectureSnapshots <- snapshot :: architectureSnapshots
                printfn "📸 Architecture Snapshot Captured (Total: %d)" architectureSnapshots.Length
            | None ->
                printfn "⚠️ No active session for architecture snapshot"

        member this.LogDataSourceAccess(agentId: string, dataSourceType: string, url: string,
                                       queryParams: Map<string, obj>, dataSize: int64,
                                       processingTime: float, success: bool, error: string option) =
            let access = {
                Timestamp = DateTime.UtcNow
                AgentId = agentId
                DataSourceType = dataSourceType
                DataSourceUrl = url
                QueryParameters = queryParams
                DataRetrieved = dataSize
                ProcessingTime = processingTime
                Success = success
                ErrorDetails = error
            }
            dataSourceAccesses <- access :: dataSourceAccesses
            let status = if success then "✅" else "❌"
            printfn "🗄️ Data Source: %s accessed %s %s (%.1fMB, %.1fms)" agentId dataSourceType status (float dataSize / 1024.0 / 1024.0) processingTime

        member this.LogWebRequest(agentId: string, method: string, url: string,
                                 headers: Map<string, string>, requestBody: string option,
                                 responseCode: int, responseSize: int64, responseTime: float, success: bool, error: string option) =
            let request = {
                Timestamp = DateTime.UtcNow
                AgentId = agentId
                Method = method
                Url = url
                Headers = headers
                RequestBody = requestBody
                ResponseCode = responseCode
                ResponseSize = responseSize
                ResponseTime = responseTime
                Success = success
                ErrorMessage = error
            }
            webRequests <- request :: webRequests
            let status = if success then "✅" else "❌"
            printfn "🌐 Web Request: %s %s %s %s (%d, %.1fKB, %.1fms)" agentId method url status responseCode (float responseSize / 1024.0) responseTime

        member this.LogTripleStoreQuery(agentId: string, queryType: string, query: string,
                                       tripleStore: string, resultCount: int, queryTime: float,
                                       dataTransferred: int64, success: bool) =
            let tsQuery = {
                Timestamp = DateTime.UtcNow
                AgentId = agentId
                QueryType = queryType
                Query = query
                TripleStore = tripleStore
                ResultCount = resultCount
                QueryTime = queryTime
                DataTransferred = dataTransferred
                Success = success
            }
            tripleStoreQueries <- tsQuery :: tripleStoreQueries
            let status = if success then "✅" else "❌"
            printfn "🔗 Triple Store: %s queried %s %s (%d results, %.1fKB, %.1fms)" agentId tripleStore status resultCount (float dataTransferred / 1024.0) queryTime

        member this.LogVectorStoreOperation(agentId: string, operation: string, vectorStore: string,
                                           documentsProcessed: int, embeddingDims: int, similarityThreshold: float option,
                                           resultCount: int, processingTime: float, success: bool) =
            let vsOp = {
                Timestamp = DateTime.UtcNow
                AgentId = agentId
                Operation = operation
                VectorStore = vectorStore
                DocumentsProcessed = documentsProcessed
                EmbeddingDimensions = embeddingDims
                SimilarityThreshold = similarityThreshold
                ResultCount = resultCount
                ProcessingTime = processingTime
                Success = success
            }
            vectorStoreOperations <- vsOp :: vectorStoreOperations
            let status = if success then "✅" else "❌"
            printfn "🔍 Vector Store: %s %s in %s %s (%d docs, %d dims, %d results, %.1fms)" agentId operation vectorStore status documentsProcessed embeddingDims resultCount processingTime

        member this.LogLLMAPICall(agentId: string, provider: string, model: string,
                                 tokensInput: int, tokensOutput: int, temperature: float,
                                 maxTokens: int, responseTime: float, cost: float option, success: bool) =
            let llmCall = {
                Timestamp = DateTime.UtcNow
                AgentId = agentId
                Provider = provider
                Model = model
                TokensInput = tokensInput
                TokensOutput = tokensOutput
                Temperature = temperature
                MaxTokens = maxTokens
                ResponseTime = responseTime
                Cost = cost
                Success = success
            }
            llmAPICalls <- llmCall :: llmAPICalls
            let status = if success then "✅" else "❌"
            let costStr = cost |> Option.map (sprintf "$%.4f") |> Option.defaultValue "free"
            printfn "🤖 LLM API: %s called %s/%s %s (%d→%d tokens, %s, %.1fms)" agentId provider model status tokensInput tokensOutput costStr responseTime

        member this.EndSession(results: Map<string, obj>, metrics: Map<string, float>, ?generateWebUI: bool, ?openBrowser: bool) =
            let shouldGenerateWebUI = generateWebUI |> Option.defaultValue true
            let shouldOpenBrowser = openBrowser |> Option.defaultValue false
            match currentSession with
            | Some session ->
                let finalSnapshot = this.CaptureArchitectureSnapshot(session.SessionId)
                let completedSession = {
                    session with
                        EndTime = Some DateTime.UtcNow
                        FinalArchitecture = Some finalSnapshot
                        AgentEvents = List.rev agentEvents
                        InterAgentCommunications = List.rev communications
                        GrammarEvolutions = List.rev grammarEvolutions
                        ArchitectureSnapshots = List.rev (finalSnapshot :: architectureSnapshots)
                        DataSourceAccesses = List.rev dataSourceAccesses
                        WebRequests = List.rev webRequests
                        TripleStoreQueries = List.rev tripleStoreQueries
                        VectorStoreOperations = List.rev vectorStoreOperations
                        LLMAPICalls = List.rev llmAPICalls
                        ResearchResults = results
                        QualityMetrics = metrics
                }
                
                this.SaveSessionToFiles(completedSession)

                // Generate React web UI if requested (temporarily disabled)
                if false then // shouldGenerateWebUI then
                    let outputDir = sprintf "output/janus_traces/%s" session.SessionId
                    printfn "🎨 React Trace Viewer generation temporarily disabled"

                currentSession <- None
                agentEvents <- []
                communications <- []
                grammarEvolutions <- []
                architectureSnapshots <- []
                dataSourceAccesses <- []
                webRequests <- []
                tripleStoreQueries <- []
                vectorStoreOperations <- []
                llmAPICalls <- []

                printfn "🏁 AGENTIC TRACE CAPTURE COMPLETED"
                printfn "Session: %s" session.SessionId
                printfn "Duration: %.1f minutes" ((DateTime.UtcNow - session.StartTime).TotalMinutes)
                printfn "Events Captured: %d agent events, %d communications, %d evolutions"
                    completedSession.AgentEvents.Length completedSession.InterAgentCommunications.Length completedSession.GrammarEvolutions.Length

                session.SessionId
            | None ->
                printfn "⚠️ No active session to end"
                ""

        member this.SaveSessionToFiles(session: JanusResearchSession) =
            let outputDir = sprintf "output/janus_traces/%s" session.SessionId
            Directory.CreateDirectory(outputDir) |> ignore
            
            // Save main session file
            let sessionData = sprintf "Session: %s\nObjective: %s\nStartTime: %s\nEvents: %d\nCommunications: %d\nEvolutions: %d" session.SessionId session.ResearchObjective (session.StartTime.ToString()) session.AgentEvents.Length session.InterAgentCommunications.Length session.GrammarEvolutions.Length
            let sessionPath = outputDir + "/session.json"
            File.WriteAllText(sessionPath, sessionData)
            
            // Save detailed traces
            this.SaveAgentEventsTrace(outputDir, session.AgentEvents)
            this.SaveInterAgentTrace(outputDir, session.InterAgentCommunications)
            this.SaveGrammarEvolutionTrace(outputDir, session.GrammarEvolutions)
            this.SaveArchitectureTrace(outputDir, session.ArchitectureSnapshots)
            this.SaveDataSourceTrace(outputDir, session.DataSourceAccesses)
            this.SaveWebRequestTrace(outputDir, session.WebRequests)
            this.SaveTripleStoreTrace(outputDir, session.TripleStoreQueries)
            this.SaveVectorStoreTrace(outputDir, session.VectorStoreOperations)
            this.SaveLLMAPITrace(outputDir, session.LLMAPICalls)
            this.SaveResearchSummary(outputDir, session)
            
            printfn "💾 Session files saved to: %s" outputDir

        member this.SaveAgentEventsTrace(outputDir: string, events: List<AgentEvent>) =
            let yaml : string = this.ConvertAgentEventsToYaml(events)
            let filePath = outputDir + "/agent_events.yaml"
            File.WriteAllText(filePath, yaml)

        member this.SaveInterAgentTrace(outputDir: string, communications: List<InterAgentCommunication>) =
            let yaml : string = this.ConvertCommunicationsToYaml(communications)
            let filePath = outputDir + "/inter_agent_communications.yaml"
            File.WriteAllText(filePath, yaml)

        member this.SaveGrammarEvolutionTrace(outputDir: string, evolutions: List<GrammarEvolutionEvent>) =
            let yaml : string = this.ConvertEvolutionsToYaml(evolutions)
            let filePath = outputDir + "/grammar_evolutions.yaml"
            File.WriteAllText(filePath, yaml)

        member this.SaveArchitectureTrace(outputDir: string, snapshots: List<TarsArchitectureSnapshot>) =
            let yaml : string = this.ConvertArchitectureToYaml(snapshots)
            let filePath = outputDir + "/architecture_snapshots.yaml"
            File.WriteAllText(filePath, yaml)

        member this.SaveDataSourceTrace(outputDir: string, accesses: List<DataSourceAccess>) =
            let yaml : string = this.ConvertDataSourceToYaml(accesses)
            let filePath = outputDir + "/data_source_accesses.yaml"
            File.WriteAllText(filePath, yaml)

        member this.SaveWebRequestTrace(outputDir: string, requests: List<WebRequest>) =
            let yaml : string = this.ConvertWebRequestsToYaml(requests)
            let filePath = outputDir + "/web_requests.yaml"
            File.WriteAllText(filePath, yaml)

        member this.SaveTripleStoreTrace(outputDir: string, queries: List<TripleStoreQuery>) =
            let yaml : string = this.ConvertTripleStoreToYaml(queries)
            let filePath = outputDir + "/triple_store_queries.yaml"
            File.WriteAllText(filePath, yaml)

        member this.SaveVectorStoreTrace(outputDir: string, operations: List<VectorStoreOperation>) =
            let yaml : string = this.ConvertVectorStoreToYaml(operations)
            let filePath = outputDir + "/vector_store_operations.yaml"
            File.WriteAllText(filePath, yaml)

        member this.SaveLLMAPITrace(outputDir: string, calls: List<LLMAPICall>) =
            let yaml : string = this.ConvertLLMAPIToYaml(calls)
            let filePath = outputDir + "/llm_api_calls.yaml"
            File.WriteAllText(filePath, yaml)

        member this.SaveResearchSummary(outputDir: string, session: JanusResearchSession) =
            let endTimeStr = session.EndTime |> Option.map (fun t -> t.ToString("yyyy-MM-dd HH:mm:ss")) |> Option.defaultValue "In Progress"
            let durationMin = ((session.EndTime |> Option.defaultValue DateTime.UtcNow) - session.StartTime).TotalMinutes
            let initialTiers = String.concat ", " (session.InitialArchitecture.ActiveTiers |> List.map string)
            let finalTiers = session.FinalArchitecture |> Option.map (fun a -> String.concat ", " (a.ActiveTiers |> List.map string)) |> Option.defaultValue "N/A"
            let resultsStr = session.ResearchResults |> Map.toList |> List.map (fun (k, v) -> sprintf "- **%s**: %A" k v) |> String.concat "\n"
            let metricsStr = session.QualityMetrics |> Map.toList |> List.map (fun (k, v) -> sprintf "- **%s**: %.3f" k v) |> String.concat "\n"

            // Calculate external data statistics
            let totalDataMB = session.DataSourceAccesses |> List.sumBy (fun ds -> float ds.DataRetrieved / 1024.0 / 1024.0)
            let totalWebKB = session.WebRequests |> List.sumBy (fun wr -> float wr.ResponseSize / 1024.0)
            let totalLLMCost = session.LLMAPICalls |> List.sumBy (fun llm -> llm.Cost |> Option.defaultValue 0.0)
            let totalTripleResults = session.TripleStoreQueries |> List.sumBy (fun ts -> ts.ResultCount)
            let totalVectorDocs = session.VectorStoreOperations |> List.sumBy (fun vs -> vs.DocumentsProcessed)

            let summaryLines = [
                "# JANUS RESEARCH SESSION SUMMARY"
                ""
                sprintf "**Session ID**: `%s`" session.SessionId
                sprintf "**Objective**: %s" session.ResearchObjective
                sprintf "**Duration**: %s to %s (%.1f minutes)" (session.StartTime.ToString("yyyy-MM-dd HH:mm:ss")) endTimeStr durationMin
                ""
                "## 🏗️ Architecture Evolution"
                sprintf "- **Initial Tiers**: %s" initialTiers
                sprintf "- **Final Tiers**: %s" finalTiers
                sprintf "- **Grammar Evolutions**: %d" session.GrammarEvolutions.Length
                sprintf "- **Architecture Snapshots**: %d" session.ArchitectureSnapshots.Length
                ""
                "## 🤖 Agent Activity"
                sprintf "- **Total Agent Events**: %d" session.AgentEvents.Length
                sprintf "- **Inter-Agent Communications**: %d" session.InterAgentCommunications.Length
                sprintf "- **Collaborative Operations**: %d" (session.AgentEvents |> List.filter (fun e -> e.EventType = "CollaborativeAnalysis") |> List.length)
                ""
                "## 🌐 External Data Interactions"
                sprintf "- **Data Sources Accessed**: %d (%.1f MB total)" session.DataSourceAccesses.Length totalDataMB
                sprintf "- **Web Requests Made**: %d (%.1f KB total)" session.WebRequests.Length totalWebKB
                sprintf "- **Triple Store Queries**: %d (%d results total)" session.TripleStoreQueries.Length totalTripleResults
                sprintf "- **Vector Store Operations**: %d (%d documents processed)" session.VectorStoreOperations.Length totalVectorDocs
                sprintf "- **LLM API Calls**: %d ($%.4f total cost)" session.LLMAPICalls.Length totalLLMCost
                ""
                "## 📊 Research Results"
                resultsStr
                ""
                "## 📈 Quality Metrics"
                metricsStr
                ""
                "## 📁 Generated Files"
                "- `session.json` - Complete session metadata"
                "- `agent_events.yaml` - Detailed agent activity trace"
                "- `inter_agent_communications.yaml` - Agent communication logs"
                "- `grammar_evolutions.yaml` - Grammar tier evolution history"
                "- `architecture_snapshots.yaml` - System architecture evolution"
                "- `data_source_accesses.yaml` - External data source interactions"
                "- `web_requests.yaml` - HTTP request logs"
                "- `triple_store_queries.yaml` - Knowledge graph queries"
                "- `vector_store_operations.yaml` - Vector database operations"
                "- `llm_api_calls.yaml` - Language model API usage"
                "- `research_summary.md` - This comprehensive summary"
                ""
                "---"
                "*Generated by TARS Agentic Trace Capture System*"
                sprintf "*Session completed at %s*" (DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm:ss"))
            ]

            let summary = String.concat "\n" summaryLines

            let summaryPath = outputDir + "/research_summary.md"
            File.WriteAllText(summaryPath, summary)

        member this.ConvertAgentEventsToYaml(events: List<AgentEvent>) : string =
            let eventYamls = events |> List.mapi (fun i event ->
                sprintf "- event_%d: agent=%s type=%s tier=%d time=%.1fms" i event.AgentId event.EventType event.GrammarTierUsed event.ProcessingTimeMs
            )
            sprintf "agent_events:\n%s" (String.concat "\n" eventYamls)

        member this.ConvertCommunicationsToYaml(communications: List<InterAgentCommunication>) : string =
            let commYamls = communications |> List.mapi (fun i comm ->
                sprintf "- comm_%d: %s->%s type=%s time=%.1fms success=%b" i comm.SourceAgent comm.TargetAgent comm.MessageType comm.ResponseTime comm.Success
            )
            sprintf "inter_agent_communications:\n%s" (String.concat "\n" commYamls)

        member this.ConvertEvolutionsToYaml(evolutions: List<GrammarEvolutionEvent>) : string =
            let evolYamls = evolutions |> List.mapi (fun i evol ->
                sprintf "- evolution_%d: tier_%d->%d trigger=%s capabilities=%d" i evol.FromTier evol.ToTier evol.EvolutionTrigger evol.NewCapabilities.Length
            )
            sprintf "grammar_evolutions:\n%s" (String.concat "\n" evolYamls)

        member this.ConvertArchitectureToYaml(snapshots: List<TarsArchitectureSnapshot>) : string =
            let snapYamls = snapshots |> List.mapi (fun i snap ->
                sprintf "- snapshot_%d: tiers=%d modules=%d agents=%d memory=%.1fMB" i snap.ActiveTiers.Length snap.LoadedModules.Length snap.AgentInstances.Count (snap.MemoryFootprint |> Map.tryFind "HeapMemory" |> Option.defaultValue 0.0)
            )
            sprintf "architecture_snapshots:\n%s" (String.concat "\n" snapYamls)

        member this.ConvertDataSourceToYaml(accesses: List<DataSourceAccess>) : string =
            let accessYamls = accesses |> List.mapi (fun i access ->
                sprintf "- access_%d: agent=%s source=%s url=%s data=%.1fMB time=%.1fms success=%b" i access.AgentId access.DataSourceType access.DataSourceUrl (float access.DataRetrieved / 1024.0 / 1024.0) access.ProcessingTime access.Success
            )
            sprintf "data_source_accesses:\n%s" (String.concat "\n" accessYamls)

        member this.ConvertWebRequestsToYaml(requests: List<WebRequest>) : string =
            let requestYamls = requests |> List.mapi (fun i req ->
                sprintf "- request_%d: agent=%s method=%s url=%s code=%d size=%.1fKB time=%.1fms success=%b" i req.AgentId req.Method req.Url req.ResponseCode (float req.ResponseSize / 1024.0) req.ResponseTime req.Success
            )
            sprintf "web_requests:\n%s" (String.concat "\n" requestYamls)

        member this.ConvertTripleStoreToYaml(queries: List<TripleStoreQuery>) : string =
            let queryYamls = queries |> List.mapi (fun i query ->
                sprintf "- query_%d: agent=%s type=%s store=%s results=%d data=%.1fKB time=%.1fms success=%b" i query.AgentId query.QueryType query.TripleStore query.ResultCount (float query.DataTransferred / 1024.0) query.QueryTime query.Success
            )
            sprintf "triple_store_queries:\n%s" (String.concat "\n" queryYamls)

        member this.ConvertVectorStoreToYaml(operations: List<VectorStoreOperation>) : string =
            let opYamls = operations |> List.mapi (fun i op ->
                sprintf "- operation_%d: agent=%s op=%s store=%s docs=%d dims=%d results=%d time=%.1fms success=%b" i op.AgentId op.Operation op.VectorStore op.DocumentsProcessed op.EmbeddingDimensions op.ResultCount op.ProcessingTime op.Success
            )
            sprintf "vector_store_operations:\n%s" (String.concat "\n" opYamls)

        member this.ConvertLLMAPIToYaml(calls: List<LLMAPICall>) : string =
            let callYamls = calls |> List.mapi (fun i call ->
                let costStr = call.Cost |> Option.map (sprintf "%.4f") |> Option.defaultValue "0.0000"
                sprintf "- call_%d: agent=%s provider=%s model=%s tokens=%d->%d cost=$%s time=%.1fms success=%b" i call.AgentId call.Provider call.Model call.TokensInput call.TokensOutput costStr call.ResponseTime call.Success
            )
            sprintf "llm_api_calls:\n%s" (String.concat "\n" callYamls)

        // ============================================================================
        // REACT APP GENERATION (TEMPORARILY DISABLED)
        // ============================================================================

        (*
        member this.GenerateReactTraceViewer(sessionId: string, traceDir: string, openBrowser: bool) =
            let appDir = traceDir + "/trace_viewer"
            let srcDir = appDir + "/src"
            let publicDir = appDir + "/public"
            let traceDataDir = publicDir + "/trace_data"

            printfn "🚀 GENERATING REACT TRACE VIEWER APP"
            printfn "===================================="
            printfn "Session: %s" sessionId
            printfn "Output: %s" appDir
            printfn ""

            // Create directories
            Directory.CreateDirectory(appDir) |> ignore
            Directory.CreateDirectory(srcDir) |> ignore
            Directory.CreateDirectory(publicDir) |> ignore
            Directory.CreateDirectory(traceDataDir) |> ignore

            // Get list of trace files
            let traceFiles = Directory.GetFiles(traceDir, "*.*")
                            |> Array.filter (fun f -> not (f.Contains("trace_viewer")))
                            |> Array.map Path.GetFileName
                            |> Array.toList

            printfn "📁 Found %d trace files to include:" traceFiles.Length
            for file in traceFiles do
                printfn "  📄 %s" file

            // Copy trace files to public directory
            for file in traceFiles do
                let sourcePath = Path.Combine(traceDir, file)
                let destPath = Path.Combine(traceDataDir, file)
                File.Copy(sourcePath, destPath, true)

            // Generate React app files
            this.CreatePackageJson(appDir, sessionId)
            this.CreateIndexHtml(publicDir, sessionId)
            this.CreateAppJs(srcDir, sessionId, traceFiles)
            this.CreateAppCss(srcDir)
            this.CreateIndexJs(srcDir)

            printfn ""
            printfn "✅ React app generated successfully!"
            printfn "📦 Installing dependencies..."

            // Install npm dependencies
            let npmInstall = ProcessStartInfo(
                FileName = "npm",
                Arguments = "install",
                WorkingDirectory = appDir,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            )

            try
                use npmProcess = Process.Start(npmInstall)
                npmProcess.WaitForExit()

                if npmProcess.ExitCode = 0 then
                    printfn "✅ Dependencies installed successfully!"

                    if openBrowser then
                        printfn ""
                        printfn "🌐 Starting development server..."

                        // Start React development server
                        let npmStart = ProcessStartInfo(
                            FileName = "npm",
                            Arguments = "start",
                            WorkingDirectory = appDir,
                            UseShellExecute = true
                        )

                        Process.Start(npmStart) |> ignore
                        printfn "🚀 React app should open in your browser at http://localhost:3000"
                    else
                        printfn ""
                        printfn "📋 To start the app manually:"
                        printfn "   cd %s" appDir
                        printfn "   npm start"
                else
                    printfn "❌ Failed to install dependencies"

            with
            | ex -> printfn "❌ Error installing dependencies: %s" ex.Message

            appDir

        member this.CreatePackageJson(appDir: string, sessionId: string) =
            let packageJson = sprintf """{
  "name": "tars-trace-viewer-%s",
  "version": "1.0.0",
  "description": "TARS Research Trace Viewer - Generated by TARS Metascript System",
  "private": true,
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-scripts": "5.0.1",
    "@monaco-editor/react": "^4.6.0",
    "js-yaml": "^4.1.0",
    "react-markdown": "^9.0.1",
    "lucide-react": "^0.263.1"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject"
  },
  "eslintConfig": {
    "extends": [
      "react-app",
      "react-app/jest"
    ]
  },
  "browserslist": {
    "production": [
      ">0.2%%",
      "not dead",
      "not op_mini all"
    ],
    "development": [
      "last 1 chrome version",
      "last 1 firefox version",
      "last 1 safari version"
    ]
  }
}""" sessionId
            File.WriteAllText(Path.Combine(appDir, "package.json"), packageJson)
            ()
            ()

        member this.CreateIndexHtml(publicDir: string, sessionId: string) =
            let indexHtml = sprintf """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <link rel="icon" href="%%PUBLIC_URL%%/favicon.ico" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="theme-color" content="#000000" />
    <meta name="description" content="TARS Research Trace Viewer for session %s" />
    <title>TARS Trace Viewer - %s</title>
    <style>
      body {
        margin: 0;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
          'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
          sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        background: linear-gradient(135deg, #667eea 0%%, #764ba2 100%%);
        min-height: 100vh;
      }
      code {
        font-family: source-code-pro, Menlo, Monaco, Consolas, 'Courier New', monospace;
      }
    </style>
  </head>
  <body>
    <noscript>You need to enable JavaScript to run this app.</noscript>
    <div id="root"></div>
  </body>
</html>""" sessionId sessionId
            File.WriteAllText(Path.Combine(publicDir, "index.html"), indexHtml)

        member this.CreateIndexJs(srcDir: string) =
            let indexJs = """import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);"""
            File.WriteAllText(Path.Combine(srcDir, "index.js"), indexJs)

        member this.CreateAppJs(srcDir: string, sessionId: string, traceFiles: string list) =
            let fileList = traceFiles |> List.map (fun f -> sprintf "\"%s\"" f) |> String.concat ", "
            let appJs = sprintf """import React, { useState, useEffect } from 'react';
import Editor from '@monaco-editor/react';
import ReactMarkdown from 'react-markdown';
import yaml from 'js-yaml';
import { FileText, Database, Globe, Brain, Zap, BarChart3, Settings, Download } from 'lucide-react';
import './App.css';

const TRACE_FILES = [%s];

const FILE_ICONS = {
  'research_summary.md': FileText,
  'agent_events.yaml': Brain,
  'inter_agent_communications.yaml': Zap,
  'data_source_accesses.yaml': Database,
  'web_requests.yaml': Globe,
  'triple_store_queries.yaml': Database,
  'vector_store_operations.yaml': Brain,
  'llm_api_calls.yaml': Brain,
  'architecture_snapshots.yaml': Settings,
  'grammar_evolutions.yaml': BarChart3,
  'session.json': FileText
};

function App() {
  const [selectedFile, setSelectedFile] = useState('research_summary.md');
  const [fileContent, setFileContent] = useState('');
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState({});

  useEffect(() => {
    loadFile(selectedFile);
    calculateStats();
  }, [selectedFile]);

  const loadFile = async (filename) => {
    setLoading(true);
    try {
      const response = await fetch(`/trace_data/${filename}`);
      const content = await response.text();
      setFileContent(content);
    } catch (error) {
      setFileContent(`Error loading file: ${error.message}`);
    }
    setLoading(false);
  };

  const calculateStats = async () => {
    try {
      const agentEvents = await fetch('/trace_data/agent_events.yaml').then(r => r.text());
      const communications = await fetch('/trace_data/inter_agent_communications.yaml').then(r => r.text());
      const dataAccess = await fetch('/trace_data/data_source_accesses.yaml').then(r => r.text());
      const webRequests = await fetch('/trace_data/web_requests.yaml').then(r => r.text());
      const llmCalls = await fetch('/trace_data/llm_api_calls.yaml').then(r => r.text());

      const agentData = yaml.load(agentEvents);
      const commData = yaml.load(communications);
      const dataData = yaml.load(dataAccess);
      const webData = yaml.load(webRequests);
      const llmData = yaml.load(llmCalls);

      setStats({
        totalEvents: agentData?.agent_events?.length || 0,
        totalCommunications: commData?.inter_agent_communications?.length || 0,
        totalDataSources: dataData?.data_source_accesses?.length || 0,
        totalWebRequests: webData?.web_requests?.length || 0,
        totalLLMCalls: llmData?.llm_api_calls?.length || 0
      });
    } catch (error) {
      console.error('Error calculating stats:', error);
    }
  };

  const getFileLanguage = (filename) => {
    if (filename.endsWith('.md')) return 'markdown';
    if (filename.endsWith('.yaml')) return 'yaml';
    if (filename.endsWith('.json')) return 'json';
    return 'text';
  };

  const renderContent = () => {
    if (loading) {
      return <div className="loading">Loading...</div>;
    }

    if (selectedFile.endsWith('.md')) {
      return (
        <div className="markdown-content">
          <ReactMarkdown>{fileContent}</ReactMarkdown>
        </div>
      );
    }

    return (
      <Editor
        height="100%%"
        language={getFileLanguage(selectedFile)}
        value={fileContent}
        theme="vs-dark"
        options={{
          readOnly: true,
          minimap: { enabled: false },
          scrollBeyondLastLine: false,
          fontSize: 14,
          lineNumbers: 'on',
          wordWrap: 'on'
        }}
      />
    );
  };

  const IconComponent = FILE_ICONS[selectedFile] || FileText;

  return (
    <div className="App">
      <header className="app-header">
        <div className="header-content">
          <h1>🌌 TARS Research Trace Viewer</h1>
          <p>Session: <code>%s</code></p>
          <div className="stats">
            <span>📊 {stats.totalEvents} Events</span>
            <span>📡 {stats.totalCommunications} Communications</span>
            <span>🗄️ {stats.totalDataSources} Data Sources</span>
            <span>🌐 {stats.totalWebRequests} Web Requests</span>
            <span>🤖 {stats.totalLLMCalls} LLM Calls</span>
          </div>
        </div>
      </header>

      <div className="app-body">
        <aside className="sidebar">
          <h3>📁 Trace Files</h3>
          <ul className="file-list">
            {TRACE_FILES.map(file => {
              const Icon = FILE_ICONS[file] || FileText;
              return (
                <li
                  key={file}
                  className={selectedFile === file ? 'active' : ''}
                  onClick={() => setSelectedFile(file)}
                >
                  <Icon size={16} />
                  <span>{file}</span>
                </li>
              );
            })}
          </ul>
        </aside>

        <main className="content">
          <div className="content-header">
            <div className="file-info">
              <IconComponent size={20} />
              <h2>{selectedFile}</h2>
            </div>
            <button
              className="download-btn"
              onClick={() => {
                const blob = new Blob([fileContent], { type: 'text/plain' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = selectedFile;
                a.click();
                URL.revokeObjectURL(url);
              }}
            >
              <Download size={16} />
              Download
            </button>
          </div>
          <div className="content-body">
            {renderContent()}
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;""" fileList sessionId
            File.WriteAllText(Path.Combine(srcDir, "App.js"), appJs)

        member this.CreateAppCss(srcDir: string) =
            let appCss = """.App {
  height: 100vh;
  display: flex;
  flex-direction: column;
  background: #1a1a1a;
  color: #ffffff;
}

.app-header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 1rem 2rem;
  box-shadow: 0 2px 10px rgba(0,0,0,0.3);
}

.header-content h1 {
  margin: 0 0 0.5rem 0;
  font-size: 2rem;
  font-weight: 700;
}

.header-content p {
  margin: 0 0 1rem 0;
  opacity: 0.9;
}

.stats {
  display: flex;
  gap: 1.5rem;
  flex-wrap: wrap;
}

.stats span {
  background: rgba(255,255,255,0.2);
  padding: 0.25rem 0.75rem;
  border-radius: 20px;
  font-size: 0.9rem;
  backdrop-filter: blur(10px);
}

.app-body {
  display: flex;
  flex: 1;
  overflow: hidden;
}

.sidebar {
  width: 300px;
  background: #2d2d2d;
  border-right: 1px solid #404040;
  padding: 1rem;
  overflow-y: auto;
}

.sidebar h3 {
  margin: 0 0 1rem 0;
  color: #ffffff;
  font-size: 1.1rem;
}

.file-list {
  list-style: none;
  padding: 0;
  margin: 0;
}

.file-list li {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.75rem;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s ease;
  margin-bottom: 0.25rem;
}

.file-list li:hover {
  background: #404040;
}

.file-list li.active {
  background: #667eea;
  color: #ffffff;
}

.file-list li span {
  font-size: 0.9rem;
  word-break: break-all;
}

.content {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.content-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem 1.5rem;
  background: #333333;
  border-bottom: 1px solid #404040;
}

.file-info {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.file-info h2 {
  margin: 0;
  font-size: 1.2rem;
  color: #ffffff;
}

.download-btn {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  background: #667eea;
  color: white;
  border: none;
  padding: 0.5rem 1rem;
  border-radius: 6px;
  cursor: pointer;
  font-size: 0.9rem;
  transition: background 0.2s ease;
}

.download-btn:hover {
  background: #5a6fd8;
}

.content-body {
  flex: 1;
  overflow: hidden;
}

.markdown-content {
  padding: 2rem;
  max-width: none;
  background: #ffffff;
  color: #333333;
  height: 100%;
  overflow-y: auto;
}

.markdown-content h1 {
  color: #2d3748;
  border-bottom: 2px solid #667eea;
  padding-bottom: 0.5rem;
}

.markdown-content h2 {
  color: #4a5568;
  margin-top: 2rem;
}

.markdown-content code {
  background: #f7fafc;
  padding: 0.2rem 0.4rem;
  border-radius: 4px;
  font-size: 0.9em;
}

.markdown-content pre {
  background: #2d3748;
  color: #ffffff;
  padding: 1rem;
  border-radius: 8px;
  overflow-x: auto;
}

.markdown-content ul {
  padding-left: 1.5rem;
}

.markdown-content li {
  margin-bottom: 0.5rem;
}

.loading {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100%;
  font-size: 1.2rem;
  color: #667eea;
}

@media (max-width: 768px) {
  .app-body {
    flex-direction: column;
  }

  .sidebar {
    width: 100%;
    height: 200px;
  }

  .stats {
    font-size: 0.8rem;
  }

  .header-content h1 {
    font-size: 1.5rem;
  }
}"""
            File.WriteAllText(Path.Combine(srcDir, "App.css"), appCss)
        *)

    // ============================================================================
    // GLOBAL TRACE INSTANCE
    // ============================================================================

    let GlobalTraceCapture = TraceCapture()
