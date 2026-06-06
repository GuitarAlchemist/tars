module Tars.Evolution.ResearchEnhancedCurriculum

open System
open System.Threading.Tasks
open Tars.Llm
open Tars.Llm.LlmService
open Tars.Core
open Tars.Tools.Research

/// Research-enhanced curriculum generation
/// Uses arXiv/web research to generate more innovative tasks

type ResearchInsight = {
    Topic: string
    Findings: string list
    Contradictions: string list
    FetchedAt: DateTime
}

type ResearchCache = {
    mutable Insights: Map<string, ResearchInsight>
    mutable LastUpdate: DateTime
}

open Tars.Knowledge

let private cache = { Insights = Map.empty; LastUpdate = DateTime.MinValue }
let private cacheDurationHours = 24.0

/// Fetch research insights on a topic, with caching (RAM + Persistent Ledger)
let fetchResearchInsightsAsync (llm: ILlmService) (ledger: KnowledgeLedger option) (topic: string) : Task<ResearchInsight option> =
    task {
        let cacheKey = topic.ToLowerInvariant().Trim()
        
        // 1. Check RAM Cache
        match Map.tryFind cacheKey cache.Insights with
        | Some insight when (DateTime.UtcNow - insight.FetchedAt).TotalHours < cacheDurationHours ->
            return Some insight
        | _ ->
            // 2. Check Persistent Ledger
            let! ledgerInsight =
                match ledger with
                | Some l ->
                    task {
                        let existing = 
                            l.Query(subject = topic, predicate = RelationType.Custom "has_insight")
                            |> Seq.toList
                        
                        if existing.IsEmpty then
                            return None
                        else
                            // Reconstruct insight from beliefs
                            let findings = existing |> List.map (fun b -> b.Object.Value)
                            // We assume last check was recent enough if found in ledger
                            return Some {
                                Topic = topic
                                Findings = findings
                                Contradictions = []
                                FetchedAt = DateTime.UtcNow
                            }
                    }
                | None -> Task.FromResult None
            
            match ledgerInsight with
            | Some i -> 
                // Update RAM cache
                cache.Insights <- Map.add cacheKey i cache.Insights
                return Some i
            | None ->
                try
                    // 3. Fetch from arXiv
                    let args = sprintf """{"query": "%s"}""" topic
                    let! result = ResearchTools.fetchArxiv args
                    
                    if String.IsNullOrWhiteSpace(result) || result.Contains("error") then
                        return None
                    else
                        // Parse papers
                        let lines = result.Split('\n') |> Array.filter (fun l -> l.Contains("**["))
                        let papers = lines |> Array.truncate 3 |> Array.toList
                        
                        if papers.IsEmpty then
                            return None
                        else
                            // Analyze papers with LLM
                            let papersText = papers |> String.concat "\n\n"
                            let prompt = sprintf """Analyze these arXiv papers and extract 3 key insights:

%s

Format:
INSIGHT: [key finding 1]
INSIGHT: [key finding 2]
INSIGHT: [key finding 3]
CONTRADICTION: [if any contradictions between papers]""" papersText

                            let request = {
                                LlmRequest.Default with
                                    ModelHint = Some "reasoning"
                                    SystemPrompt = Some "You are a research analyst extracting actionable insights."
                                    Messages = [{ Role = Role.User; Content = prompt }]
                                    Temperature = Some 0.3
                                    MaxTokens = Some 500
                            }
                            
                            let! response = llm.CompleteAsync request
                            let text = response.Text
                            
                            // Parse insights
                            let insightPattern = @"INSIGHT:\s*(.+?)(?=INSIGHT:|CONTRADICTION:|$)"
                            let insights =
                                System.Text.RegularExpressions.Regex.Matches(text, insightPattern, 
                                    System.Text.RegularExpressions.RegexOptions.Singleline)
                                |> Seq.cast<System.Text.RegularExpressions.Match>
                                |> Seq.map (fun m -> m.Groups.[1].Value.Trim())
                                |> Seq.filter (fun s -> s.Length > 15 && s.Length < 500)
                                |> Seq.toList
                            
                            let contradictionPattern = @"CONTRADICTION:\s*(.+?)(?=INSIGHT:|CONTRADICTION:|$)"
                            let contradictions =
                                System.Text.RegularExpressions.Regex.Matches(text, contradictionPattern,
                                    System.Text.RegularExpressions.RegexOptions.Singleline)
                                |> Seq.cast<System.Text.RegularExpressions.Match>
                                |> Seq.map (fun m -> m.Groups.[1].Value.Trim())
                                |> Seq.filter (fun s -> s.Length > 10)
                                |> Seq.toList
                            
                            let insight = {
                                Topic = topic
                                Findings = insights
                                Contradictions = contradictions
                                FetchedAt = DateTime.UtcNow
                            }
                            
                            // 4. Update RAM Cache
                            cache.Insights <- Map.add cacheKey insight cache.Insights
                            cache.LastUpdate <- DateTime.UtcNow
                            
                            // 5. Persist to Ledger
                            match ledger with
                            | Some l ->
                                for finding in insights do
                                    // Use SafeAssertTriple indirectly via Assert since we want simple construction
                                    // Or construct manual belief
                                    let provenance = Provenance.FromExternal(Uri("https://arxiv.org/search?q=" + topic), None, 0.9)
                                    
                                    let belief = 
                                        { Belief.create topic (RelationType.Custom "has_insight") finding provenance with
                                            Confidence = 0.9
                                            Tags = ["research"; "auto-generated"] }
                                          
                                    l.Assert(belief, AgentId.System) |> ignore
                            | None -> ()

                            return Some insight
                with _ ->
                    return None
    }

/// Generate research-enhanced curriculum guidance
let enhanceCurriculumGuidanceAsync (llm: ILlmService) (ledger: KnowledgeLedger option) (baseGuidance: string) (recentTasks: string list) : Task<string> =
    task {
        // Extract keywords from recent tasks to find relevant research
        let keywords = 
            recentTasks
            |> List.truncate 5
            |> String.concat " "
            |> fun s -> s.ToLowerInvariant()
        
        // Determine research topics based on task domain
        // Strategy: Combine domain with specialized context for better relevancy
        let domain, context =
            if keywords.Contains("neural") || keywords.Contains("attention") || keywords.Contains("imagination") then
                "transformer neural networks", "neuro-symbolic reasoning"
            elif keywords.Contains("graph") || keywords.Contains("node") || keywords.Contains("identity") then
                "knowledge graphs", "temporal reasoning"
            elif keywords.Contains("refactor") || keywords.Contains("code") || keywords.Contains("clean") then
                "automated code refactoring", "static analysis"
            elif keywords.Contains("test") || keywords.Contains("coverage") || keywords.Contains("correctness") then
                "formal verification", "automated software testing"
            elif keywords.Contains("optimization") || keywords.Contains("performance") || keywords.Contains("speed") then
                "compiler optimization", "high-performance computing"
            elif keywords.Contains("agent") || keywords.Contains("autonomous") || keywords.Contains("governance") then
                "AI agent constitutions", "autonomous reasoning"
            else
                "F# functional programming", "meta-programming"
        
        let researchTopic = sprintf "%s %s" domain context
        
        let! insightOpt = fetchResearchInsightsAsync llm ledger researchTopic
        
        match insightOpt with
        | None ->
            return baseGuidance
        | Some insight when insight.Findings.IsEmpty ->
            return baseGuidance
        | Some insight ->
            // Inject research insights into guidance
            let researchAddendum =
                let findings = insight.Findings |> List.truncate 2 |> String.concat "; "
                sprintf """

RESEARCH INSIGHTS (from recent %s papers):
%s

Consider generating tasks that apply these cutting-edge techniques to real-world problems."""
                    insight.Topic findings
            
            return baseGuidance + researchAddendum
    }

/// Suggest research-inspired tasks
let suggestResearchTasksAsync (llm: ILlmService) (ledger: KnowledgeLedger option) (currentGeneration: int) : Task<string list> =
    task {
        // Topics relevant to TARS development
        let topics = [
            "LLM agent architectures"
            "neuro-symbolic reasoning"
            "automated code repair"
        ]
        
        let! insights =
            topics
            |> List.map (fetchResearchInsightsAsync llm ledger)
            |> Task.WhenAll
        
        let allFindings =
            insights
            |> Array.choose id
            |> Array.collect (fun i -> i.Findings |> List.toArray)
            |> Array.distinct
            |> Array.truncate 5
        
        if Array.isEmpty allFindings then
            return []
        else
            // Generate task suggestions based on research
            let findingsText = allFindings |> Array.mapi (fun i f -> sprintf "%d. %s" (i+1) f) |> String.concat "\n"
            
            let prompt = sprintf """Based on these research insights:

%s

Suggest 2 concrete F# programming tasks that apply these ideas.
Each task should be a self-contained coding challenge.

Format:
TASK: [description of task 1]
TASK: [description of task 2]""" findingsText

            let request = {
                LlmRequest.Default with
                    ModelHint = Some "reasoning"
                    SystemPrompt = Some "You are generating practical coding tasks based on research findings."
                    Messages = [{ Role = Role.User; Content = prompt }]
                    Temperature = Some 0.5
                    MaxTokens = Some 400
            }
            
            let! response = llm.CompleteAsync request
            
            // Parse suggested tasks
            let taskPattern = @"TASK:\s*(.+?)(?=TASK:|$)"
            let tasks =
                System.Text.RegularExpressions.Regex.Matches(response.Text, taskPattern,
                    System.Text.RegularExpressions.RegexOptions.Singleline)
                |> Seq.cast<System.Text.RegularExpressions.Match>
                |> Seq.map (fun m -> m.Groups.[1].Value.Trim().Replace("\"", "'"))
                |> Seq.filter (fun s -> s.Length > 20)
                |> Seq.toList
            
            return tasks
    }

/// Clear the research cache (for testing or forced refresh)
let clearCache () =
    cache.Insights <- Map.empty
    cache.LastUpdate <- DateTime.MinValue
