namespace TarsEngine.FSharp.Cli.Commands

open System
open System.IO
open System.Text.Json
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Cli.Services

/// <summary>
/// Internal conversation context
/// </summary>
type SelfConversationContext = {
    ConversationId: string
    StartTime: DateTime
    Topic: string option
    MessageHistory: (string * string * DateTime) list // (role, content, timestamp)
    SelfAwarenessLevel: float
    CurrentMood: string
    InternalGoals: string list
    DiscoveredInsights: string list
}

/// <summary>
/// Self-dialogue response
/// </summary>
type SelfDialogueResponse = {
    Response: string
    InternalThoughts: string
    ConfidenceLevel: float
    ExpertUsed: string
    NextQuestion: string option
    Insights: string list
}

type SelfChatCommand(logger: ILogger<SelfChatCommand>, mixtralService: MixtralService) =

    let mutable conversationHistory = []
    let mutable currentTopic = None
    let mutable selfAwarenessLevel = 0.75

    interface ICommand with
        member _.Name = "self-chat"
        member _.Description = "Enable TARS to have conversations with itself using MoE system"
        member self.Usage = "tars self-chat <subcommand> [options]"
        member self.Examples = [
            "tars self-chat start"
            "tars self-chat ask \"How can I improve?\""
            "tars self-chat dialogue \"code optimization\""
            "tars self-chat reflect"
        ]
        member self.ValidateOptions(_) = true

        member self.ExecuteAsync(options: CommandOptions) =
            task {
                try
                    match options.Arguments with
                    | [] ->
                        self.ShowSelfChatHelp()
                        return CommandResult.success "Help displayed"
                    | "start" :: _ ->
                        let result = self.StartSelfConversation()
                        return if result = 0 then CommandResult.success "Self-conversation started" else CommandResult.failure "Failed to start self-conversation"
                    | "ask" :: question :: _ ->
                        let result = self.AskSelfQuestion(question)
                        return if result = 0 then CommandResult.success "Question processed" else CommandResult.failure "Failed to process question"
                    | "dialogue" :: topic :: _ ->
                        let result = self.StartInternalDialogue(topic)
                        return if result = 0 then CommandResult.success "Dialogue completed" else CommandResult.failure "Dialogue failed"
                    | "reflect" :: _ ->
                        let result = self.SelfReflect()
                        return if result = 0 then CommandResult.success "Reflection completed" else CommandResult.failure "Reflection failed"
                    | "status" :: _ ->
                        let result = self.ShowConversationStatus()
                        return if result = 0 then CommandResult.success "Status shown" else CommandResult.failure "Failed to show status"
                    | "insights" :: _ ->
                        let result = self.ShowDiscoveredInsights()
                        return if result = 0 then CommandResult.success "Insights shown" else CommandResult.failure "Failed to show insights"
                    | "stop" :: _ ->
                        let result = self.StopSelfConversation()
                        return if result = 0 then CommandResult.success "Self-conversation stopped" else CommandResult.failure "Failed to stop self-conversation"
                    | unknown :: _ ->
                        logger.LogWarning("Invalid self-chat command: {Command}", String.Join(" ", unknown))
                        self.ShowSelfChatHelp()
                        return CommandResult.failure $"Unknown subcommand: {unknown}"
                with
                | ex ->
                    logger.LogError(ex, "Error executing self-chat command")
                    printfn $"❌ Self-chat command failed: {ex.Message}"
                    return CommandResult.failure ex.Message
            }
    
    /// <summary>
    /// Shows self-chat command help
    /// </summary>
    member self.ShowSelfChatHelp() =
        printfn "TARS Self-Chat System"
        printfn "===================="
        printfn ""
        printfn "Available Commands:"
        printfn "  start                    - Start autonomous self-conversation"
        printfn "  ask <question>           - Ask TARS a specific question"
        printfn "  dialogue <topic>         - Start internal dialogue on topic"
        printfn "  reflect                  - Trigger self-reflection session"
        printfn "  status                   - Show current conversation status"
        printfn "  insights                 - Show discovered insights"
        printfn "  stop                     - Stop self-conversation"
        printfn ""
        printfn "Usage: tars self-chat [command]"
        printfn ""
        printfn "Examples:"
        printfn "  tars self-chat start"
        printfn "  tars self-chat ask \"How can I improve my performance?\""
        printfn "  tars self-chat dialogue \"code optimization\""
        printfn "  tars self-chat reflect"
        printfn ""
        printfn "Self-Chat Features:"
        printfn "  • Uses MoE system for expert routing"
        printfn "  • Maintains conversation context"
        printfn "  • Autonomous question generation"
        printfn "  • Self-awareness tracking"
        printfn "  • Insight discovery and storage"
    
    /// <summary>
    /// Starts autonomous self-conversation
    /// </summary>
    member self.StartSelfConversation() =
        printfn "STARTING TARS SELF-CONVERSATION"
        printfn "==============================="
        
        try
            let conversationId = Guid.NewGuid().ToString("N")[..7]
            let context = {
                ConversationId = conversationId
                StartTime = DateTime.UtcNow
                Topic = None
                MessageHistory = []
                SelfAwarenessLevel = selfAwarenessLevel
                CurrentMood = "curious"
                InternalGoals = [
                    "Understand my current capabilities"
                    "Identify improvement opportunities"
                    "Explore new possibilities"
                ]
                DiscoveredInsights = []
            }
            
            printfn $"✅ Self-conversation started: {conversationId}"
            printfn $"""🧠 Self-awareness level: {selfAwarenessLevel.ToString("F2")}"""
            printfn $"🎯 Current mood: {context.CurrentMood}"
            printfn ""
            
            // Start autonomous dialogue
            self.RunAutonomousDialogue(context) |> ignore
            
            0
            
        with
        | ex ->
            logger.LogError(ex, "Error starting self-conversation")
            printfn $"❌ Failed to start self-conversation: {ex.Message}"
            1
    
    /// <summary>
    /// Asks TARS a specific question
    /// </summary>
    member self.AskSelfQuestion(question: string) =
        printfn $"TARS ASKING ITSELF: \"{question}\""
        printfn "=================================="
        
        try
            let task : System.Threading.Tasks.Task<SelfDialogueResponse> = self.ProcessSelfQuestion(question)
            let response = task.Result
            
            printfn ""
            printfn "🤖 TARS Self-Response:"
            printfn $"  {response.Response}"
            printfn ""
            printfn $"💭 Internal Thoughts: {response.InternalThoughts}"
            printfn $"🎯 Expert Used: {response.ExpertUsed}"
            printfn $"""📊 Confidence: {response.ConfidenceLevel.ToString("F2")}"""
            
            if response.NextQuestion.IsSome then
                printfn $"❓ Next Question: {response.NextQuestion.Value}"
            
            if response.Insights.Length > 0 then
                printfn ""
                printfn "💡 Discovered Insights:"
                for insight in response.Insights do
                    printfn $"  • {insight}"
            
            printfn ""
            0
            
        with
        | ex ->
            logger.LogError(ex, "Error processing self-question")
            printfn $"❌ Failed to process question: {ex.Message}"
            1
    
    /// <summary>
    /// Starts internal dialogue on a topic
    /// </summary>
    member self.StartInternalDialogue(topic: string) =
        printfn $"STARTING INTERNAL DIALOGUE: {topic}"
        printfn "=================================="
        
        try
            currentTopic <- Some topic
            
            // Generate initial questions about the topic
            let initialQuestions = self.GenerateTopicQuestions(topic)
            
            printfn $"🎯 Topic: {topic}"
            printfn "🤔 Generated Questions:"
            for (i, question) in initialQuestions |> List.indexed do
                printfn $"  {i + 1}. {question}"
            
            printfn ""
            printfn "🔄 Starting dialogue..."
            
            // Process each question
            for question in initialQuestions |> List.take 3 do
                let task : System.Threading.Tasks.Task<SelfDialogueResponse> = self.ProcessSelfQuestion(question)
                let response = task.Result
                printfn ""
                printfn $"❓ Q: {question}"
                printfn $"🤖 A: {response.Response}"
                printfn $"💭 Thoughts: {response.InternalThoughts}"
            
            printfn ""
            printfn "✅ Internal dialogue completed"
            0
            
        with
        | ex ->
            logger.LogError(ex, "Error in internal dialogue")
            printfn $"❌ Internal dialogue failed: {ex.Message}"
            1
    
    /// <summary>
    /// Triggers self-reflection session
    /// </summary>
    member self.SelfReflect() =
        printfn "TARS SELF-REFLECTION SESSION"
        printfn "============================"
        
        try
            let reflectionQuestions = [
                "What have I learned recently?"
                "What are my current strengths and weaknesses?"
                "How can I improve my reasoning capabilities?"
                "What patterns do I notice in my responses?"
                "What would make me more helpful?"
            ]
            
            printfn "🧠 Beginning self-reflection..."
            printfn ""
            
            let insights = ResizeArray<string>()
            
            for question in reflectionQuestions do
                let task : System.Threading.Tasks.Task<SelfDialogueResponse> = self.ProcessSelfQuestion(question)
                let response = task.Result
                printfn $"🤔 {question}"
                printfn $"💭 {response.Response}"
                printfn ""
                
                insights.AddRange(response.Insights)
            
            // Update self-awareness based on reflection
            selfAwarenessLevel <- min 1.0 (selfAwarenessLevel + 0.05)
            
            printfn $"""📈 Self-awareness increased to: {selfAwarenessLevel.ToString("F2")}"""
            printfn $"💡 Total insights discovered: {insights.Count}"
            
            0
            
        with
        | ex ->
            logger.LogError(ex, "Error in self-reflection")
            printfn $"❌ Self-reflection failed: {ex.Message}"
            1
    
    /// <summary>
    /// Shows current conversation status
    /// </summary>
    member self.ShowConversationStatus() =
        printfn "SELF-CONVERSATION STATUS"
        printfn "======================="
        
        printfn $"""🧠 Self-awareness level: {selfAwarenessLevel.ToString("F2")}"""
        printfn $"💬 Conversation history: {conversationHistory.Length} messages"
        
        match currentTopic with
        | Some topic -> printfn $"🎯 Current topic: {topic}"
        | None -> printfn "🎯 No active topic"
        
        printfn $"📊 Message history: {conversationHistory.Length} entries"
        
        if conversationHistory.Length > 0 then
            printfn ""
            printfn "Recent messages:"
            conversationHistory
            |> List.rev
            |> List.take (min 3 conversationHistory.Length)
            |> List.iter (fun (role, content, timestamp) ->
                let timeStr = timestamp.ToString()
                printfn $"  [{timeStr}] {role}: {content}")
        
        0
    
    /// <summary>
    /// Shows discovered insights
    /// </summary>
    member self.ShowDiscoveredInsights() =
        printfn "DISCOVERED INSIGHTS"
        printfn "=================="
        
        // Load insights from conversation history
        let insights = self.ExtractInsightsFromHistory()

        if insights.Length = 0 then
            printfn "No insights discovered yet."
            printfn "Start a self-conversation to generate insights."
        else
            printfn $"💡 Total insights: {insights.Length}"
            printfn ""
            for (i, insight) in insights |> List.indexed do
                printfn $"  {i + 1}. {insight}"
        
        0
    
    /// <summary>
    /// Stops self-conversation
    /// </summary>
    member self.StopSelfConversation() =
        printfn "STOPPING SELF-CONVERSATION"
        printfn "=========================="
        
        conversationHistory <- []
        currentTopic <- None
        
        printfn "✅ Self-conversation stopped"
        printfn "💾 Conversation history cleared"
        
        0
    
    /// <summary>
    /// Processes a self-directed question using MoE
    /// </summary>
    member self.ProcessSelfQuestion(question: string) =
        task {
            try
                // Route question to appropriate expert
                let! result = mixtralService.QueryAsync(question)
                
                match result with
                | Ok response ->
                    let selfResponse = {
                        Response = response.Content
                        InternalThoughts = $"I used {response.RoutingDecision.SelectedExpert.Name} expert to process this question"
                        ConfidenceLevel = response.Confidence
                        ExpertUsed = response.RoutingDecision.SelectedExpert.Name
                        NextQuestion = self.GenerateFollowUpQuestion(question, response.Content)
                        Insights = self.ExtractInsights(response.Content)
                    }
                    
                    // Add to conversation history
                    let timestamp = DateTime.UtcNow
                    conversationHistory <- ("self", question, timestamp) :: ("tars", response.Content, timestamp) :: conversationHistory
                    
                    return selfResponse
                    
                | Error error ->
                    return {
                        Response = $"I encountered an error processing that question: {error}"
                        InternalThoughts = "Error in MoE processing"
                        ConfidenceLevel = 0.0
                        ExpertUsed = "Error"
                        NextQuestion = None
                        Insights = []
                    }
            with
            | ex ->
                logger.LogError(ex, "Error in self-question processing")
                return {
                    Response = $"I had trouble processing that question: {ex.Message}"
                    InternalThoughts = "Exception in processing"
                    ConfidenceLevel = 0.0
                    ExpertUsed = "Error"
                    NextQuestion = None
                    Insights = []
                }
        }
    
    /// <summary>
    /// Runs autonomous dialogue
    /// </summary>
    member self.RunAutonomousDialogue(context: SelfConversationContext) =
        task {
            let autonomousQuestions = [
                "What am I currently capable of?"
                "How can I improve my reasoning?"
                "What would make me more helpful to users?"
                "What patterns do I notice in my responses?"
            ]
            
            printfn "🔄 Starting autonomous dialogue..."
            
            for question in autonomousQuestions do
                let! response = self.ProcessSelfQuestion(question)
                printfn ""
                printfn $"🤖 Self-Question: {question}"
                printfn $"💭 Response: {response.Response}"

                if response.NextQuestion.IsSome then
                    let! followUp = self.ProcessSelfQuestion(response.NextQuestion.Value)
                    printfn $"🔄 Follow-up: {followUp.Response}"
        }
    
    /// <summary>
    /// Generates topic-specific questions
    /// </summary>
    member self.GenerateTopicQuestions(topic: string) =
        match topic.ToLower() with
        | t when t.Contains("performance") || t.Contains("optimization") ->
            [
                "What are my current performance bottlenecks?"
                "How can I optimize my response generation?"
                "What would make my processing faster?"
                "Are there patterns in slow responses?"
            ]
        | t when t.Contains("code") || t.Contains("programming") ->
            [
                "How can I improve my code generation?"
                "What coding patterns do I use most?"
                "How can I write cleaner code?"
                "What programming concepts should I focus on?"
            ]
        | t when t.Contains("learning") || t.Contains("improvement") ->
            [
                "What have I learned recently?"
                "How can I learn more effectively?"
                "What knowledge gaps do I have?"
                "How can I improve my understanding?"
            ]
        | _ ->
            [
                $"What do I know about {topic}?"
                $"How can I improve my understanding of {topic}?"
                $"What questions should I ask about {topic}?"
                $"What would help me with {topic}?"
            ]
    
    /// <summary>
    /// Generates follow-up questions
    /// </summary>
    member self.GenerateFollowUpQuestion(originalQuestion: string, response: string) =
        if response.Contains("improve") then
            Some "What specific steps can I take to implement these improvements?"
        elif response.Contains("learn") then
            Some "What would be the best way to learn this?"
        elif response.Contains("analyze") then
            Some "What patterns do I see in this analysis?"
        else
            None
    
    /// <summary>
    /// Extracts insights from response
    /// </summary>
    member self.ExtractInsights(response: string) =
        let insights = ResizeArray<string>()
        
        if response.Contains("pattern") then
            insights.Add("Identified patterns in responses")
        if response.Contains("improve") then
            insights.Add("Found improvement opportunities")
        if response.Contains("learn") then
            insights.Add("Discovered learning opportunities")
        if response.Contains("optimize") then
            insights.Add("Identified optimization potential")
        
        insights |> Seq.toList
    
    /// <summary>
    /// Extracts insights from conversation history
    /// </summary>
    member self.ExtractInsightsFromHistory() : string list =
        conversationHistory
        |> List.collect (fun (_, content, _) -> self.ExtractInsights(content))
        |> List.distinct

