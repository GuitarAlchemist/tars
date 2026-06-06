namespace TarsEngine.SelfImprovement

open System
open System.IO
open System.Text.RegularExpressions
open FSharp.Data

type AgentRole =
    | Planner
    | Coder
    | Critic
    | Executor

type AgentConfig = {
    Role: AgentRole
    Model: string
    Temperature: float
    Description: string
}

type AgentMessage = {
    Role: AgentRole
    Content: string
    Timestamp: DateTime
}

type AgentTask = {
    Id: string
    Description: string
    AssignedTo: AgentRole
    Status: string
    Messages: AgentMessage list
    Result: string option
}

type WorkflowState = {
    Tasks: AgentTask list
    CurrentTaskIndex: int
    StartTime: DateTime
    EndTime: DateTime option
    Status: string
}

module AgentCoordination =
    let createAgent (role: AgentRole) (model: string) (temperature: float) (description: string) =
        { Role = role
          Model = model
          Temperature = temperature
          Description = description }
    
    let createTask (id: string) (description: string) (assignedTo: AgentRole) =
        { Id = id
          Description = description
          AssignedTo = assignedTo
          Status = "Pending"
          Messages = []
          Result = None }
    
    let createMessage (role: AgentRole) (content: string) =
        { Role = role
          Content = content
          Timestamp = DateTime.UtcNow }
    
    let createWorkflow (tasks: AgentTask list) =
        { Tasks = tasks
          CurrentTaskIndex = 0
          StartTime = DateTime.UtcNow
          EndTime = None
          Status = "Running" }
    
    let addMessageToTask (task: AgentTask) (message: AgentMessage) =
        { task with Messages = task.Messages @ [message] }
    
    let completeTask (task: AgentTask) (result: string) =
        { task with 
            Status = "Completed"
            Result = Some result }
    
    let failTask (task: AgentTask) (error: string) =
        { task with 
            Status = "Failed"
            Result = Some error }
    
    let moveToNextTask (workflow: WorkflowState) =
        if workflow.CurrentTaskIndex >= workflow.Tasks.Length - 1 then
            { workflow with 
                Status = "Completed"
                EndTime = Some DateTime.UtcNow }
        else
            { workflow with CurrentTaskIndex = workflow.CurrentTaskIndex + 1 }
    
    let getCurrentTask (workflow: WorkflowState) =
        if workflow.CurrentTaskIndex < workflow.Tasks.Length then
            Some workflow.Tasks.[workflow.CurrentTaskIndex]
        else
            None
    
    let executeAgentPrompt (ollamaEndpoint: string) (agent: AgentConfig) (prompt: string) =
        async {
            try
                // Call Ollama API
                let ollamaUrl = sprintf "%s/api/generate" ollamaEndpoint
                let requestBody = 
                    sprintf "{\"model\": \"%s\", \"prompt\": \"%s\", \"temperature\": %f, \"stream\": false}"
                        agent.Model (prompt.Replace("\"", "\\\"").Replace("\n", "\\n")) agent.Temperature
                
                let! response = Http.AsyncRequestString(ollamaUrl, httpMethod = "POST", body = TextRequest requestBody)
                
                // Extract the response content
                let responsePattern = "\"response\":\"([\\s\\S]*?)\""
                let responseMatch = Regex.Match(response, responsePattern)
                
                if responseMatch.Success && responseMatch.Groups.Count > 1 then
                    return Ok responseMatch.Groups.[1].Value
                else
                    return Error "Failed to parse AI response"
            with ex ->
                return Error (sprintf "Error calling AI: %s" ex.Message)
        }
    
    let executeTask (ollamaEndpoint: string) (agents: Map<AgentRole, AgentConfig>) (task: AgentTask) =
        async {
            if not (agents.ContainsKey(task.AssignedTo)) then
                return failTask task (sprintf "No agent configured for role %A" task.AssignedTo)
            else
                let agent = agents.[task.AssignedTo]
                
                // Create the prompt based on the agent role and task
                let prompt = 
                    match task.AssignedTo with
                    | Planner ->
                        sprintf "You are a planning agent for the TARS system. Your role is to break down tasks into smaller steps.\n\nTask: %s\n\nProvide a detailed plan with numbered steps." task.Description
                    | Coder ->
                        sprintf "You are a coding agent for the TARS system. Your role is to write code based on requirements.\n\nTask: %s\n\nProvide the implementation code." task.Description
                    | Critic ->
                        sprintf "You are a critic agent for the TARS system. Your role is to review and provide feedback.\n\nTask: %s\n\nProvide a detailed critique and suggestions for improvement." task.Description
                    | Executor ->
                        sprintf "You are an executor agent for the TARS system. Your role is to execute plans and report results.\n\nTask: %s\n\nDescribe how you would execute this task and what the expected outcome would be." task.Description
                
                // Execute the prompt
                let! result = executeAgentPrompt ollamaEndpoint agent prompt
                
                match result with
                | Ok response ->
                    // Add the response as a message
                    let message = createMessage task.AssignedTo response
                    let updatedTask = addMessageToTask task message
                    
                    // Complete the task
                    return completeTask updatedTask response
                | Error error ->
                    return failTask task error
        }
    
    let executeWorkflow (ollamaEndpoint: string) (agents: Map<AgentRole, AgentConfig>) (workflow: WorkflowState) =
        let rec executeNextTask (currentWorkflow: WorkflowState) =
            async {
                match getCurrentTask currentWorkflow with
                | None ->
                    // No more tasks, workflow is complete
                    return { currentWorkflow with Status = "Completed"; EndTime = Some DateTime.UtcNow }
                | Some task ->
                    // Execute the current task
                    let! updatedTask = executeTask ollamaEndpoint agents task
                    
                    // Update the task in the workflow
                    let updatedTasks = 
                        currentWorkflow.Tasks 
                        |> List.mapi (fun i t -> if i = currentWorkflow.CurrentTaskIndex then updatedTask else t)
                    
                    let updatedWorkflow = { currentWorkflow with Tasks = updatedTasks }
                    
                    // Check if the task failed
                    if updatedTask.Status = "Failed" then
                        return { updatedWorkflow with Status = "Failed"; EndTime = Some DateTime.UtcNow }
                    else
                        // Move to the next task
                        let nextWorkflow = moveToNextTask updatedWorkflow
                        
                        // If we've moved to a new task, continue execution
                        if nextWorkflow.Status = "Completed" then
                            return nextWorkflow
                        else
                            return! executeNextTask nextWorkflow
            }
        
        executeNextTask workflow
    
    let createStandardWorkflow (taskDescription: string) =
        let planningTask = createTask "1" (sprintf "Create a plan for: %s" taskDescription) Planner
        let codingTask = createTask "2" "Implement the plan created in the previous step" Coder
        let reviewTask = createTask "3" "Review the implementation from the previous step" Critic
        let executionTask = createTask "4" "Execute the implementation and report results" Executor
        
        createWorkflow [planningTask; codingTask; reviewTask; executionTask]
    
    let createStandardAgents (ollamaEndpoint: string) =
        let planner = createAgent Planner "llama3" 0.7 "Plans the overall approach and breaks down tasks"
        let coder = createAgent Coder "codellama:13b-code" 0.2 "Writes and refines code based on the plan"
        let critic = createAgent Critic "llama3" 0.5 "Reviews and critiques code and plans"
        let executor = createAgent Executor "llama3" 0.3 "Executes plans and reports results"
        
        Map.ofList [(Planner, planner); (Coder, coder); (Critic, critic); (Executor, executor)]
    
    let runStandardWorkflow (ollamaEndpoint: string) (taskDescription: string) =
        async {
            let workflow = createStandardWorkflow taskDescription
            let agents = createStandardAgents ollamaEndpoint
            
            return! executeWorkflow ollamaEndpoint agents workflow
        }
