namespace TarsEngine.FSharp.Cli.Core

open System
open System.Threading.Tasks
open TarsEngine.FSharp.Cli.Core.MultiAgentDomain

// ============================================================================
// RESULT-BASED ERROR HANDLING - RAILWAY ORIENTED PROGRAMMING
// ============================================================================

module ResultBasedErrorHandling =

    // ============================================================================
    // CORE RESULT TYPES
    // ============================================================================

    /// Enhanced Result type with detailed error information
    type TarsResult<'Success, 'Error> =
        | Success of 'Success
        | Error of 'Error

    /// Detailed error information
    type TarsError = {
        ErrorType: ErrorType
        Message: string
        Details: string option
        InnerError: exn option
        Timestamp: DateTime
        Context: Map<string, obj>
    }

    and ErrorType =
        | ValidationError
        | ProcessingError
        | CommunicationError
        | ResourceError
        | ConfigurationError
        | UnexpectedError

    /// Async Result type
    type AsyncTarsResult<'Success, 'Error> = Task<TarsResult<'Success, 'Error>>

    // ============================================================================
    // RESULT BUILDERS
    // ============================================================================

    /// Result computation expression
    type TarsResultBuilder() =
        member _.Return(value: 'T) : TarsResult<'T, 'Error> = Success value
        
        member _.ReturnFrom(result: TarsResult<'T, 'Error>) : TarsResult<'T, 'Error> = result
        
        member _.Bind(result: TarsResult<'T, 'Error>, binder: 'T -> TarsResult<'U, 'Error>) : TarsResult<'U, 'Error> =
            match result with
            | Success value -> binder value
            | Error error -> Error error
        
        member _.Zero() : TarsResult<unit, 'Error> = Success ()
        
        member _.Combine(result1: TarsResult<unit, 'Error>, result2: TarsResult<'T, 'Error>) : TarsResult<'T, 'Error> =
            match result1 with
            | Success _ -> result2
            | Error error -> Error error
        
        member _.Delay(f: unit -> TarsResult<'T, 'Error>) : TarsResult<'T, 'Error> = f()
        
        member _.TryWith(body: unit -> TarsResult<'T, 'Error>, handler: exn -> TarsResult<'T, 'Error>) : TarsResult<'T, 'Error> =
            try body()
            with ex -> handler ex
        
        member _.TryFinally(body: unit -> TarsResult<'T, 'Error>, compensation: unit -> unit) : TarsResult<'T, 'Error> =
            try body()
            finally compensation()

    /// Async Result computation expression
    type AsyncTarsResultBuilder() =
        member _.Return(value: 'T) : AsyncTarsResult<'T, 'Error> = 
            Task.FromResult(Success value)
        
        member _.ReturnFrom(result: AsyncTarsResult<'T, 'Error>) : AsyncTarsResult<'T, 'Error> = result
        
        member _.Bind(result: AsyncTarsResult<'T, 'Error>, binder: 'T -> AsyncTarsResult<'U, 'Error>) : AsyncTarsResult<'U, 'Error> = task {
            let! r = result
            match r with
            | Success value -> return! binder value
            | Error error -> return Error error
        }
        
        member _.Zero() : AsyncTarsResult<unit, 'Error> = 
            Task.FromResult(Success ())
        
        member _.Combine(result1: AsyncTarsResult<unit, 'Error>, result2: AsyncTarsResult<'T, 'Error>) : AsyncTarsResult<'T, 'Error> = task {
            let! r1 = result1
            match r1 with
            | Success _ -> return! result2
            | Error error -> return Error error
        }
        
        member _.Delay(f: unit -> AsyncTarsResult<'T, 'Error>) : AsyncTarsResult<'T, 'Error> = 
            task { return! f() }

    let result = TarsResultBuilder()
    let asyncResult = AsyncTarsResultBuilder()

    // ============================================================================
    // ERROR CREATION UTILITIES
    // ============================================================================

    let createError (errorType: ErrorType) (message: string) (details: string option) (context: Map<string, obj>) : TarsError =
        {
            ErrorType = errorType
            Message = message
            Details = details
            InnerError = None
            Timestamp = DateTime.UtcNow
            Context = context
        }

    let validationError (message: string) (details: string option) : TarsError =
        createError ValidationError message details Map.empty

    let processingError (message: string) (details: string option) : TarsError =
        createError ProcessingError message details Map.empty

    let communicationError (message: string) (details: string option) : TarsError =
        createError CommunicationError message details Map.empty

    let resourceError (message: string) (details: string option) : TarsError =
        createError ResourceError message details Map.empty

    let configurationError (message: string) (details: string option) : TarsError =
        createError ConfigurationError message details Map.empty

    let unexpectedError (message: string) (ex: exn option) : TarsError =
        {
            ErrorType = UnexpectedError
            Message = message
            Details = ex |> Option.map (fun e -> e.ToString())
            InnerError = ex
            Timestamp = DateTime.UtcNow
            Context = Map.empty
        }

    // ============================================================================
    // RESULT UTILITIES
    // ============================================================================

    module Result =
        
        let map (mapper: 'T -> 'U) (result: TarsResult<'T, 'Error>) : TarsResult<'U, 'Error> =
            match result with
            | Success value -> Success (mapper value)
            | Error error -> Error error

        let mapError (mapper: 'Error1 -> 'Error2) (result: TarsResult<'Success, 'Error1>) : TarsResult<'Success, 'Error2> =
            match result with
            | Success value -> Success value
            | Error error -> Error (mapper error)

        let bind (binder: 'T -> TarsResult<'U, 'Error>) (result: TarsResult<'T, 'Error>) : TarsResult<'U, 'Error> =
            match result with
            | Success value -> binder value
            | Error error -> Error error

        let apply (funcResult: TarsResult<'T -> 'U, 'Error>) (valueResult: TarsResult<'T, 'Error>) : TarsResult<'U, 'Error> =
            match funcResult, valueResult with
            | Success func, Success value -> Success (func value)
            | Error error, _ -> Error error
            | _, Error error -> Error error

        let combine (results: TarsResult<'T, 'Error> list) : TarsResult<'T list, 'Error> =
            let rec loop acc remaining =
                match remaining with
                | [] -> Success (List.rev acc)
                | (Success value) :: rest -> loop (value :: acc) rest
                | (Error error) :: _ -> Error error
            loop [] results

        let traverse (mapper: 'T -> TarsResult<'U, 'Error>) (items: 'T list) : TarsResult<'U list, 'Error> =
            items |> List.map mapper |> combine

        let fold (folder: 'State -> 'T -> TarsResult<'State, 'Error>) (initial: 'State) (items: 'T list) : TarsResult<'State, 'Error> =
            let rec loop state remaining =
                match remaining with
                | [] -> Success state
                | item :: rest ->
                    match folder state item with
                    | Success newState -> loop newState rest
                    | Error error -> Error error
            loop initial items

        let defaultValue (defaultVal: 'T) (result: TarsResult<'T, 'Error>) : 'T =
            match result with
            | Success value -> value
            | Error _ -> defaultVal

        let defaultWith (defaultFunc: 'Error -> 'T) (result: TarsResult<'T, 'Error>) : 'T =
            match result with
            | Success value -> value
            | Error error -> defaultFunc error

        let isSuccess (result: TarsResult<'T, 'Error>) : bool =
            match result with
            | Success _ -> true
            | Error _ -> false

        let isError (result: TarsResult<'T, 'Error>) : bool =
            not (isSuccess result)

        let toOption (result: TarsResult<'T, 'Error>) : 'T option =
            match result with
            | Success value -> Some value
            | Error _ -> None

        let fromOption (error: 'Error) (option: 'T option) : TarsResult<'T, 'Error> =
            match option with
            | Some value -> Success value
            | None -> Error error

        let catch (operation: unit -> 'T) : TarsResult<'T, TarsError> =
            try
                Success (operation())
            with
            | ex -> Error (unexpectedError "Operation failed with exception" (Some ex))

    module AsyncResult =
        
        let map (mapper: 'T -> 'U) (result: AsyncTarsResult<'T, 'Error>) : AsyncTarsResult<'U, 'Error> = task {
            let! r = result
            return Result.map mapper r
        }

        let mapError (mapper: 'Error1 -> 'Error2) (result: AsyncTarsResult<'Success, 'Error1>) : AsyncTarsResult<'Success, 'Error2> = task {
            let! r = result
            return Result.mapError mapper r
        }

        let bind (binder: 'T -> AsyncTarsResult<'U, 'Error>) (result: AsyncTarsResult<'T, 'Error>) : AsyncTarsResult<'U, 'Error> = task {
            let! r = result
            match r with
            | Success value -> return! binder value
            | Error error -> return Error error
        }

        let apply (funcResult: AsyncTarsResult<'T -> 'U, 'Error>) (valueResult: AsyncTarsResult<'T, 'Error>) : AsyncTarsResult<'U, 'Error> = task {
            let! func = funcResult
            let! value = valueResult
            return Result.apply func value
        }

        let combine (results: AsyncTarsResult<'T, 'Error> list) : AsyncTarsResult<'T list, 'Error> = task {
            let! resolvedResults = Task.WhenAll(results)
            return Result.combine (resolvedResults |> Array.toList)
        }

        let traverse (mapper: 'T -> AsyncTarsResult<'U, 'Error>) (items: 'T list) : AsyncTarsResult<'U list, 'Error> =
            items |> List.map mapper |> combine

        let catch (operation: unit -> Task<'T>) : AsyncTarsResult<'T, TarsError> = task {
            try
                let! result = operation()
                return Success result
            with
            | ex -> return Error (unexpectedError "Async operation failed with exception" (Some ex))
        }

    // ============================================================================
    // DOMAIN-SPECIFIC ERROR HANDLING
    // ============================================================================

    module AgentErrorHandling =
        
        type AgentOperationError =
            | AgentNotFound of agentId: string
            | InvalidSpecialization of specialization: string
            | CommunicationFailure of fromAgent: string * toAgent: string * reason: string
            | TaskAssignmentFailure of agentId: string * task: string * reason: string
            | PerformanceThresholdNotMet of agentId: string * expected: float * actual: float

        let validateAgent (agent: UnifiedAgent) : TarsResult<UnifiedAgent, TarsError> =
            result {
                if String.IsNullOrWhiteSpace(agent.Id) then
                    return! Error (validationError "Agent ID cannot be empty" None)
                elif String.IsNullOrWhiteSpace(agent.Name) then
                    return! Error (validationError "Agent name cannot be empty" None)
                elif agent.QualityScore < 0.0 || agent.QualityScore > 1.0 then
                    return! Error (validationError "Agent quality score must be between 0 and 1" (Some $"Current value: {agent.QualityScore}"))
                else
                    return agent
            }

        let assignTask (task: string) (agent: UnifiedAgent) : TarsResult<UnifiedAgent, TarsError> =
            result {
                let! validAgent = validateAgent agent
                if validAgent.Status = Working(_) then
                    return! Error (processingError "Agent is already working on a task" (Some $"Current task: {validAgent.CurrentTask |> Option.defaultValue "Unknown"}"))
                else
                    return { validAgent with 
                               Status = Working(task)
                               CurrentTask = Some task }
            }

        let communicateBetweenAgents (fromAgent: UnifiedAgent) (toAgent: UnifiedAgent) (message: string) : TarsResult<unit, TarsError> =
            result {
                let! validFromAgent = validateAgent fromAgent
                let! validToAgent = validateAgent toAgent
                
                if validFromAgent.Id = validToAgent.Id then
                    return! Error (validationError "Agent cannot communicate with itself" None)
                elif String.IsNullOrWhiteSpace(message) then
                    return! Error (validationError "Communication message cannot be empty" None)
                else
                    // Simulate communication success/failure
                    let successRate = min validFromAgent.PerformanceMetrics.CommunicationEfficiency validToAgent.PerformanceMetrics.CommunicationEfficiency
                    if Random().NextDouble() > successRate then
                        return! Error (communicationError "Communication failed due to network issues" (Some $"Success rate: {successRate:P1}"))
                    else
                        return ()
            }

    module DepartmentErrorHandling =
        
        type DepartmentOperationError =
            | DepartmentNotFound of departmentId: string
            | InsufficientAgents of required: int * available: int
            | ResourceConstraint of resource: string * required: float * available: float
            | CoordinationFailure of reason: string

        let validateDepartment (dept: UnifiedDepartment) : TarsResult<UnifiedDepartment, TarsError> =
            result {
                if String.IsNullOrWhiteSpace(dept.Id) then
                    return! Error (validationError "Department ID cannot be empty" None)
                elif String.IsNullOrWhiteSpace(dept.Name) then
                    return! Error (validationError "Department name cannot be empty" None)
                elif dept.Agents.IsEmpty then
                    return! Error (validationError "Department must have at least one agent" None)
                else
                    return dept
            }

        let addAgentToDepartment (agent: UnifiedAgent) (dept: UnifiedDepartment) : TarsResult<UnifiedDepartment, TarsError> =
            result {
                let! validAgent = AgentErrorHandling.validateAgent agent
                let! validDept = validateDepartment dept
                
                if validDept.Agents |> List.exists (fun a -> a.Id = validAgent.Id) then
                    return! Error (validationError "Agent is already in the department" (Some $"Agent ID: {validAgent.Id}"))
                else
                    return { validDept with Agents = validAgent :: validDept.Agents }
            }

    module ProblemErrorHandling =
        
        type ProblemOperationError =
            | ProblemNotFound of problemId: string
            | InvalidComplexity of complexity: string
            | CircularDependency of subProblemIds: string list
            | InsufficientExpertise of required: string list * available: string list

        let validateProblem (problem: UnifiedProblem) : TarsResult<UnifiedProblem, TarsError> =
            result {
                if String.IsNullOrWhiteSpace(problem.Id) then
                    return! Error (validationError "Problem ID cannot be empty" None)
                elif String.IsNullOrWhiteSpace(problem.OriginalStatement) then
                    return! Error (validationError "Problem statement cannot be empty" None)
                elif problem.ConfidenceScore < 0.0 || problem.ConfidenceScore > 1.0 then
                    return! Error (validationError "Problem confidence score must be between 0 and 1" (Some $"Current value: {problem.ConfidenceScore}"))
                else
                    return problem
            }

        let checkDependencyCycles (problem: UnifiedProblem) : TarsResult<unit, TarsError> =
            // Simplified cycle detection
            let dependencies = problem.Dependencies |> List.map (fun d -> (d.FromSubProblem, d.ToSubProblem))
            let rec hasCycle visited current path =
                if Set.contains current visited then
                    List.contains current path
                else
                    let newVisited = Set.add current visited
                    let newPath = current :: path
                    dependencies
                    |> List.filter (fun (from, _) -> from = current)
                    |> List.map snd
                    |> List.exists (hasCycle newVisited current newPath)
            
            let subProblemIds = problem.SubProblems |> List.map (fun sp -> sp.Id)
            let cycleExists = subProblemIds |> List.exists (hasCycle Set.empty)
            
            if cycleExists then
                Error (validationError "Circular dependency detected in sub-problems" None)
            else
                Success ()

    // ============================================================================
    // ERROR REPORTING AND LOGGING
    // ============================================================================

    module ErrorReporting =
        
        let formatError (error: TarsError) : string =
            let errorTypeStr = 
                match error.ErrorType with
                | ValidationError -> "VALIDATION"
                | ProcessingError -> "PROCESSING"
                | CommunicationError -> "COMMUNICATION"
                | ResourceError -> "RESOURCE"
                | ConfigurationError -> "CONFIGURATION"
                | UnexpectedError -> "UNEXPECTED"
            
            let detailsStr = 
                error.Details 
                |> Option.map (fun d -> $"\nDetails: {d}")
                |> Option.defaultValue ""
            
            let contextStr = 
                if error.Context.IsEmpty then ""
                else
                    let contextItems = 
                        error.Context 
                        |> Map.toList 
                        |> List.map (fun (k, v) -> $"{k}: {v}")
                        |> String.concat ", "
                    $"\nContext: {contextItems}"
            
            $"[{errorTypeStr}] {error.Message}{detailsStr}{contextStr}"

        let logError (error: TarsError) : unit =
            let formattedError = formatError error
            Console.WriteLine($"ERROR [{error.Timestamp:yyyy-MM-dd HH:mm:ss}]: {formattedError}")
            
            match error.InnerError with
            | Some ex -> Console.WriteLine($"Inner Exception: {ex}")
            | None -> ()

        let handleError<'T> (defaultValue: 'T) (result: TarsResult<'T, TarsError>) : 'T =
            match result with
            | Success value -> value
            | Error error ->
                logError error
                defaultValue

        let handleErrorAsync<'T> (defaultValue: 'T) (result: AsyncTarsResult<'T, TarsError>) : Task<'T> = task {
            let! r = result
            return handleError defaultValue r
        }
