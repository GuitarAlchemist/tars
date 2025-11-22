namespace TarsEngine.FSharp.Core.Context

open System
open System.IO
open System.Text.Json
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Context.Types

/// JSON schema definitions for TARS outputs
module Schemas =
    
    let planStepSchema = """{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "PlanStep",
  "type": "object",
  "required": ["id", "title", "actions"],
  "properties": {
    "id": {"type": "string"},
    "title": {"type": "string"},
    "intent": {"type": "string", "enum": ["plan", "codegen", "eval", "refactor", "reasoning", "metascript_execution", "autonomous_improvement"]},
    "actions": {"type": "array", "items": {"type": "string"}},
    "dependsOn": {"type": "array", "items": {"type": "string"}},
    "estimatedEffort": {"type": "string", "enum": ["XS", "S", "M", "L", "XL"]},
    "priority": {"type": "integer", "minimum": 1, "maximum": 10}
  }
}"""

    let beliefUpdateSchema = """{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "BeliefUpdate",
  "type": "object",
  "required": ["summary", "provenance", "contradictions_checked"],
  "properties": {
    "summary": {"type": "string"},
    "provenance": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["file"],
        "properties": {
          "file": {"type": "string"},
          "line": {"type": "integer"},
          "hash": {"type": "string"},
          "idx": {"type": "integer"},
          "confidence": {"type": "number", "minimum": 0, "maximum": 1}
        }
      }
    },
    "contradictions_checked": {"type": "boolean"},
    "salience": {"type": "number", "minimum": 0, "maximum": 1},
    "intent_context": {"type": "string"},
    "validation_status": {"type": "string", "enum": ["confirmed", "pending", "rejected"]},
    "impact_assessment": {"type": "string"}
  }
}"""

    let contextMetricsSchema = """{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "ContextMetrics",
  "type": "object",
  "required": ["timestamp", "step_name", "token_usage", "retrieval_performance"],
  "properties": {
    "timestamp": {"type": "string", "format": "date-time"},
    "step_name": {"type": "string"},
    "token_usage": {
      "type": "object",
      "required": ["budget", "used", "efficiency"],
      "properties": {
        "budget": {"type": "integer"},
        "used": {"type": "integer"},
        "efficiency": {"type": "number", "minimum": 0, "maximum": 1},
        "compression_savings": {"type": "integer"}
      }
    },
    "retrieval_performance": {
      "type": "object",
      "required": ["intent_classification_confidence", "spans_retrieved"],
      "properties": {
        "intent_classification_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "spans_retrieved": {"type": "integer"},
        "spans_after_budgeting": {"type": "integer"},
        "spans_after_compression": {"type": "integer"},
        "retrieval_time_ms": {"type": "integer"},
        "cache_hit_rate": {"type": "number", "minimum": 0, "maximum": 1}
      }
    },
    "memory_utilization": {
      "type": "object",
      "properties": {
        "ephemeral_spans": {"type": "integer"},
        "working_set_spans": {"type": "integer"},
        "long_term_spans": {"type": "integer"},
        "promotion_candidates": {"type": "integer"},
        "consolidation_pending": {"type": "boolean"}
      }
    },
    "quality_indicators": {
      "type": "object",
      "properties": {
        "salience_distribution": {"type": "array", "items": {"type": "number"}},
        "compression_quality": {"type": "number", "minimum": 0, "maximum": 1},
        "intent_routing_accuracy": {"type": "number", "minimum": 0, "maximum": 1},
        "security_violations": {"type": "integer"},
        "schema_validation_pass_rate": {"type": "number", "minimum": 0, "maximum": 1}
      }
    }
  }
}"""

    let autonomousObjectiveSchema = """{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "AutonomousObjective",
  "type": "object",
  "required": ["id", "name", "description", "priority", "success_criteria"],
  "properties": {
    "id": {"type": "string"},
    "name": {"type": "string"},
    "description": {"type": "string"},
    "priority": {"type": "integer", "minimum": 1, "maximum": 10},
    "estimated_effort": {"type": "integer", "minimum": 1},
    "dependencies": {"type": "array", "items": {"type": "string"}},
    "success_criteria": {"type": "array", "items": {"type": "string"}},
    "agent_os_spec": {"type": "string"},
    "performance_targets": {"type": "array", "items": {"type": "string"}},
    "risk_assessment": {"type": "array", "items": {"type": "string"}}
  }
}"""

/// Configuration for output validation
type ValidationConfig = {
    SchemaDirectory: string
    EnableStrictValidation: bool
    EnableAutoRepair: bool
    MaxRepairAttempts: int
}

/// JSON schema-based output validator
type JsonSchemaOutputValidator(config: ValidationConfig, logger: ILogger<JsonSchemaOutputValidator>) =
    
    /// Built-in schemas
    let builtInSchemas = Map.ofList [
        ("plan_step", Schemas.planStepSchema)
        ("belief_update", Schemas.beliefUpdateSchema)
        ("context_metrics", Schemas.contextMetricsSchema)
        ("autonomous_objective", Schemas.autonomousObjectiveSchema)
    ]
    
    /// Load schema from file or built-in
    let loadSchema (schemaName: string) =
        // Try built-in schemas first
        match builtInSchemas.TryFind schemaName with
        | Some schema -> Some schema
        | None ->
            // Try loading from file
            let schemaPath = Path.Combine(config.SchemaDirectory, $"{schemaName}.schema.json")
            if File.Exists schemaPath then
                try
                    Some (File.ReadAllText schemaPath)
                with
                | ex ->
                    logger.LogError(ex, "Failed to load schema from {SchemaPath}", schemaPath)
                    None
            else
                logger.LogWarning("Schema not found: {SchemaName}", schemaName)
                None
    
    /// Simple JSON validation (basic structure check)
    let validateJsonStructure (json: string) =
        try
            JsonDocument.Parse(json) |> ignore
            (true, [])
        with
        | ex -> (false, [ex.Message])
    
    /// Validate required fields
    let validateRequiredFields (jsonDoc: JsonDocument) (requiredFields: string list) =
        let errors = ResizeArray<string>()
        let root = jsonDoc.RootElement
        
        for field in requiredFields do
            if not (root.TryGetProperty(field).HasValue) then
                errors.Add($"Missing required field: {field}")
        
        errors |> List.ofSeq
    
    /// Extract required fields from schema
    let extractRequiredFields (schema: string) =
        try
            use schemaDoc = JsonDocument.Parse(schema)
            let root = schemaDoc.RootElement
            
            if root.TryGetProperty("required").HasValue then
                let requiredArray = root.GetProperty("required")
                [
                    for i in 0 .. requiredArray.GetArrayLength() - 1 do
                        yield requiredArray[i].GetString()
                ]
            else
                []
        with
        | ex ->
            logger.LogError(ex, "Failed to extract required fields from schema")
            []
    
    /// Validate enum values
    let validateEnumFields (jsonDoc: JsonDocument) (schema: string) =
        let errors = ResizeArray<string>()
        
        try
            use schemaDoc = JsonDocument.Parse(schema)
            let schemaRoot = schemaDoc.RootElement
            let jsonRoot = jsonDoc.RootElement
            
            if schemaRoot.TryGetProperty("properties").HasValue then
                let properties = schemaRoot.GetProperty("properties")
                
                for prop in properties.EnumerateObject() do
                    if prop.Value.TryGetProperty("enum").HasValue then
                        let enumValues = prop.Value.GetProperty("enum")
                        let allowedValues = [
                            for i in 0 .. enumValues.GetArrayLength() - 1 do
                                yield enumValues[i].GetString()
                        ]
                        
                        if jsonRoot.TryGetProperty(prop.Name).HasValue then
                            let actualValue = jsonRoot.GetProperty(prop.Name).GetString()
                            if not (allowedValues |> List.contains actualValue) then
                                errors.Add($"Invalid enum value for {prop.Name}: {actualValue}. Allowed: {String.concat ", " allowedValues}")
        with
        | ex ->
            logger.LogError(ex, "Failed to validate enum fields")
            errors.Add($"Enum validation error: {ex.Message}")
        
        errors |> List.ofSeq
    
    /// Attempt to repair invalid JSON
    let repairJson (json: string) (errors: string list) =
        let mutable repairedJson = json
        
        // Basic repair attempts
        for error in errors do
            if error.Contains("Missing required field") then
                let fieldName = error.Substring(error.LastIndexOf(": ") + 2)
                
                // Add missing field with default value
                let defaultValue = 
                    match fieldName with
                    | "id" -> $"\"{Guid.NewGuid().ToString("N").[0..7]}\""
                    | "timestamp" -> $"\"{DateTime.UtcNow:yyyy-MM-ddTHH:mm:ssZ}\""
                    | "contradictions_checked" -> "true"
                    | "priority" -> "5"
                    | "actions" | "dependencies" | "success_criteria" -> "[]"
                    | _ -> "\"default\""
                
                // Simple insertion (this is a basic implementation)
                if repairedJson.EndsWith("}") then
                    let insertPos = repairedJson.LastIndexOf("}")
                    let insertion = $",\n  \"{fieldName}\": {defaultValue}"
                    repairedJson <- repairedJson.Insert(insertPos, insertion)
        
        repairedJson
    
    interface IOutputValidator with
        
        member _.ValidateAsync(output, schemaName) =
            task {
                logger.LogDebug("Validating output against schema: {SchemaName}", schemaName)
                
                // Basic JSON structure validation
                let (isValidJson, jsonErrors) = validateJsonStructure output
                if not isValidJson then
                    logger.LogWarning("Invalid JSON structure in output")
                    return (false, jsonErrors)
                
                // Load schema
                match loadSchema schemaName with
                | Some schema ->
                    try
                        use jsonDoc = JsonDocument.Parse(output)
                        
                        // Validate required fields
                        let requiredFields = extractRequiredFields schema
                        let requiredFieldErrors = validateRequiredFields jsonDoc requiredFields
                        
                        // Validate enum fields
                        let enumErrors = validateEnumFields jsonDoc schema
                        
                        let allErrors = requiredFieldErrors @ enumErrors
                        
                        if allErrors.IsEmpty then
                            logger.LogDebug("Output validation passed for schema: {SchemaName}", schemaName)
                            return (true, [])
                        else
                            logger.LogWarning("Output validation failed for schema {SchemaName}: {Errors}", 
                                schemaName, String.concat "; " allErrors)
                            return (false, allErrors)
                            
                    with
                    | ex ->
                        logger.LogError(ex, "Validation error for schema {SchemaName}", schemaName)
                        return (false, [ex.Message])
                
                | None ->
                    logger.LogWarning("Schema not found: {SchemaName}", schemaName)
                    return (false, [$"Schema not found: {schemaName}"])
            }
        
        member _.GetSchema(outputType) =
            loadSchema outputType
        
        member _.RepairOutputAsync(output, schemaName, errors) =
            task {
                if not config.EnableAutoRepair then
                    return output
                
                logger.LogInformation("Attempting to repair output for schema: {SchemaName}", schemaName)
                
                let mutable repairedOutput = output
                let mutable attemptCount = 0
                
                while attemptCount < config.MaxRepairAttempts do
                    attemptCount <- attemptCount + 1
                    
                    repairedOutput <- repairJson repairedOutput errors
                    
                    // Validate the repaired output
                    let! (isValid, newErrors) = (this :> IOutputValidator).ValidateAsync(repairedOutput, schemaName)
                    
                    if isValid then
                        logger.LogInformation("Successfully repaired output after {Attempts} attempts", attemptCount)
                        return repairedOutput
                    else
                        logger.LogDebug("Repair attempt {Attempt} failed with errors: {Errors}", 
                            attemptCount, String.concat "; " newErrors)
                
                logger.LogWarning("Failed to repair output after {MaxAttempts} attempts", config.MaxRepairAttempts)
                return repairedOutput
            }
