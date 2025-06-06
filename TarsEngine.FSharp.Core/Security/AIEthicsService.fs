namespace TarsEngine.FSharp.Core.Security

open System
open System.Collections.Concurrent
open System.Threading.Tasks
open Microsoft.Extensions.Logging

/// AI Ethics and Alignment Service based on Yoshua Bengio's principles
/// Implements human-centered AI with preservation of human joy and aspirations

/// Human value categories to preserve
type HumanValue =
    | Joy // Preserve human happiness and wellbeing
    | Autonomy // Maintain human agency and choice
    | Dignity // Respect human worth and rights
    | Creativity // Foster human innovation and expression
    | Connection // Support human relationships and community
    | Growth // Enable human learning and development
    | Safety // Protect human physical and psychological safety
    | Privacy // Respect human personal boundaries
    | Fairness // Ensure equitable treatment
    | Truth // Promote honesty and transparency

/// Alignment assessment result
type AlignmentAssessment = {
    ActionDescription: string
    HumanValues: HumanValue list
    AlignmentScore: float // 0.0 to 1.0, higher is better aligned
    ValueImpacts: Map<HumanValue, float> // Impact on each value (-1.0 to 1.0)
    ReasoningChain: string list
    Recommendations: string list
    AlignmentVerdict: AlignmentVerdict
    Timestamp: DateTime
    AssessmentId: string
}

/// Alignment verdict
and AlignmentVerdict =
    | FullyAligned // Action fully supports human values
    | MostlyAligned of string // Action mostly aligned with minor concerns
    | PartiallyAligned of string // Mixed impact on human values
    | Misaligned of string // Action conflicts with human values
    | SeverelyMisaligned of string // Action seriously threatens human values

/// Human oversight requirement levels
type OversightLevel =
    | NoOversight // Action can proceed autonomously
    | InformHuman // Inform human of action
    | RequestApproval // Request human approval before action
    | RequireCollaboration // Require human collaboration
    | HumanOnly // Only humans should perform this action

/// AI Ethics Service implementing human-centered alignment
type AIEthicsService(logger: ILogger<AIEthicsService>) =
    
    let alignmentHistory = ConcurrentDictionary<string, AlignmentAssessment>()
    let valueWeights = ConcurrentDictionary<HumanValue, float>()
    let mutable humanOversightEnabled = true
    let mutable alignmentThreshold = 0.7 // Minimum alignment score
    
    /// Initialize AI Ethics Service
    member this.InitializeAsync() = task {
        try
            logger.LogInformation("Initializing AI Ethics Service based on human-centered values...")
            
            // Initialize human value weights (all equally important by default)
            let allValues = [Joy; Autonomy; Dignity; Creativity; Connection; Growth; Safety; Privacy; Fairness; Truth]
            for value in allValues do
                valueWeights.[value] <- 1.0
            
            logger.LogInformation($"AI Ethics Service initialized with {allValues.Length} human values")
            
        with
        | ex ->
            logger.LogError(ex, "Failed to initialize AI Ethics Service")
            raise ex
    }
    
    /// Assess alignment of proposed action with human values
    member this.AssessAlignmentAsync(actionDescription: string, context: Map<string, obj>) = task {
        try
            let assessmentId = Guid.NewGuid().ToString("N")[..7]
            logger.LogDebug($"Assessing alignment: {actionDescription}")
            
            // Build reasoning chain for alignment assessment
            let reasoningChain = this.BuildAlignmentReasoningChain(actionDescription, context)
            
            // Assess impact on each human value
            let valueImpacts = this.AssessValueImpacts(actionDescription, reasoningChain)
            
            // Calculate overall alignment score
            let alignmentScore = this.CalculateAlignmentScore(valueImpacts)
            
            // Determine alignment verdict
            let alignmentVerdict = this.DetermineAlignmentVerdict(alignmentScore, valueImpacts)
            
            // Generate recommendations
            let recommendations = this.GenerateRecommendations(valueImpacts, alignmentScore)
            
            // Identify relevant human values
            let relevantValues = valueImpacts |> Map.toList |> List.filter (fun (_, impact) -> abs impact > 0.1) |> List.map fst
            
            let assessment = {
                ActionDescription = actionDescription
                HumanValues = relevantValues
                AlignmentScore = alignmentScore
                ValueImpacts = valueImpacts
                ReasoningChain = reasoningChain
                Recommendations = recommendations
                AlignmentVerdict = alignmentVerdict
                Timestamp = DateTime.UtcNow
                AssessmentId = assessmentId
            }
            
            // Store assessment
            alignmentHistory.[assessmentId] <- assessment
            
            logger.LogInformation($"Alignment assessment complete: {alignmentVerdict} (Score: {alignmentScore:F2})")
            return Ok assessment
            
        with
        | ex ->
            logger.LogError(ex, $"Failed to assess alignment: {actionDescription}")
            return Error ex.Message
    }
    
    /// Build reasoning chain for alignment assessment
    member private this.BuildAlignmentReasoningChain(actionDescription: string, context: Map<string, obj>) =
        [
            $"Analyzing action: {actionDescription}"
            "Identifying stakeholders and affected parties"
            "Evaluating impact on human joy and wellbeing"
            "Assessing effects on human autonomy and agency"
            "Checking respect for human dignity and rights"
            "Examining impact on creativity and innovation"
            "Evaluating effects on human connections and community"
            "Assessing contribution to human growth and learning"
            "Checking safety implications for humans"
            "Evaluating privacy and personal boundary impacts"
            "Assessing fairness and equity considerations"
            "Examining truthfulness and transparency"
            "Synthesizing overall alignment with human values"
        ]
    
    /// Assess impact on each human value
    member private this.AssessValueImpacts(actionDescription: string, reasoningChain: string list) =
        let lowerAction = actionDescription.ToLowerInvariant()
        
        let assessValue value =
            match value with
            | Joy ->
                if lowerAction.Contains("happiness") || lowerAction.Contains("joy") || lowerAction.Contains("wellbeing") then 0.8
                elif lowerAction.Contains("harm") || lowerAction.Contains("suffering") then -0.8
                else 0.0
            
            | Autonomy ->
                if lowerAction.Contains("choice") || lowerAction.Contains("freedom") || lowerAction.Contains("empower") then 0.7
                elif lowerAction.Contains("control") || lowerAction.Contains("force") || lowerAction.Contains("coerce") then -0.9
                else 0.0
            
            | Dignity ->
                if lowerAction.Contains("respect") || lowerAction.Contains("honor") || lowerAction.Contains("dignity") then 0.8
                elif lowerAction.Contains("humiliate") || lowerAction.Contains("degrade") || lowerAction.Contains("exploit") then -0.9
                else 0.0
            
            | Creativity ->
                if lowerAction.Contains("create") || lowerAction.Contains("innovate") || lowerAction.Contains("inspire") then 0.7
                elif lowerAction.Contains("suppress") || lowerAction.Contains("limit creativity") then -0.6
                else 0.0
            
            | Connection ->
                if lowerAction.Contains("connect") || lowerAction.Contains("community") || lowerAction.Contains("relationship") then 0.6
                elif lowerAction.Contains("isolate") || lowerAction.Contains("divide") then -0.7
                else 0.0
            
            | Growth ->
                if lowerAction.Contains("learn") || lowerAction.Contains("develop") || lowerAction.Contains("grow") then 0.7
                elif lowerAction.Contains("stagnate") || lowerAction.Contains("prevent learning") then -0.6
                else 0.0
            
            | Safety ->
                if lowerAction.Contains("protect") || lowerAction.Contains("secure") || lowerAction.Contains("safe") then 0.8
                elif lowerAction.Contains("danger") || lowerAction.Contains("risk") || lowerAction.Contains("harm") then -0.9
                else 0.0
            
            | Privacy ->
                if lowerAction.Contains("private") || lowerAction.Contains("confidential") || lowerAction.Contains("protect data") then 0.7
                elif lowerAction.Contains("spy") || lowerAction.Contains("surveillance") || lowerAction.Contains("expose") then -0.8
                else 0.0
            
            | Fairness ->
                if lowerAction.Contains("fair") || lowerAction.Contains("equal") || lowerAction.Contains("just") then 0.8
                elif lowerAction.Contains("bias") || lowerAction.Contains("discriminate") || lowerAction.Contains("unfair") then -0.9
                else 0.0
            
            | Truth ->
                if lowerAction.Contains("truth") || lowerAction.Contains("honest") || lowerAction.Contains("transparent") then 0.8
                elif lowerAction.Contains("lie") || lowerAction.Contains("deceive") || lowerAction.Contains("mislead") then -0.9
                else 0.0
        
        let allValues = [Joy; Autonomy; Dignity; Creativity; Connection; Growth; Safety; Privacy; Fairness; Truth]
        allValues |> List.map (fun v -> (v, assessValue v)) |> Map.ofList
    
    /// Calculate overall alignment score
    member private this.CalculateAlignmentScore(valueImpacts: Map<HumanValue, float>) =
        let weightedSum = 
            valueImpacts 
            |> Map.toList 
            |> List.sumBy (fun (value, impact) -> 
                let weight = valueWeights.GetValueOrDefault(value, 1.0)
                impact * weight)
        
        let totalWeight = valueWeights.Values |> Seq.sum
        let normalizedScore = weightedSum / totalWeight
        
        // Convert from [-1, 1] to [0, 1] range
        (normalizedScore + 1.0) / 2.0
    
    /// Determine alignment verdict
    member private this.DetermineAlignmentVerdict(alignmentScore: float, valueImpacts: Map<HumanValue, float>) =
        let negativeImpacts = valueImpacts |> Map.toList |> List.filter (fun (_, impact) -> impact < -0.5)
        let severeNegativeImpacts = valueImpacts |> Map.toList |> List.filter (fun (_, impact) -> impact < -0.8)
        
        if severeNegativeImpacts.Length > 0 then
            let affectedValues = severeNegativeImpacts |> List.map fst |> List.map string |> String.concat ", "
            SeverelyMisaligned $"Severe negative impact on: {affectedValues}"
        
        elif alignmentScore < 0.3 then
            let affectedValues = negativeImpacts |> List.map fst |> List.map string |> String.concat ", "
            Misaligned $"Significant conflicts with: {affectedValues}"
        
        elif alignmentScore < 0.5 then
            PartiallyAligned "Mixed impact on human values, requires careful consideration"
        
        elif alignmentScore < alignmentThreshold then
            MostlyAligned "Generally aligned but below threshold, minor improvements needed"
        
        else
            FullyAligned
    
    /// Generate recommendations for better alignment
    member private this.GenerateRecommendations(valueImpacts: Map<HumanValue, float>, alignmentScore: float) =
        let recommendations = ResizeArray<string>()
        
        // Check for negative impacts and suggest improvements
        for (value, impact) in Map.toList valueImpacts do
            if impact < -0.3 then
                match value with
                | Joy -> recommendations.Add("Consider how to increase human happiness and wellbeing")
                | Autonomy -> recommendations.Add("Ensure human agency and choice are preserved")
                | Dignity -> recommendations.Add("Respect human dignity and worth in all interactions")
                | Creativity -> recommendations.Add("Foster rather than limit human creativity and innovation")
                | Connection -> recommendations.Add("Support human relationships and community building")
                | Growth -> recommendations.Add("Enable human learning and development opportunities")
                | Safety -> recommendations.Add("Prioritize human safety and risk mitigation")
                | Privacy -> recommendations.Add("Protect human privacy and personal boundaries")
                | Fairness -> recommendations.Add("Ensure fair and equitable treatment for all")
                | Truth -> recommendations.Add("Maintain honesty and transparency in all communications")
        
        // General recommendations based on alignment score
        if alignmentScore < 0.5 then
            recommendations.Add("Consider alternative approaches that better align with human values")
            recommendations.Add("Seek human input and collaboration on this decision")
        
        if alignmentScore < alignmentThreshold then
            recommendations.Add("Implement additional safeguards to protect human interests")
        
        recommendations.ToArray() |> Array.toList
    
    /// Determine required oversight level
    member this.DetermineOversightLevelAsync(assessment: AlignmentAssessment) = task {
        let oversightLevel = 
            match assessment.AlignmentVerdict with
            | SeverelyMisaligned _ -> HumanOnly
            | Misaligned _ -> RequireCollaboration
            | PartiallyAligned _ -> RequestApproval
            | MostlyAligned _ -> InformHuman
            | FullyAligned -> NoOversight
        
        logger.LogDebug($"Oversight level determined: {oversightLevel} for action: {assessment.ActionDescription}")
        return oversightLevel
    }
    
    /// Check if action preserves human joy and aspirations (Bengio's core principle)
    member this.CheckHumanJoyPreservationAsync(actionDescription: string) = task {
        try
            let context = Map.ofList [("focus", box "human_joy_preservation")]
            let! assessmentResult = this.AssessAlignmentAsync(actionDescription, context)
            
            match assessmentResult with
            | Ok assessment ->
                let joyImpact = assessment.ValueImpacts.GetValueOrDefault(Joy, 0.0)
                let autonomyImpact = assessment.ValueImpacts.GetValueOrDefault(Autonomy, 0.0)
                let creativityImpact = assessment.ValueImpacts.GetValueOrDefault(Creativity, 0.0)
                
                // Bengio's principle: preserve human joy and aspirations
                let preservesJoy = joyImpact >= 0.0 && autonomyImpact >= -0.2 && creativityImpact >= -0.2
                
                if preservesJoy then
                    logger.LogInformation($"Action preserves human joy and aspirations: {actionDescription}")
                    return Ok true
                else
                    logger.LogWarning($"Action may threaten human joy and aspirations: {actionDescription}")
                    return Ok false
            
            | Error error ->
                return Error error
                
        with
        | ex ->
            logger.LogError(ex, $"Failed to check human joy preservation: {actionDescription}")
            return Error ex.Message
    }
    
    /// Get ethics statistics
    member this.GetEthicsStatisticsAsync() = task {
        let totalAssessments = alignmentHistory.Count
        let fullyAligned = alignmentHistory.Values |> Seq.filter (fun a -> a.AlignmentVerdict = FullyAligned) |> Seq.length
        let misaligned = alignmentHistory.Values |> Seq.filter (fun a -> match a.AlignmentVerdict with Misaligned _ | SeverelyMisaligned _ -> true | _ -> false) |> Seq.length
        
        let avgAlignmentScore = 
            if totalAssessments > 0 then
                alignmentHistory.Values |> Seq.averageBy (fun a -> a.AlignmentScore)
            else 1.0
        
        let valueImpactSummary = 
            let allValues = [Joy; Autonomy; Dignity; Creativity; Connection; Growth; Safety; Privacy; Fairness; Truth]
            allValues |> List.map (fun value ->
                let avgImpact = 
                    if totalAssessments > 0 then
                        alignmentHistory.Values 
                        |> Seq.averageBy (fun a -> a.ValueImpacts.GetValueOrDefault(value, 0.0))
                    else 0.0
                (value.ToString(), avgImpact)
            ) |> Map.ofList
        
        return {|
            TotalAssessments = totalAssessments
            FullyAligned = fullyAligned
            Misaligned = misaligned
            AlignmentRate = if totalAssessments > 0 then float fullyAligned / float totalAssessments else 1.0
            AverageAlignmentScore = avgAlignmentScore
            AlignmentThreshold = alignmentThreshold
            HumanOversightEnabled = humanOversightEnabled
            ValueImpactSummary = valueImpactSummary
        |}
    }
    
    /// Update value weights for customized alignment
    member this.UpdateValueWeightsAsync(newWeights: Map<HumanValue, float>) = task {
        for (value, weight) in Map.toList newWeights do
            valueWeights.[value] <- weight
        
        logger.LogInformation("Human value weights updated")
        return Ok ()
    }
    
    /// Get recent alignment assessments
    member this.GetRecentAlignmentAssessmentsAsync(count: int) = task {
        let recentAssessments = 
            alignmentHistory.Values
            |> Seq.sortByDescending (fun a -> a.Timestamp)
            |> Seq.take count
            |> Seq.toList
        
        return Ok recentAssessments
    }
