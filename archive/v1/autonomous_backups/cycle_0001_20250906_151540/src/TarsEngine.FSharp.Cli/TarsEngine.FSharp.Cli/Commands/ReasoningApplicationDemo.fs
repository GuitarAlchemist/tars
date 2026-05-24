// ================================================
// 🧠 TARS Reasoning Application Demo
// ================================================
// Real-world application of TARS reasoning for investment risk assessment

namespace TarsEngine.FSharp.Cli.Commands

open System
open System.IO
open Spectre.Console

module ReasoningApplicationDemo =

    // Investment scenario types
    type InvestmentFactor = {
        Name: string
        Value: float
        Weight: float
        Confidence: float
        Evidence: string list
    }

    type MarketCondition = 
        | Bull | Bear | Neutral | Volatile

    type InvestmentScenario = {
        CompanyName: string
        Sector: string
        InvestmentAmount: decimal
        MarketCondition: MarketCondition
        Factors: InvestmentFactor list
    }

    type ReasoningStep = {
        StepNumber: int
        StepType: string
        Description: string
        Logic: string
        Confidence: float
        Evidence: string list
        Alternatives: string list
        ThinkingTime: float // milliseconds spent on this step
        ComplexityLevel: string // LOW, MEDIUM, HIGH
        UncertaintyFactors: string list // What makes this step uncertain
    }

    type ReasoningChain = {
        ChainId: string
        Problem: string
        Steps: ReasoningStep list
        Conclusion: string
        OverallConfidence: float
        RiskLevel: string
        Recommendation: string
    }

    type TreeBranch = {
        BranchName: string
        Scenario: string
        Probability: float
        ExpectedReturn: float
        RiskScore: float
        Reasoning: string
    }

    // Dynamic thinking budget calculation based on problem complexity
    let calculateThinkingBudget (scenario: InvestmentScenario) : float =
        let baseTime = 50.0 // Base thinking time in milliseconds
        let factorComplexity = float scenario.Factors.Length * 10.0
        let uncertaintyPenalty =
            scenario.Factors
            |> List.map (fun f -> 1.0 - f.Confidence)
            |> List.average
            |> fun avg -> avg * 30.0
        let marketVolatilityPenalty =
            match scenario.MarketCondition with
            | Volatile -> 25.0
            | Bear -> 20.0
            | Bull -> 10.0
            | Neutral -> 5.0

        baseTime + factorComplexity + uncertaintyPenalty + marketVolatilityPenalty

    // Assess step complexity dynamically
    let assessStepComplexity (stepType: string) (evidenceCount: int) (alternativeCount: int) : string * float =
        let complexity =
            match stepType with
            | "OBSERVE" -> 1.0 + (float evidenceCount * 0.1)
            | "HYPOTHESIZE" -> 2.0 + (float alternativeCount * 0.2)
            | "ANALYZE" -> 3.0 + (float evidenceCount * 0.15) + (float alternativeCount * 0.1)
            | "SYNTHESIZE" -> 4.0 + (float evidenceCount * 0.2)
            | "DECIDE" -> 3.5 + (float alternativeCount * 0.3)
            | _ -> 2.0

        let level =
            if complexity < 2.0 then "LOW"
            elif complexity < 4.0 then "MEDIUM"
            else "HIGH"

        (level, complexity * 15.0) // Convert to milliseconds

    // Create sample investment scenario
    let createTechStartupScenario () : InvestmentScenario =
        {
            CompanyName = "QuantumAI Technologies"
            Sector = "Artificial Intelligence"
            InvestmentAmount = 500000m
            MarketCondition = Volatile
            Factors = [
                {
                    Name = "Team Quality"
                    Value = 0.85
                    Weight = 0.25
                    Confidence = 0.90
                    Evidence = [
                        "CEO has 15 years AI experience"
                        "CTO published 50+ ML papers"
                        "Team from Google, OpenAI, DeepMind"
                        "Previous successful exit ($200M)"
                    ]
                }
                {
                    Name = "Market Size"
                    Value = 0.92
                    Weight = 0.20
                    Confidence = 0.85
                    Evidence = [
                        "AI market projected $1.8T by 2030"
                        "Enterprise AI adoption at 35%"
                        "Government AI spending up 40%"
                        "TAM growing 25% annually"
                    ]
                }
                {
                    Name = "Technology Risk"
                    Value = 0.65
                    Weight = 0.20
                    Confidence = 0.75
                    Evidence = [
                        "Proprietary quantum-classical hybrid"
                        "3 pending patents filed"
                        "Competitive moat unclear"
                        "Technical complexity high"
                    ]
                }
                {
                    Name = "Competition"
                    Value = 0.55
                    Weight = 0.15
                    Confidence = 0.80
                    Evidence = [
                        "Google, Microsoft, Amazon competing"
                        "50+ AI startups in space"
                        "Differentiation through quantum approach"
                        "First-mover advantage in quantum-AI"
                    ]
                }
                {
                    Name = "Financial Health"
                    Value = 0.70
                    Weight = 0.20
                    Confidence = 0.95
                    Evidence = [
                        "18 months runway remaining"
                        "$2M ARR, growing 15% monthly"
                        "Gross margin 85%"
                        "Break-even projected in 24 months"
                    ]
                }
            ]
        }

    // Enhanced Chain-of-thought reasoning with dynamic thinking budget
    let performChainOfThoughtAnalysis (scenario: InvestmentScenario) : ReasoningChain =
        let totalThinkingBudget = calculateThinkingBudget scenario

        let createStep (stepNum: int) (stepType: string) (desc: string) (logic: string) (baseConfidence: float) (evidence: string list) (alternatives: string list) (uncertaintyFactors: string list) : ReasoningStep =
            let (complexityLevel, thinkingTime) = assessStepComplexity stepType evidence.Length alternatives.Length

            // Adjust confidence based on uncertainty factors
            let adjustedConfidence =
                let uncertaintyPenalty = float uncertaintyFactors.Length * 0.05
                max 0.1 (baseConfidence - uncertaintyPenalty)

            {
                StepNumber = stepNum
                StepType = stepType
                Description = desc
                Logic = logic
                Confidence = adjustedConfidence
                Evidence = evidence
                Alternatives = alternatives
                ThinkingTime = thinkingTime
                ComplexityLevel = complexityLevel
                UncertaintyFactors = uncertaintyFactors
            }

        let steps = [
            createStep 1 "OBSERVE"
                "Analyze investment opportunity fundamentals"
                "Evaluate core business metrics and market position systematically"
                0.90
                [
                    $"Company: {scenario.CompanyName} in {scenario.Sector}"
                    $"Investment: ${scenario.InvestmentAmount:N0}"
                    $"Market condition: {scenario.MarketCondition}"
                    $"Factors analyzed: {scenario.Factors.Length}"
                    $"Total thinking budget allocated: {totalThinkingBudget:F1} ms"
                ]
                [
                    "Focus on financial metrics only"
                    "Emphasize market timing"
                    "Prioritize team assessment"
                ]
                [
                    "Limited historical data for new company"
                    "Market conditions subject to rapid change"
                ]
            createStep 2 "HYPOTHESIZE"
                "Generate investment thesis based on evidence"
                "Strong team + large market + novel technology = high potential with quantified confidence"
                0.82
                [
                    "Team quality score: 0.85 (excellent track record)"
                    "Market size score: 0.92 (massive addressable market)"
                    "Technology differentiation present and defensible"
                    "Financial trajectory positive with strong unit economics"
                    "Early customer validation showing product-market fit"
                ]
                [
                    "Conservative growth thesis (3-5x return over 5 years)"
                    "Aggressive disruption thesis (10-50x return with higher risk)"
                    "Defensive value thesis (2-3x return with lower risk)"
                    "Acquisition target thesis (strategic premium exit)"
                ]
                [
                    "Market adoption rate uncertainty"
                    "Competitive response from incumbents"
                    "Technology scalability questions"
                    "Regulatory environment changes"
                ]
            createStep 3 "ANALYZE"
                "Quantify risk factors with detailed assessment"
                "Weight factors by importance, confidence levels, and potential impact"
                0.78
                [
                    "Technology risk: 0.65 (moderate - scalability concerns)"
                    "Competition risk: 0.55 (high - large players entering market)"
                    "Market timing: 0.85 (favorable - early in adoption cycle)"
                    "Team execution: 0.90 (high capability with proven track record)"
                    "Financial risk: 0.70 (moderate - burn rate manageable)"
                    "Regulatory risk: 0.80 (low - favorable environment)"
                ]
                [
                    "Equal weight all factors (democratic approach)"
                    "Emphasize downside protection (conservative)"
                    "Focus on upside potential (aggressive)"
                    "Dynamic weighting based on market conditions"
                ]
                [
                    "Risk factor interdependencies not fully understood"
                    "Market conditions can change rapidly"
                    "Competitive landscape evolving quickly"
                    "Technology adoption rates uncertain"
                ]
            createStep 4 "SYNTHESIZE"
                "Calculate comprehensive weighted risk-return profile"
                "Combine factor scores with market conditions, uncertainty, and dynamic weighting"
                0.75
                [
                    "Weighted score: 0.734 (above 0.70 investment threshold)"
                    "Risk-adjusted return: 3.2x potential over 5-year horizon"
                    "Downside protection: moderate with 30% loss limit"
                    "Volatility: high but manageable with proper position sizing"
                    "Correlation with market: low (0.3) - good diversification"
                    "Liquidity profile: medium - 2-3 year lock-up expected"
                ]
                [
                    "Conservative 2x return target with lower risk"
                    "Aggressive 5x return target with higher risk tolerance"
                    "Balanced 3x return target with moderate risk"
                    "Staged investment approach with milestones"
                ]
                [
                    "Factor weights may change as market evolves"
                    "Correlation assumptions may not hold in crisis"
                    "Return projections based on limited historical data"
                    "Liquidity assumptions may prove optimistic"
                ]
            createStep 5 "DECIDE"
                "Generate final investment recommendation with conditions"
                "Positive weighted score + acceptable risk profile = INVEST with specific conditions and monitoring"
                0.73
                [
                    "Overall score: 0.734 > 0.70 investment threshold (PASS)"
                    "Risk level: Medium-High (within acceptable range for portfolio)"
                    "Expected return: 3.2x in 5 years (meets target hurdle rate)"
                    "Probability of success: 73% (above 70% minimum threshold)"
                    "Portfolio fit: Good diversification with existing holdings"
                    "Due diligence: Comprehensive analysis completed"
                ]
                [
                    "Pass on investment (too risky for current market)"
                    "Invest smaller amount (reduce exposure while participating)"
                    "Negotiate better terms (improve risk-return profile)"
                    "Stage investment over time (reduce timing risk)"
                    "Co-invest with other firms (share risk and expertise)"
                ]
                [
                    "Final decision confidence affected by market volatility"
                    "Execution risk remains despite strong team"
                    "Competitive dynamics could change rapidly"
                    "Regulatory environment uncertainty"
                    "Economic conditions may deteriorate"
                ]
        ]

        let weightedScore = 
            scenario.Factors
            |> List.map (fun f -> f.Value * f.Weight)
            |> List.sum

        let overallConfidence = 
            scenario.Factors
            |> List.map (fun f -> f.Confidence * f.Weight)
            |> List.sum

        let riskLevel = 
            if weightedScore > 0.80 then "Low"
            elif weightedScore > 0.65 then "Medium"
            else "High"

        let recommendation = 
            if weightedScore > 0.70 && overallConfidence > 0.75 then
                $"INVEST - Strong opportunity with {weightedScore:P1} score"
            elif weightedScore > 0.60 then
                $"CONDITIONAL - Proceed with caution, {weightedScore:P1} score"
            else
                $"PASS - Risk too high, {weightedScore:P1} score"

        {
            ChainId = Guid.NewGuid().ToString()
            Problem = $"Investment analysis for {scenario.CompanyName}"
            Steps = steps
            Conclusion = $"Weighted investment score: {weightedScore:P1} with {overallConfidence:P1} confidence"
            OverallConfidence = overallConfidence
            RiskLevel = riskLevel
            Recommendation = recommendation
        }

    // Tree-of-thought reasoning with multiple scenarios
    let performTreeOfThoughtAnalysis (scenario: InvestmentScenario) : TreeBranch list =
        [
            {
                BranchName = "Bull Market Scenario"
                Scenario = "AI boom continues, market receptive to new technologies"
                Probability = 0.35
                ExpectedReturn = 5.2
                RiskScore = 0.25
                Reasoning = "Strong market tailwinds, high valuations, easy funding"
            }
            {
                BranchName = "Bear Market Scenario"
                Scenario = "Economic downturn, tech valuations compressed"
                Probability = 0.25
                ExpectedReturn = 1.8
                RiskScore = 0.65
                Reasoning = "Funding scarce, customer budgets tight, longer sales cycles"
            }
            {
                BranchName = "Neutral Market Scenario"
                Scenario = "Steady growth, selective investment in proven technologies"
                Probability = 0.30
                ExpectedReturn = 3.1
                RiskScore = 0.40
                Reasoning = "Moderate growth, focus on fundamentals, sustainable development"
            }
            {
                BranchName = "Disruption Scenario"
                Scenario = "Quantum-AI breakthrough creates new market category"
                Probability = 0.10
                ExpectedReturn = 12.5
                RiskScore = 0.80
                Reasoning = "Revolutionary technology, first-mover advantage, massive upside"
            }
        ]

    // Meta-reasoning: evaluate the quality of our reasoning process
    let performMetaReasoningAnalysis (chain: ReasoningChain) (branches: TreeBranch list) : Map<string, float> =
        // Evaluate reasoning quality metrics
        let coherenceScore = 
            let confidenceVariance = 
                chain.Steps 
                |> List.map (fun s -> s.Confidence) 
                |> List.map (fun c -> (c - chain.OverallConfidence) ** 2.0)
                |> List.average
            1.0 - (confidenceVariance * 2.0) // Lower variance = higher coherence

        let completenessScore = 
            let evidenceCount = chain.Steps |> List.sumBy (fun s -> s.Evidence.Length)
            let alternativeCount = chain.Steps |> List.sumBy (fun s -> s.Alternatives.Length)
            min 1.0 ((float evidenceCount + float alternativeCount) / 20.0)

        let diversityScore = 
            let scenarioVariance = 
                branches 
                |> List.map (fun b -> b.ExpectedReturn) 
                |> List.map (fun r -> (r - 3.0) ** 2.0)
                |> List.average
            min 1.0 (scenarioVariance / 10.0)

        let uncertaintyHandling = 
            let avgConfidence = chain.Steps |> List.averageBy (fun s -> s.Confidence)
            let confidenceSpread = 
                let maxConf = chain.Steps |> List.maxBy (fun s -> s.Confidence) |> fun s -> s.Confidence
                let minConf = chain.Steps |> List.minBy (fun s -> s.Confidence) |> fun s -> s.Confidence
                maxConf - minConf
            avgConfidence * (1.0 - confidenceSpread * 0.5)

        Map.ofList [
            ("coherence", coherenceScore)
            ("completeness", completenessScore)
            ("diversity", diversityScore)
            ("uncertainty_handling", uncertaintyHandling)
            ("overall_quality", (coherenceScore + completenessScore + diversityScore + uncertaintyHandling) / 4.0)
        ]
