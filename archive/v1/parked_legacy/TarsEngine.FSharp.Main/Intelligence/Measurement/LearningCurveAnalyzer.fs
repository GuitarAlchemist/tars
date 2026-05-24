namespace TarsEngine.FSharp.Main.Intelligence.Measurement

open System
open System.Collections.Generic
open Microsoft.Extensions.Logging

/// <summary>
/// Analyzer for learning curves.
/// </summary>
type LearningCurveAnalyzer(logger: ILogger<LearningCurveAnalyzer>) =
    /// <summary>
    /// Analyzes a learning curve based on historical data points.
    /// </summary>
    /// <param name="metricName">The name of the metric being analyzed.</param>
    /// <param name="dataPoints">The historical data points.</param>
    /// <returns>A learning curve analysis.</returns>
    member this.AnalyzeLearningCurve(metricName: string, dataPoints: (DateTime * float) list) : LearningCurveAnalysis =
        logger.LogInformation("Analyzing learning curve for metric: {MetricName}", metricName)
        
        if dataPoints.Length < 2 then
            logger.LogWarning("Insufficient data points for learning curve analysis. At least 2 points are required.")
            {
                MetricName = metricName
                DataPoints = []
                LearningRate = 0.0
                LogLearningRate = 0.0
                PlateauValue = 0.0
                TimeToPlateauDays = 0.0
                CurveType = LearningCurveType.Unknown
                EfficiencyScore = 0.0
                ForecastValues = Map.empty
            }
        else
            // Sort data points by timestamp
            let sortedDataPoints = dataPoints |> List.sortBy fst
            
            // Calculate learning data points with derived metrics
            let learningDataPoints = 
                sortedDataPoints 
                |> List.pairwise 
                |> List.map (fun ((t1, v1), (t2, v2)) ->
                    let improvementRatio = 
                        if v1 = 0.0 then 0.0
                        else (v2 - v1) / v1
                    
                    let logV1 = if v1 > 0.0 then Math.Log(v1) else 0.0
                    let logV2 = if v2 > 0.0 then Math.Log(v2) else 0.0
                    let logLearningRate = 
                        if logV1 = 0.0 then 0.0
                        else (logV2 - logV1) / (t2 - t1).TotalDays
                    
                    {
                        Timestamp = t2
                        Value = v2
                        PreviousValue = v1
                        ImprovementRatio = improvementRatio
                        LogValue = logV2
                        LogPreviousValue = logV1
                        LogLearningRate = logLearningRate
                    })
            
            // Add the first data point
            let firstPoint = 
                let (t, v) = sortedDataPoints.[0]
                {
                    Timestamp = t
                    Value = v
                    PreviousValue = 0.0
                    ImprovementRatio = 0.0
                    LogValue = if v > 0.0 then Math.Log(v) else 0.0
                    LogPreviousValue = 0.0
                    LogLearningRate = 0.0
                }
            
            let allDataPoints = firstPoint :: learningDataPoints
            
            // Calculate overall learning rate
            let totalDays = (allDataPoints.[allDataPoints.Length - 1].Timestamp - allDataPoints.[0].Timestamp).TotalDays
            let totalDays = if totalDays = 0.0 then 1.0 else totalDays
            
            let firstValue = allDataPoints.[0].Value
            let lastValue = allDataPoints.[allDataPoints.Length - 1].Value
            
            let learningRate = 
                if firstValue = 0.0 then 0.0
                else (lastValue - firstValue) / (firstValue * totalDays)
            
            // Calculate logarithmic learning rate
            let firstLogValue = allDataPoints.[0].LogValue
            let lastLogValue = allDataPoints.[allDataPoints.Length - 1].LogValue
            
            let logLearningRate = 
                if firstLogValue = 0.0 then 0.0
                else (lastLogValue - firstLogValue) / totalDays
            
            // Determine curve type
            let curveType = 
                if learningRate > 0.05 then LearningCurveType.Exponential
                elif learningRate > 0.01 then LearningCurveType.Linear
                elif learningRate > 0.0 then LearningCurveType.Logarithmic
                elif learningRate = 0.0 then LearningCurveType.Plateau
                else LearningCurveType.Declining
            
            // Calculate plateau value and time to plateau
            let plateauValue = 
                match curveType with
                | LearningCurveType.Exponential -> lastValue * 2.0
                | LearningCurveType.Linear -> lastValue * 1.5
                | LearningCurveType.Logarithmic -> lastValue * 1.1
                | _ -> lastValue
            
            let timeToPlateauDays = 
                match curveType with
                | LearningCurveType.Exponential -> 30.0
                | LearningCurveType.Linear -> 60.0
                | LearningCurveType.Logarithmic -> 90.0
                | _ -> 0.0
            
            // Calculate efficiency score
            let efficiencyScore = 
                match curveType with
                | LearningCurveType.Exponential -> 0.9
                | LearningCurveType.Linear -> 0.7
                | LearningCurveType.Logarithmic -> 0.5
                | LearningCurveType.Plateau -> 0.3
                | _ -> 0.1
            
            // Generate forecast values
            let forecastDays = 30
            let lastTimestamp = allDataPoints.[allDataPoints.Length - 1].Timestamp
            
            let forecastValues = 
                [1..forecastDays]
                |> List.map (fun day ->
                    let forecastDate = lastTimestamp.AddDays(float day)
                    let forecastValue = 
                        match curveType with
                        | LearningCurveType.Exponential -> 
                            lastValue * Math.Exp(logLearningRate * float day)
                        | LearningCurveType.Linear -> 
                            lastValue * (1.0 + learningRate * float day)
                        | LearningCurveType.Logarithmic -> 
                            lastValue + (Math.Log(float day + 1.0) * learningRate * lastValue)
                        | LearningCurveType.Plateau -> 
                            lastValue
                        | _ -> 
                            lastValue * (1.0 - Math.Abs(learningRate) * float day)
                    (forecastDate, forecastValue))
                |> Map.ofList
            
            // Return the analysis
            {
                MetricName = metricName
                DataPoints = allDataPoints
                LearningRate = learningRate
                LogLearningRate = logLearningRate
                PlateauValue = plateauValue
                TimeToPlateauDays = timeToPlateauDays
                CurveType = curveType
                EfficiencyScore = efficiencyScore
                ForecastValues = forecastValues
            }
    
    /// <summary>
    /// Generates a learning curve forecast based on an existing analysis.
    /// </summary>
    /// <param name="analysis">The learning curve analysis.</param>
    /// <param name="forecastDays">The number of days to forecast.</param>
    /// <returns>A map of forecast dates to values.</returns>
    member this.GenerateForecast(analysis: LearningCurveAnalysis, forecastDays: int) : Map<DateTime, float> =
        logger.LogInformation("Generating forecast for metric: {MetricName}, days: {ForecastDays}", 
                             analysis.MetricName, forecastDays)
        
        if analysis.DataPoints.IsEmpty then
            logger.LogWarning("No data points available for forecast generation.")
            Map.empty
        else
            let lastDataPoint = analysis.DataPoints |> List.maxBy (fun dp -> dp.Timestamp)
            let lastTimestamp = lastDataPoint.Timestamp
            let lastValue = lastDataPoint.Value
            
            [1..forecastDays]
            |> List.map (fun day ->
                let forecastDate = lastTimestamp.AddDays(float day)
                let forecastValue = 
                    match analysis.CurveType with
                    | LearningCurveType.Exponential -> 
                        lastValue * Math.Exp(analysis.LogLearningRate * float day)
                    | LearningCurveType.Linear -> 
                        lastValue * (1.0 + analysis.LearningRate * float day)
                    | LearningCurveType.Logarithmic -> 
                        lastValue + (Math.Log(float day + 1.0) * analysis.LearningRate * lastValue)
                    | LearningCurveType.Plateau -> 
                        lastValue
                    | _ -> 
                        lastValue * (1.0 - Math.Abs(analysis.LearningRate) * float day)
                (forecastDate, forecastValue))
            |> Map.ofList
    
    /// <summary>
    /// Calculates the confidence intervals for a forecast.
    /// </summary>
    /// <param name="forecast">The forecast values.</param>
    /// <param name="confidenceLevel">The confidence level (0.0-1.0).</param>
    /// <returns>A map of forecast dates to confidence intervals (lower, upper).</returns>
    member this.CalculateConfidenceIntervals(forecast: Map<DateTime, float>, confidenceLevel: float) 
        : Map<DateTime, float * float> =
        
        logger.LogInformation("Calculating confidence intervals with confidence level: {ConfidenceLevel}", 
                             confidenceLevel)
        
        // Z-score for the given confidence level (approximation)
        let zScore = 
            match confidenceLevel with
            | cl when cl >= 0.99 -> 2.576
            | cl when cl >= 0.98 -> 2.326
            | cl when cl >= 0.95 -> 1.96
            | cl when cl >= 0.90 -> 1.645
            | cl when cl >= 0.85 -> 1.44
            | cl when cl >= 0.80 -> 1.282
            | _ -> 1.0
        
        // Calculate standard deviation as a percentage of the forecast value
        // This is a simplified approach; in a real implementation, we would use historical error rates
        let stdDevPercent = 0.1 // 10% standard deviation
        
        forecast
        |> Map.map (fun date value ->
            let stdDev = value * stdDevPercent
            let margin = zScore * stdDev
            (value - margin, value + margin))
    
    /// <summary>
    /// Evaluates the accuracy of a previous forecast against actual values.
    /// </summary>
    /// <param name="forecast">The forecast values.</param>
    /// <param name="actual">The actual values.</param>
    /// <returns>The forecast accuracy as a value between 0.0 and 1.0.</returns>
    member this.EvaluateForecastAccuracy(forecast: Map<DateTime, float>, actual: Map<DateTime, float>) : float =
        logger.LogInformation("Evaluating forecast accuracy")
        
        if forecast.IsEmpty || actual.IsEmpty then
            logger.LogWarning("Empty forecast or actual values for accuracy evaluation.")
            0.0
        else
            // Find common dates between forecast and actual
            let commonDates = 
                Set.intersect (forecast |> Map.keys |> Set.ofSeq) (actual |> Map.keys |> Set.ofSeq)
            
            if commonDates.IsEmpty then
                logger.LogWarning("No common dates between forecast and actual values.")
                0.0
            else
                // Calculate mean absolute percentage error (MAPE)
                let mape = 
                    commonDates
                    |> Seq.sumBy (fun date ->
                        let forecastValue = forecast.[date]
                        let actualValue = actual.[date]
                        
                        if actualValue = 0.0 then 0.0
                        else Math.Abs((actualValue - forecastValue) / actualValue))
                    |> fun total -> total / float commonDates.Count
                
                // Convert MAPE to accuracy (1.0 - MAPE), clamped between 0.0 and 1.0
                let accuracy = 1.0 - mape
                Math.Max(0.0, Math.Min(1.0, accuracy))
