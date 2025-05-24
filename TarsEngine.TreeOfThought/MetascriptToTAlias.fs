// Alias module for MetascriptToT
module TarsEngine.FSharp.MetascriptToT

// Re-export types and functions from the compatibility module
open TarsEngine.FSharp.Core.Compatibility.MetascriptToTCompat

// Re-export types
type MetascriptEvaluationMetrics = TarsEngine.FSharp.Core.Compatibility.MetascriptToTCompat.MetascriptEvaluationMetrics
type MetascriptThoughtNode = TarsEngine.FSharp.Core.Compatibility.MetascriptToTCompat.MetascriptThoughtNode

// Re-export functions
let createNode = TarsEngine.FSharp.Core.Compatibility.MetascriptToTCompat.createNode
let addChild = TarsEngine.FSharp.Core.Compatibility.MetascriptToTCompat.addChild
let evaluateNode = TarsEngine.FSharp.Core.Compatibility.MetascriptToTCompat.evaluateNode
let pruneNode = TarsEngine.FSharp.Core.Compatibility.MetascriptToTCompat.pruneNode
let addMetadata = TarsEngine.FSharp.Core.Compatibility.MetascriptToTCompat.addMetadata
let getMetadata = TarsEngine.FSharp.Core.Compatibility.MetascriptToTCompat.getMetadata
let getScore = TarsEngine.FSharp.Core.Compatibility.MetascriptToTCompat.getScore
let createMetrics = TarsEngine.FSharp.Core.Compatibility.MetascriptToTCompat.createMetrics
let createWeightedMetrics = TarsEngine.FSharp.Core.Compatibility.MetascriptToTCompat.createWeightedMetrics
let normalizeMetrics = TarsEngine.FSharp.Core.Compatibility.MetascriptToTCompat.normalizeMetrics
let compareMetrics = TarsEngine.FSharp.Core.Compatibility.MetascriptToTCompat.compareMetrics
let thresholdMetrics = TarsEngine.FSharp.Core.Compatibility.MetascriptToTCompat.thresholdMetrics
let toJson = TarsEngine.FSharp.Core.Compatibility.MetascriptToTCompat.toJson
let treeToJson = TarsEngine.FSharp.Core.Compatibility.MetascriptToTCompat.treeToJson

// Re-export nested modules
module Evaluation =
    let createMetrics = TarsEngine.FSharp.Core.Compatibility.MetascriptToTCompat.createMetrics
    let createWeightedMetrics = TarsEngine.FSharp.Core.Compatibility.MetascriptToTCompat.createWeightedMetrics
    let normalizeMetrics = TarsEngine.FSharp.Core.Compatibility.MetascriptToTCompat.normalizeMetrics
    let compareMetrics = TarsEngine.FSharp.Core.Compatibility.MetascriptToTCompat.compareMetrics
    let thresholdMetrics = TarsEngine.FSharp.Core.Compatibility.MetascriptToTCompat.thresholdMetrics
    let toJson = TarsEngine.FSharp.Core.Compatibility.MetascriptToTCompat.toJson

module ThoughtTree =
    let createNode = TarsEngine.FSharp.Core.Compatibility.MetascriptToTCompat.createNode
    let addChild = TarsEngine.FSharp.Core.Compatibility.MetascriptToTCompat.addChild
    let evaluateNode = TarsEngine.FSharp.Core.Compatibility.MetascriptToTCompat.evaluateNode
    let pruneNode = TarsEngine.FSharp.Core.Compatibility.MetascriptToTCompat.pruneNode
    let addMetadata = TarsEngine.FSharp.Core.Compatibility.MetascriptToTCompat.addMetadata
    let getMetadata = TarsEngine.FSharp.Core.Compatibility.MetascriptToTCompat.getMetadata
    let getScore = TarsEngine.FSharp.Core.Compatibility.MetascriptToTCompat.getScore
    let toJson = TarsEngine.FSharp.Core.Compatibility.MetascriptToTCompat.treeToJson
