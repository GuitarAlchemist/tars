namespace Tars.Evolution

open System
open System.IO
open System.Text.Json
open System.Text.Json.Serialization
open System.Collections.Concurrent

module private StoreUtils =
    let jsonOptions =
        let o = JsonSerializerOptions(JsonSerializerDefaults.General)
        o.Converters.Add(JsonFSharpConverter())
        o.WriteIndented <- true
        o

    let getDefaultPromotionDir () =
        let dir = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".tars", "promotion")
        if not (Directory.Exists dir) then
            Directory.CreateDirectory dir |> ignore
        dir

/// DTO for JSON serialization of WeightedRule to match legacy format
type WeightedRuleDto = {
    PatternId: string
    PatternName: string
    Level: string
    RawScore: int
    Weight: float
    Confidence: float
    SuccessRate: float
    SelectionCount: int
    Source: string
    LastUpdated: string
}

type InMemoryPromotionStore() =
    let recurrenceStore = ConcurrentDictionary<string, RecurrenceRecord>()
    let lineageStore = ConcurrentDictionary<string, LineageRecord>()
    let mutable weightsStore : WeightedRule list = []

    interface IPromotionStore with
        member _.GetRecurrenceRecords() = recurrenceStore.Values |> Seq.toList
        member _.UpsertRecurrenceRecord(r) = recurrenceStore.[r.PatternName] <- r
        member _.GetLineageRecords() = lineageStore.Values |> Seq.toList
        member _.AddLineageRecord(l) = lineageStore.[l.Id] <- l
        member _.GetWeights() = weightsStore
        member _.SaveWeights(w) = weightsStore <- w

type FilePromotionStore(promotionDir: string) =
    let recurrencePath = Path.Combine(promotionDir, "recurrence.json")
    let lineagePath = Path.Combine(promotionDir, "lineage.json")
    let weightsPath = Path.Combine(promotionDir, "weights.json")

    let toDto (r: WeightedRule) : WeightedRuleDto =
        { PatternId = r.PatternId
          PatternName = r.PatternName
          Level = PromotionLevel.label r.Level
          RawScore = r.RawScore
          Weight = r.Weight
          Confidence = r.Confidence
          SuccessRate = r.SuccessRate
          SelectionCount = r.SelectionCount
          Source = match r.Source with
                   | Tars -> "tars" | GuitarAlchemist -> "guitar_alchemist"
                   | MachinDeOuf -> "ix" | Evolved -> "evolved" | Manual -> "manual"
          LastUpdated = r.LastUpdated.ToString("o") }

    let fromDto (dto: WeightedRuleDto) : WeightedRule =
        let level =
            match dto.Level with
            | "helper" -> Helper | "builder" -> Builder
            | "dsl_clause" -> DslClause | "grammar_rule" -> GrammarRule
            | _ -> Implementation
        let source =
            match dto.Source with
            | "guitar_alchemist" -> GuitarAlchemist | "ix" -> MachinDeOuf
            | "evolved" -> Evolved | "manual" -> Manual | _ -> Tars
        { PatternId = dto.PatternId
          PatternName = dto.PatternName
          Level = level
          RawScore = dto.RawScore
          Weight = dto.Weight
          Confidence = dto.Confidence
          SuccessRate = dto.SuccessRate
          SelectionCount = dto.SelectionCount
          Source = source
          LastUpdated = try DateTime.Parse(dto.LastUpdated) with _ -> DateTime.UtcNow }

    interface IPromotionStore with
        member _.GetRecurrenceRecords() =
            try
                if File.Exists recurrencePath then
                    let json = File.ReadAllText(recurrencePath)
                    JsonSerializer.Deserialize<RecurrenceRecord list>(json, StoreUtils.jsonOptions)
                else []
            with _ -> []

        member this.UpsertRecurrenceRecord(r) =
            let store = this :> IPromotionStore
            let records = store.GetRecurrenceRecords()
            let updated =
                match records |> List.tryFindIndex (fun x -> x.PatternName = r.PatternName) with
                | Some idx -> records |> List.updateAt idx r
                | None -> r :: records
            if not (Directory.Exists promotionDir) then Directory.CreateDirectory promotionDir |> ignore
            File.WriteAllText(recurrencePath, JsonSerializer.Serialize(updated, StoreUtils.jsonOptions))

        member _.GetLineageRecords() =
            try
                if File.Exists lineagePath then
                    let json = File.ReadAllText(lineagePath)
                    JsonSerializer.Deserialize<LineageRecord list>(json, StoreUtils.jsonOptions)
                else []
            with _ -> []

        member this.AddLineageRecord(l) =
            let store = this :> IPromotionStore
            let records = store.GetLineageRecords()
            let updated = l :: records
            if not (Directory.Exists promotionDir) then Directory.CreateDirectory promotionDir |> ignore
            File.WriteAllText(lineagePath, JsonSerializer.Serialize(updated, StoreUtils.jsonOptions))

        member this.GetWeights() =
            try
                if File.Exists weightsPath then
                    let json = File.ReadAllText(weightsPath)
                    let dtos = JsonSerializer.Deserialize<WeightedRuleDto list>(json, StoreUtils.jsonOptions)
                    dtos |> List.map fromDto
                else []
            with _ -> []

        member this.SaveWeights(weights) =
            let dtos = weights |> List.map toDto
            if not (Directory.Exists promotionDir) then Directory.CreateDirectory promotionDir |> ignore
            File.WriteAllText(weightsPath, JsonSerializer.Serialize(dtos, StoreUtils.jsonOptions))

module PromotionStore =
    let createInMemory () = InMemoryPromotionStore() :> IPromotionStore
    let createFileNamed (dir: string) = FilePromotionStore(dir) :> IPromotionStore
    let createDefault () = createFileNamed (StoreUtils.getDefaultPromotionDir())
