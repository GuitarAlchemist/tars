namespace Tars.Registry

open System.Text.Json.Nodes

/// Record describing a single discovered Tars skill.
///
/// `Schema` returns the input/output JSON schema (lazy so reflection on
/// schema-producing methods stays cheap when only the listing is needed).
/// `Handler` is the invoke-by-reflection dispatch function — it accepts
/// a JsonNode payload and returns a Result.
type TarsSkill = {
    Name: string
    Domain: string
    Description: string
    Schema: unit -> JsonNode
    Handler: JsonNode -> Result<JsonNode, string>
}
