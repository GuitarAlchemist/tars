namespace Tars.Tests

open System.Text.Json
open Xunit
open Tars.Llm

module JsonParsingTests =

    [<Fact>]
    let ``JsonParsing extracts array from fenced output`` () =
        let text =
            "Here is JSON:\n```json\n[{\"subject\":\"A\",\"predicate\":\"supports\",\"object\":\"B\",\"confidence\":0.9}]\n```"

        match JsonParsing.tryParseElement text with
        | Ok elem -> Assert.Equal(JsonValueKind.Array, elem.ValueKind)
        | Error err -> Assert.True(false, err)

    [<Fact>]
    let ``JsonParsing extracts object after prefix`` () =
        let text = "ACT: INFORM: {\"passed\":true,\"confidence\":0.7}"

        match JsonParsing.tryParseElement text with
        | Ok elem -> Assert.Equal(JsonValueKind.Object, elem.ValueKind)
        | Error err -> Assert.True(false, err)

    [<Fact>]
    let ``JsonParsing extracts embedded JSON span from noisy text`` () =
        let text =
            "NOTE: task result {\"result\":\"ok\",\"confidence\":0.83} should be recorded."

        match JsonParsing.tryParseElement text with
        | Ok elem -> Assert.Equal(JsonValueKind.Object, elem.ValueKind)
        | Error err -> Assert.True(false, err)

    [<Fact>]
    let ``JsonParsing handles single triple object`` () =
        let text =
            "{\"subject\":\"A\",\"predicate\":\"supports\",\"object\":\"B\",\"confidence\":0.9}"

        match JsonParsing.tryParseElement text with
        | Ok elem -> Assert.Equal(JsonValueKind.Object, elem.ValueKind)
        | Error err -> Assert.True(false, err)

    [<Fact>]
    let ``JsonParsing handles wrapped triples array`` () =
        let text =
            "{\"triples\":[{\"subject\":\"A\",\"predicate\":\"supports\",\"object\":\"B\"}]}"

        match JsonParsing.tryParseElement text with
        | Ok elem ->
            let mutable prop = Unchecked.defaultof<JsonElement>
            Assert.Equal(JsonValueKind.Object, elem.ValueKind)
            Assert.True(elem.TryGetProperty("triples", &prop))
        | Error err -> Assert.True(false, err)
