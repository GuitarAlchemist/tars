module TarsEngine.FSharp.SelfBuildingUI.Tests.SelfBuildingUiTests

open System
open System.Text.Json
open TarsEngine.FSharp.SelfBuildingUIApp.TarsSelfBuildingUI
open Xunit

[<Fact>]
let ``initializeConfig returns expected defaults`` () =
    let config = initializeConfig ()

    Assert.Equal("TARS-Dark", config.Theme)
    Assert.True(config.AutoGenerate)
    Assert.Contains("Dashboard", config.Components)

[<Fact>]
let ``buildUI generates matching components`` () =
    let config = initializeConfig ()
    let components = buildUI config

    let names = components |> List.map (fun c -> c.Name)

    Assert.Equal(config.Components.Length, components.Length)
    Assert.Contains("Dashboard", names)
    Assert.Contains("ControlPanel", names)
    Assert.Contains("Monitoring", names)

[<Fact>]
let ``updateUIState increments build count and timestamp`` () =
    let config = initializeConfig ()
    let initialState = {
        Config = config
        Components = []
        LastBuilt = DateTime.MinValue
        BuildCount = 0
        UserFeedback = []
    }

    let components = buildUI config
    let updated = updateUIState initialState components

    Assert.Equal(1, updated.BuildCount)
    Assert.Equal(components.Length, updated.Components.Length)
    Assert.True(updated.LastBuilt > initialState.LastBuilt)

[<Fact>]
let ``serializeToJson produces valid json`` () =
    let components = buildUI (initializeConfig ())
    let json = serializeToJson components

    Assert.False(String.IsNullOrWhiteSpace(json))

    let deserialized =
        JsonSerializer.Deserialize<UIComponent list>(json, JsonSerializerOptions(PropertyNameCaseInsensitive = true))

    Assert.NotNull(deserialized)
    Assert.Equal(components.Length, deserialized |> List.length)
