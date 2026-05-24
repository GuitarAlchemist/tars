module TarsEngine.SelfImprovement.Tests.ModelRecommendationServiceTests

open System
open System.IO
open Microsoft.Extensions.Logging.Abstractions
open Xunit
open TarsEngine.FSharp.SelfImprovement.Services

type ModelRecommendationServiceTests() =

    let writeTempFile (contents: string) =
        let path = Path.Combine(Path.GetTempPath(), $"tars-model-{Guid.NewGuid():N}.yaml")
        File.WriteAllText(path, contents)
        path

    [<Fact>]
    member _.``loadRecommendations resolves placeholders and env hints`` () =
        let sampleYaml = """
best_models:
  sample:
    display_name: Example Model
    provider: custom-provider
    model: ${CUSTOM_MODEL:-default-model}
    capabilities: [reasoning, code]
    requires_api_key: true
    env:
      api_key: CUSTOM_API_KEY
      endpoint: ${CUSTOM_ENDPOINT:-https://example.local}
      optional:
        group_id: ${GROUP_ID:-default-group}
    notes: "Use for tests"
"""

        let tempPath = writeTempFile sampleYaml
        let originalModelVar = Environment.GetEnvironmentVariable("CUSTOM_MODEL")
        let originalEndpointVar = Environment.GetEnvironmentVariable("CUSTOM_ENDPOINT")
        let originalGroupVar = Environment.GetEnvironmentVariable("GROUP_ID")

        try
            Environment.SetEnvironmentVariable("CUSTOM_MODEL", "overridden-model")
            Environment.SetEnvironmentVariable("CUSTOM_ENDPOINT", null)
            Environment.SetEnvironmentVariable("GROUP_ID", "primary-group")

            let logger = NullLoggerFactory.Instance.CreateLogger("test")
            let recommendations = ModelRecommendationService.loadRecommendations logger (Some tempPath)

            Assert.True(recommendations.ContainsKey("sample"), "Expected sample recommendation to be present");
            let sample = recommendations.["sample"]
            Assert.Equal("Example Model", sample.DisplayName)
            Assert.Equal("custom-provider", sample.Provider)
            Assert.Equal("overridden-model", sample.Model)
            Assert.True(sample.RequiresApiKey)
            Assert.Equal("https://example.local", sample.Environment.["endpoint"])
            Assert.Equal("CUSTOM_API_KEY", sample.Environment.["api_key"])
            Assert.Equal("primary-group", sample.OptionalEnvironment.["group_id"])
            Assert.Contains("reasoning", sample.Capabilities)
            Assert.Contains("code", sample.Capabilities)
        finally
            Environment.SetEnvironmentVariable("CUSTOM_MODEL", originalModelVar)
            Environment.SetEnvironmentVariable("CUSTOM_ENDPOINT", originalEndpointVar)
            Environment.SetEnvironmentVariable("GROUP_ID", originalGroupVar)
            if File.Exists(tempPath) then File.Delete(tempPath)
