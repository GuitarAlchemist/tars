module Tars.Tests.DemoGeneratorTests

open Xunit
open Tars.Core.DemoGenerator

// ============================================================================
// DemoFormat Tests
// ============================================================================

[<Fact>]
let ``DemoFormat types are correctly defined`` () =
    let formats = [ MarkdownReport; HtmlPresentation; JsonSummary; PlainText ]
    Assert.Equal(4, formats.Length)

// ============================================================================
// DemoSection Tests
// ============================================================================

[<Fact>]
let ``DemoSection has correct structure`` () =
    let section =
        { Stage = "Vision"
          Title = "Vision Phase"
          Summary = "Completed vision"
          Artifacts = [ "spec.md" ]
          Highlights = [ "Feature A" ] }

    Assert.Equal("Vision", section.Stage)
    Assert.Equal(1, section.Artifacts.Length)
    Assert.Equal(1, section.Highlights.Length)

// ============================================================================
// DemoBuilder Tests
// ============================================================================

[<Fact>]
let ``DemoBuilder creates demo with sections`` () =
    let builder = DemoBuilder()
    builder.SetProject("test-1", "Test Project")
    builder.SetTitle("Test Demo")
    builder.SetSubtitle("Test Subtitle")
    builder.AddSection("Vision", "Vision Phase", "Completed vision work")
    builder.AddArtifact("Vision", "spec.md")
    builder.AddHighlight("Vision", "Key decision made")

    let demo = builder.Build(MarkdownReport)

    Assert.Equal("test-1", demo.ProjectId)
    Assert.Equal("Test Project", demo.ProjectName)
    Assert.Equal("Test Demo", demo.Title)
    Assert.True(demo.Subtitle.IsSome)
    Assert.Equal(1, demo.Sections.Length)
    Assert.Equal(1, demo.Sections.[0].Artifacts.Length)
    Assert.Equal(1, demo.Sections.[0].Highlights.Length)

// ============================================================================
// Renderer Tests
// ============================================================================

[<Fact>]
let ``renderMarkdown includes project info`` () =
    let builder = DemoBuilder()
    builder.SetProject("md-test", "Markdown Test")
    builder.SetTitle("MD Demo")
    builder.AddSection("Dev", "Development", "Built features")

    let demo = builder.Build(MarkdownReport)
    let output = renderMarkdown demo

    Assert.Contains("# MD Demo", output)
    Assert.Contains("md-test", output)
    Assert.Contains("## Development", output)

[<Fact>]
let ``renderHtml produces valid HTML structure`` () =
    let builder = DemoBuilder()
    builder.SetProject("html-test", "HTML Test")
    builder.SetTitle("HTML Demo")
    builder.AddSection("QA", "Quality Assurance", "Tested all features")

    let demo = builder.Build(HtmlPresentation)
    let output = renderHtml demo

    Assert.Contains("<!DOCTYPE html>", output)
    Assert.Contains("<h1>HTML Demo</h1>", output)
    Assert.Contains("</body></html>", output)

[<Fact>]
let ``renderJson produces valid JSON`` () =
    let builder = DemoBuilder()
    builder.SetProject("json-test", "JSON Test")
    builder.SetTitle("JSON Demo")
    builder.AddSection("Demo", "Demo Phase", "Created demo")

    let demo = builder.Build(JsonSummary)
    let output = renderJson demo

    Assert.Contains("\"projectId\":\"json-test\"", output)
    Assert.Contains("\"title\":\"JSON Demo\"", output)
    Assert.Contains("\"sections\":[", output)

[<Fact>]
let ``renderPlainText produces formatted text`` () =
    let builder = DemoBuilder()
    builder.SetProject("text-test", "Text Test")
    builder.SetTitle("Text Demo")
    builder.AddSection("Vision", "Vision", "Defined vision")

    let demo = builder.Build(PlainText)
    let output = renderPlainText demo

    Assert.Contains("TEXT DEMO", output) // Title is uppercased
    Assert.Contains("VISION", output)

[<Fact>]
let ``render dispatches to correct renderer`` () =
    let builder = DemoBuilder()
    builder.SetProject("dispatch", "Dispatch Test")
    builder.SetTitle("Dispatch Demo")

    let mdDemo = builder.Build(MarkdownReport)
    let htmlDemo = builder.Build(HtmlPresentation)

    Assert.Contains("# Dispatch Demo", render mdDemo)
    Assert.Contains("<!DOCTYPE html>", render htmlDemo)

// ============================================================================
// saveDemo Tests
// ============================================================================

[<Fact>]
let ``saveDemo returns error for invalid path`` () =
    let builder = DemoBuilder()
    builder.SetProject("save-test", "Save Test")
    let demo = builder.Build(MarkdownReport)

    // Try to save to an invalid path (with invalid characters on Windows)
    let result = saveDemo demo "/definitely/invalid<>path/demo.md"

    Assert.True(Result.isError result)
