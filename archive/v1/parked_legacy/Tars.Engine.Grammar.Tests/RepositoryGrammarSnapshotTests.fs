namespace Tars.Engine.Grammar.Tests

open System
open System.IO
open System.Security.Cryptography
open System.Text
open Xunit
open Tars.Engine.Grammar

module RepositoryGrammarSnapshotTests =

    let private repoRoot =
        Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", ".."))

    let private grammarDirectory =
        Path.Combine(repoRoot, ".tars", "evolution", "grammars", "base")

    let private trackedGrammars =
        [ "MiniQuery.tars", "09d6badda7d7a0d471fd45095c171e9e79f7aaed6ba67b5bde15cee8a666674d"
          "RFC3986_URI.tars", "66cecf2262e21aba953ab5c512851a7fdf7c9e0343140db876162d83647b1396"
          "Wolfram.tars", "b5f4d5227bbe37d3a14bcbd4852bed3eda92d553eeb37317b62e4745971e80aa" ]

    let private computeHash (text: string) =
        use sha = SHA256.Create()
        text
        |> Encoding.UTF8.GetBytes
        |> sha.ComputeHash
        |> Array.map (fun b -> b.ToString("x2"))
        |> String.concat ""

    [<Fact>]
    let ``Repository grammars are present and unchanged`` () =
        for (fileName, expectedHash) in trackedGrammars do
            let path = Path.Combine(grammarDirectory, fileName)
            Assert.True(File.Exists(path), $"Missing grammar file {path}")

            let source = GrammarSource.External(FileInfo(path))
            let content = GrammarSource.getContent source
            let actualHash = computeHash content

            Assert.Equal<string>(expectedHash, actualHash)

    [<Theory>]
    [<InlineData("MiniQuery.tars", "query = \"find\"")>]
    [<InlineData("RFC3986_URI.tars", "URI = scheme")>]
    [<InlineData("Wolfram.tars", "program = statement_list")>]
    let ``Repository grammars contain expected productions`` (fileName: string, expectedFragment: string) =
        let path = Path.Combine(grammarDirectory, fileName)
        Assert.True(File.Exists(path), $"Missing grammar file {path}")

        let content =
            GrammarSource.External(FileInfo(path))
            |> GrammarSource.getContent

        Assert.Contains(expectedFragment, content)
