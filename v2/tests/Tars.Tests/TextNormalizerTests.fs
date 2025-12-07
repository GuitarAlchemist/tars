module Tars.Tests.TextNormalizerTests

open Xunit
open Tars.Core

[<Fact>]
let ``normalize removes special chars and lowercases`` () =
    let input = "Hello, World! @ 2025."
    let expected = "hello world 2025"
    let actual = TextNormalizer.normalize input
    Assert.Equal(expected, actual)

[<Fact>]
let ``tokenize splits words correctly`` () =
    let input = "one two three"
    let expected = [ "one"; "two"; "three" ]
    let actual = TextNormalizer.tokenize input
    Assert.Equal<string list>(expected, actual)

[<Fact>]
let ``removeStopWords filters common words`` () =
    let input = [ "the"; "quick"; "brown"; "fox"; "is"; "a"; "dog" ]
    let expected = [ "quick"; "brown"; "fox"; "dog" ]
    let actual = TextNormalizer.removeStopWords input
    Assert.Equal<string list>(expected, actual)

[<Fact>]
let ``extractKeywords performs full pipeline`` () =
    // "The Quick, Brown Fox -> quick brown fox" + deduplication
    let input = "The Quick, Brown Fox jumps over the quick dog."
    let expected = [ "quick"; "brown"; "fox"; "jumps"; "over"; "dog" ]
    let actual = TextNormalizer.extractKeywords input
    Assert.Equal<string list>(expected, actual)
