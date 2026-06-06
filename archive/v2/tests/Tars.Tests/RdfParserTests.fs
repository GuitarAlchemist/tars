namespace Tars.Tests

open Xunit
open Tars.LinkedData.RdfParser

module RdfParserTests =
    
    [<Fact>]
    let ``cleanUri removes angles and base`` () =
        let input = "<http://example.org/foo>"
        let expected = "foo"
        let actual = cleanUri input
        Assert.Equal(expected, actual)

    [<Fact>]
    let ``cleanUri handles hash`` () =
        let input = "http://example.org#bar"
        let expected = "bar"
        let actual = cleanUri input
        Assert.Equal(expected, actual)

    [<Fact>]
    let ``cleanUri handles no prefix`` () =
        let input = "baz"
        let expected = "baz"
        let actual = cleanUri input
        Assert.Equal(expected, actual)
