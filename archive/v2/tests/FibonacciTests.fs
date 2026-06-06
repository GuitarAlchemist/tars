module FibonacciTests

open Xunit
open Fibonacci

[<Theory>]
[<InlineData(0, 0)>]
[<InlineData(1, 1)>]
[<InlineData(2, 1)>]
[<InlineData(3, 2)>]
[<InlineData(4, 3)>]
[<InlineData(5, 5)>]
[<InlineData(6, 8)>]
[<InlineData(7, 13)>]
[<InlineData(8, 21)>]
[<InlineData(9, 34)>]
[<InlineData(10, 55)>]
[<InlineData(11, 89)>]
[<InlineData(12, 144)>]
[<InlineData(13, 233)>]
[<InlineData(14, 377)>]
[<InlineData(15, 610)>]
[<InlineData(16, 987)>]
[<InlineData(17, 1597)>]
[<InlineData(18, 2584)>]
[<InlineData(19, 4181)>]
[<InlineData(20, 6765)>]
let ``fib returns correct value`` (n: int, expected: int) =
    Assert.Equal(expected, fib n)

[<Theory>]
[<InlineData(0, 2)>]
[<InlineData(1, 1)>]
[<InlineData(2, 3)>]
[<InlineData(3, 4)>]
[<InlineData(4, 7)>]
[<InlineData(5, 11)>]
[<InlineData(6, 18)>]
[<InlineData(7, 29)>]
[<InlineData(8, 47)>]
[<InlineData(9, 76)>]
[<InlineData(10, 123)>]
[<InlineData(11, 199)>]
[<InlineData(12, 322)>]
[<InlineData(13, 521)>]
[<InlineData(14, 843)>]
[<InlineData(15, 1364)>]
[<InlineData(16, 2207)>]
[<InlineData(17, 3571)>]
[<InlineData(18, 5778)>]
[<InlineData(19, 9349)>]
[<InlineData(20, 15127)>]
let ``lucas returns correct value`` (n: int, expected: int) =
    Assert.Equal(expected, lucas n)
