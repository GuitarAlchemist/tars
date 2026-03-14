module SumEvenTests

open NUnit.Framework
open SumEven

[<TestFixture>]
type SumEvenTests() =
    [<Test>]
    member x."sumEvenNumbers should return 6 for [1;2;3;4]" () =
        let result = sumEvenNumbers [1;2;3;4]
        Assert.AreEqual(6, result)

    [<Test>]
    member x."sumEvenNumbers returns 0 for empty list" () =
        let result = sumEvenNumbers []
        Assert.AreEqual(0, result)

    [<Test>]
    member x."sumEvenNumbers handles all odd numbers" () =
        let result = sumEvenNumbers [1;3;5;7]
        Assert.AreEqual(0, result)