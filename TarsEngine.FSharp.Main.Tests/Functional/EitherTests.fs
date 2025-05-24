namespace TarsEngine.FSharp.Main.Tests.Functional

open System
open Xunit
open TarsEngine.FSharp.Main.Functional

/// <summary>
/// Tests for the Either type
/// </summary>
module EitherTests =
    [<Fact>]
    let ``Either.match' should apply the left function for Left`` () =
        let either = Either<string, int>.Left "error"
        let result = Either.match' (fun s -> s.Length) (fun i -> i * 2) either
        Assert.Equal(5, result)
    
    [<Fact>]
    let ``Either.match' should apply the right function for Right`` () =
        let either = Either<string, int>.Right 21
        let result = Either.match' (fun s -> s.Length) (fun i -> i * 2) either
        Assert.Equal(42, result)
    
    [<Fact>]
    let ``Either.map should transform the Right value`` () =
        let either = Either<string, int>.Right 21
        let result = Either.map (fun i -> i * 2) either
        match result with
        | Right value -> Assert.Equal(42, value)
        | Left _ -> Assert.True(false, "Expected Right, got Left")
    
    [<Fact>]
    let ``Either.map should not transform the Left value`` () =
        let either = Either<string, int>.Left "error"
        let result = Either.map (fun i -> i * 2) either
        match result with
        | Right _ -> Assert.True(false, "Expected Left, got Right")
        | Left value -> Assert.Equal("error", value)
    
    [<Fact>]
    let ``Either.mapLeft should transform the Left value`` () =
        let either = Either<string, int>.Left "error"
        let result = Either.mapLeft (fun s -> s.Length) either
        match result with
        | Right _ -> Assert.True(false, "Expected Left, got Right")
        | Left value -> Assert.Equal(5, value)
    
    [<Fact>]
    let ``Either.mapLeft should not transform the Right value`` () =
        let either = Either<string, int>.Right 42
        let result = Either.mapLeft (fun s -> s.Length) either
        match result with
        | Right value -> Assert.Equal(42, value)
        | Left _ -> Assert.True(false, "Expected Right, got Left")
    
    [<Fact>]
    let ``Either.bind should apply the function to the Right value`` () =
        let either = Either<string, int>.Right 21
        let result = Either.bind (fun i -> Either<string, int>.Right (i * 2)) either
        match result with
        | Right value -> Assert.Equal(42, value)
        | Left _ -> Assert.True(false, "Expected Right, got Left")
    
    [<Fact>]
    let ``Either.bind should not apply the function to the Left value`` () =
        let either = Either<string, int>.Left "error"
        let result = Either.bind (fun i -> Either<string, int>.Right (i * 2)) either
        match result with
        | Right _ -> Assert.True(false, "Expected Left, got Right")
        | Left value -> Assert.Equal("error", value)
    
    [<Fact>]
    let ``Either.isLeft should return true for Left`` () =
        let either = Either<string, int>.Left "error"
        Assert.True(Either.isLeft either)
    
    [<Fact>]
    let ``Either.isLeft should return false for Right`` () =
        let either = Either<string, int>.Right 42
        Assert.False(Either.isLeft either)
    
    [<Fact>]
    let ``Either.isRight should return true for Right`` () =
        let either = Either<string, int>.Right 42
        Assert.True(Either.isRight either)
    
    [<Fact>]
    let ``Either.isRight should return false for Left`` () =
        let either = Either<string, int>.Left "error"
        Assert.False(Either.isRight either)
    
    [<Fact>]
    let ``Either.leftValue should return the Left value`` () =
        let either = Either<string, int>.Left "error"
        Assert.Equal("error", Either.leftValue either)
    
    [<Fact>]
    let ``Either.leftValue should throw for Right`` () =
        let either = Either<string, int>.Right 42
        Assert.Throws<InvalidOperationException>(fun () -> Either.leftValue either |> ignore)
    
    [<Fact>]
    let ``Either.rightValue should return the Right value`` () =
        let either = Either<string, int>.Right 42
        Assert.Equal(42, Either.rightValue either)
    
    [<Fact>]
    let ``Either.rightValue should throw for Left`` () =
        let either = Either<string, int>.Left "error"
        Assert.Throws<InvalidOperationException>(fun () -> Either.rightValue either |> ignore)
    
    [<Fact>]
    let ``Either.leftValueOrDefault should return the Left value`` () =
        let either = Either<string, int>.Left "error"
        Assert.Equal("error", Either.leftValueOrDefault "default" either)
    
    [<Fact>]
    let ``Either.leftValueOrDefault should return the default value for Right`` () =
        let either = Either<string, int>.Right 42
        Assert.Equal("default", Either.leftValueOrDefault "default" either)
    
    [<Fact>]
    let ``Either.rightValueOrDefault should return the Right value`` () =
        let either = Either<string, int>.Right 42
        Assert.Equal(42, Either.rightValueOrDefault 0 either)
    
    [<Fact>]
    let ``Either.rightValueOrDefault should return the default value for Left`` () =
        let either = Either<string, int>.Left "error"
        Assert.Equal(0, Either.rightValueOrDefault 0 either)
    
    [<Fact>]
    let ``Either.tryFunc should return Right for successful function`` () =
        let result = Either.tryFunc (fun () -> 42)
        match result with
        | Right value -> Assert.Equal(42, value)
        | Left _ -> Assert.True(false, "Expected Right, got Left")
    
    [<Fact>]
    let ``Either.tryFunc should return Left for failing function`` () =
        let result = Either.tryFunc (fun () -> raise (InvalidOperationException("error")))
        match result with
        | Right _ -> Assert.True(false, "Expected Left, got Right")
        | Left ex -> Assert.IsType<InvalidOperationException>(ex)
    
    [<Fact>]
    let ``Either.ofOption should return Right for Some`` () =
        let option = Some 42
        let result = Either.ofOption "error" option
        match result with
        | Right value -> Assert.Equal(42, value)
        | Left _ -> Assert.True(false, "Expected Right, got Left")
    
    [<Fact>]
    let ``Either.ofOption should return Left for None`` () =
        let option = None
        let result = Either.ofOption "error" option
        match result with
        | Right _ -> Assert.True(false, "Expected Left, got Right")
        | Left value -> Assert.Equal("error", value)
    
    [<Fact>]
    let ``Either.toOption should return Some for Right`` () =
        let either = Either<string, int>.Right 42
        let result = Either.toOption either
        match result with
        | Some value -> Assert.Equal(42, value)
        | None -> Assert.True(false, "Expected Some, got None")
    
    [<Fact>]
    let ``Either.toOption should return None for Left`` () =
        let either = Either<string, int>.Left "error"
        let result = Either.toOption either
        match result with
        | Some _ -> Assert.True(false, "Expected None, got Some")
        | None -> Assert.True(true)
