module Tars.Tests.FunctionalPatternsTests

open Xunit
open Tars.Core

// ============================================================================
// VALIDATION APPLICATIVE TESTS
// ============================================================================

[<Fact>]
let ``Validation.valid creates Valid`` () =
    let result = Validation.valid 42
    match result with
    | Valid v -> Assert.Equal(42, v)
    | Invalid _ -> Assert.Fail("Expected Valid")

[<Fact>]
let ``Validation.invalid creates Invalid with single error`` () =
    let result = Validation.invalid "error"
    match result with
    | Valid _ -> Assert.Fail("Expected Invalid")
    | Invalid es -> Assert.Equal(1, es.Length)

[<Fact>]
let ``Validation.apply accumulates multiple errors`` () =
    let vf = Validation.invalid "error1"
    let vx = Validation.invalid "error2"
    let result = Validation.apply vf vx
    match result with
    | Valid _ -> Assert.Fail("Expected Invalid")
    | Invalid es ->
        Assert.Equal(2, es.Length)
        Assert.Contains("error1", es)
        Assert.Contains("error2", es)

[<Fact>]
let ``Validation.ofResult converts Ok to Valid`` () =
    let result = Validation.ofResult (Ok 42)
    Assert.True(Validation.isValid result)

[<Fact>]
let ``Validation.toResult converts Valid to Ok`` () =
    let result = Validation.valid 42 |> Validation.toResult
    match result with
    | Ok v -> Assert.Equal(42, v)
    | Error _ -> Assert.Fail("Expected Ok")

// ============================================================================
// ASYNCRESULT MONAD TESTS
// ============================================================================

[<Fact>]
let ``AsyncResult.retn creates Ok`` () =
    async {
        let! result = AsyncResult.retn 42
        match result with
        | Ok v -> Assert.Equal(42, v)
        | Error _ -> Assert.Fail("Expected Ok")
    } |> Async.RunSynchronously

[<Fact>]
let ``AsyncResult.bind chains operations`` () =
    async {
        let! result =
            AsyncResult.retn 21
            |> AsyncResult.bind (fun x -> AsyncResult.retn (x * 2))
        match result with
        | Ok v -> Assert.Equal(42, v)
        | Error _ -> Assert.Fail("Expected Ok")
    } |> Async.RunSynchronously

[<Fact>]
let ``AsyncResult.map transforms value`` () =
    async {
        let! result =
            AsyncResult.retn 21
            |> AsyncResult.map (fun x -> x * 2)
        match result with
        | Ok v -> Assert.Equal(42, v)
        | Error _ -> Assert.Fail("Expected Ok")
    } |> Async.RunSynchronously

// ============================================================================
// READER MONAD TESTS
// ============================================================================

[<Fact>]
let ``Reader.ask returns environment`` () =
    let result = Reader.run Reader.ask 42
    Assert.Equal(42, result)

[<Fact>]
let ``Reader computation expression works`` () =
    let workflow = reader {
        let! env = Reader.ask
        return env * 2
    }
    let result = Reader.run workflow 21
    Assert.Equal(42, result)

[<Fact>]
let ``Reader.local modifies environment`` () =
    let workflow = Reader.local (fun x -> x + 1) Reader.ask
    let result = Reader.run workflow 41
    Assert.Equal(42, result)

// ============================================================================
// WRITER MONAD TESTS
// ============================================================================

[<Fact>]
let ``Writer.tell accumulates logs`` () =
    let workflow = writer {
        do! Writer.tell "log1"
        do! Writer.tell "log2"
        return 42
    }
    let (value, logs) = Writer.run workflow
    Assert.Equal(42, value)
    Assert.Equal(2, logs.Length)
    Assert.Contains("log1", logs)
    Assert.Contains("log2", logs)

[<Fact>]
let ``Writer.bind chains with logs`` () =
    let workflow = writer {
        do! Writer.tell "start"
        let result = 42
        do! Writer.tell "end"
        return result
    }
    let (value, logs) = Writer.run workflow
    Assert.Equal(42, value)
    Assert.Equal(["start"; "end"], logs)

// ============================================================================
// NONEMPTYLIST TESTS
// ============================================================================

[<Fact>]
let ``NonEmptyList.singleton creates list with one element`` () =
    let nel = NonEmptyList.singleton 42
    Assert.Equal(42, NonEmptyList.head nel)
    Assert.Equal(1, NonEmptyList.length nel)

[<Fact>]
let ``NonEmptyList.cons adds element`` () =
    let nel = NonEmptyList.singleton 2 |> NonEmptyList.cons 1
    Assert.Equal(1, NonEmptyList.head nel)
    Assert.Equal(2, NonEmptyList.length nel)

[<Fact>]
let ``NonEmptyList.ofList rejects empty list`` () =
    let result = NonEmptyList.ofList []
    Assert.True(Option.isNone result)

[<Fact>]
let ``NonEmptyList.ofList accepts non-empty list`` () =
    let result = NonEmptyList.ofList [1; 2; 3]
    Assert.True(Option.isSome result)
    match result with
    | Some nel -> Assert.Equal(3, NonEmptyList.length nel)
    | None -> Assert.Fail("Expected Some")

[<Fact>]
let ``NonEmptyList.map transforms elements`` () =
    let nel = NonEmptyList.singleton 21 |> NonEmptyList.map (fun x -> x * 2)
    Assert.Equal(42, NonEmptyList.head nel)

// ============================================================================
// VALIDATORS TESTS
// ============================================================================

[<Fact>]
let ``Validators.notEmpty rejects empty string`` () =
    let result = Validators.notEmpty "error" ""
    Assert.True(Validation.isInvalid result)

[<Fact>]
let ``Validators.notEmpty accepts non-empty string`` () =
    let result = Validators.notEmpty "error" "hello"
    Assert.True(Validation.isValid result)

[<Fact>]
let ``Validators.satisfies validates predicate`` () =
    let result = Validators.satisfies (fun x -> x > 0) "error" 42
    Assert.True(Validation.isValid result)

[<Fact>]
let ``Validators.inRange validates within range`` () =
    let result = Validators.inRange 0 100 "error" 50
    Assert.True(Validation.isValid result)

[<Fact>]
let ``Validators.inRange rejects outside range`` () =
    let result = Validators.inRange 0 100 "error" 150
    Assert.True(Validation.isInvalid result)

// ============================================================================
// OPERATORS TESTS
// ============================================================================

[<Fact>]
let ``Option map operator works`` () =
    let result = Some 21 |> Option.map (fun x -> x * 2)
    Assert.Equal(Some 42, result)

[<Fact>]
let ``Option bind operator works`` () =
    let result = Some 21 |> Option.bind (fun x -> Some (x * 2))
    Assert.Equal(Some 42, result)

[<Fact>]
let ``Result map operator works`` () =
    let result = Ok 21 |> Result.map (fun x -> x * 2)
    Assert.Equal(Ok 42, result)

[<Fact>]
let ``Result bind operator works`` () =
    let result = Ok 21 |> Result.bind (fun x -> Ok (x * 2))
    Assert.Equal(Ok 42, result)

[<Fact>]
let ``Kleisli composition works`` () =
    let parse x = try Ok (int x) with _ -> Error "Parse error"
    let validate x = if x > 0 then Ok x else Error "Must be positive"
    let parseAndValidate x = parse x |> Result.bind validate
    let result = parseAndValidate "42"
    Assert.Equal(Ok 42, result)

// ============================================================================
// INTEGRATION TESTS
// ============================================================================

type ConfigError =
    | NameEmpty
    | PortInvalid
    | TimeoutInvalid

[<Fact>]
let ``Real-world config validation accumulates all errors`` () =
    let validateName name = Validators.notEmpty NameEmpty name
    let validatePort port = Validators.inRange 1 65535 PortInvalid port
    let validateTimeout timeout = Validators.satisfies (fun t -> t >= 0) TimeoutInvalid timeout
    
    // All invalid
    let errors =
        [validateName ""; validatePort 99999; validateTimeout -1]
        |> List.choose (function Invalid es -> Some es | _ -> None)
        |> List.concat
    
    Assert.Equal(3, errors.Length)
    Assert.Contains(NameEmpty, errors)
    Assert.Contains(PortInvalid, errors)
    Assert.Contains(TimeoutInvalid, errors)

[<Fact>]
let ``Real-world AsyncResult pipeline`` () =
    async {
        let loadData () : AsyncResult<int, string> = async { return Ok 42 }
        let processData (x: int) : AsyncResult<int, string> = async { return Ok (x * 2) }
        
        let! result =
            loadData()
            |> AsyncResult.bind processData
        
        match result with
        | Ok v -> Assert.Equal(84, v)
        | Error _ -> Assert.Fail("Expected Ok")
    } |> Async.RunSynchronously

[<Fact>]
let ``Real-world Reader dependency injection`` () =
    let workflow = reader {
        let! config = Reader.ask
        return 21 * config.Multiplier
    }
    let result = Reader.run workflow { Multiplier = 2 }
    Assert.Equal(42, result)

[<Fact>]
let ``Real-world Writer logging`` () =
    let workflow = writer {
        do! Writer.tell "Processing started"
        let result = 42
        do! Writer.tell $"Result: {result}"
        return result
    }
    let (value, logs) = Writer.run workflow
    Assert.Equal(42, value)
    Assert.Equal(2, logs.Length)
