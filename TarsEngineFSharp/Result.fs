namespace TarsEngineFSharp

/// <summary>
/// Discriminated union for representing a result that can be either a success or a failure
/// </summary>
type Result<'TSuccess, 'TFailure> =
    /// <summary>
    /// Represents a successful result
    /// </summary>
    | Success of 'TSuccess
    /// <summary>
    /// Represents a failed result
    /// </summary>
    | Failure of 'TFailure

/// <summary>
/// Module containing helper functions for Result
/// </summary>
module Result =
    /// <summary>
    /// Create a success result
    /// </summary>
    /// <param name="value">The success value</param>
    /// <returns>A success result</returns>
    let success value = Success value

    /// <summary>
    /// Create a failure result
    /// </summary>
    /// <param name="error">The error value</param>
    /// <returns>A failure result</returns>
    let failure error = Failure error

    /// <summary>
    /// Apply a function to the success value of a result
    /// </summary>
    /// <param name="f">The function to apply</param>
    /// <param name="result">The result</param>
    /// <returns>A new result with the function applied to the success value</returns>
    let map f result =
        match result with
        | Success value -> Success (f value)
        | Failure error -> Failure error

    /// <summary>
    /// Apply a function to the failure value of a result
    /// </summary>
    /// <param name="f">The function to apply</param>
    /// <param name="result">The result</param>
    /// <returns>A new result with the function applied to the failure value</returns>
    let mapError f result =
        match result with
        | Success value -> Success value
        | Failure error -> Failure (f error)

    /// <summary>
    /// Apply a function that returns a result to the success value of a result
    /// </summary>
    /// <param name="f">The function to apply</param>
    /// <param name="result">The result</param>
    /// <returns>A new result with the function applied to the success value</returns>
    let bind f result =
        match result with
        | Success value -> f value
        | Failure error -> Failure error

    /// <summary>
    /// Apply a function to both the success and failure values of a result
    /// </summary>
    /// <param name="onSuccess">The function to apply to the success value</param>
    /// <param name="onFailure">The function to apply to the failure value</param>
    /// <param name="result">The result</param>
    /// <returns>The result of applying the appropriate function</returns>
    let either onSuccess onFailure result =
        match result with
        | Success value -> onSuccess value
        | Failure error -> onFailure error

    /// <summary>
    /// Get the success value or a default value if the result is a failure
    /// </summary>
    /// <param name="defaultValue">The default value</param>
    /// <param name="result">The result</param>
    /// <returns>The success value or the default value</returns>
    let defaultValue defaultValue result =
        match result with
        | Success value -> value
        | Failure _ -> defaultValue

    /// <summary>
    /// Get the success value or apply a function to the failure value
    /// </summary>
    /// <param name="defaultFn">The function to apply to the failure value</param>
    /// <param name="result">The result</param>
    /// <returns>The success value or the result of applying the function to the failure value</returns>
    let defaultWith defaultFn result =
        match result with
        | Success value -> value
        | Failure error -> defaultFn error

    /// <summary>
    /// Try to execute a function and return a result
    /// </summary>
    /// <param name="f">The function to execute</param>
    /// <returns>A success result with the function result or a failure result with the exception</returns>
    let tryWith f =
        try
            Success (f())
        with
        | ex -> Failure ex
