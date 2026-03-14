namespace TarsEngine.FSharp.Main.Functional

/// <summary>
/// Functional programming types for the TARS Engine.
/// </summary>
module Types =
    /// <summary>
    /// Function composition.
    /// </summary>
    let compose f g x = f (g x)
    
    /// <summary>
    /// Function application.
    /// </summary>
    let apply f x = f x
    
    /// <summary>
    /// Function currying.
    /// </summary>
    let curry f a b = f (a, b)
    
    /// <summary>
    /// Function uncurrying.
    /// </summary>
    let uncurry f (a, b) = f a b
