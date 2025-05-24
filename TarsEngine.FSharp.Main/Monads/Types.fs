namespace TarsEngine.FSharp.Main.Monads

/// <summary>
/// Monadic programming types for the TARS Engine.
/// </summary>
module Types =
    /// <summary>
    /// Identity monad.
    /// </summary>
    type Identity<'T> = Identity of 'T
    
    /// <summary>
    /// Reader monad.
    /// </summary>
    type Reader<'Env, 'T> = Reader of ('Env -> 'T)
    
    /// <summary>
    /// Writer monad.
    /// </summary>
    type Writer<'W, 'T> = Writer of 'T * 'W
    
    /// <summary>
    /// State monad.
    /// </summary>
    type State<'S, 'T> = State of ('S -> 'T * 'S)
