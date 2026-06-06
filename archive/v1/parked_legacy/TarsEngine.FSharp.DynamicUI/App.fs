namespace TarsEngine.FSharp.DynamicUI

open Elmish
open Elmish.React
open State
open Update
open View

module App =
    
    /// Main Elmish program
    let program =
        Program.mkProgram init update view
        |> Program.withSubscription subscription
        |> Program.withReactSynchronous "tars-dynamic-ui-root"
    
    /// Start the application
    let run () =
        program |> Program.run
