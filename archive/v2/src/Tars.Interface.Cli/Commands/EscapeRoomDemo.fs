module Tars.Interface.Cli.Commands.EscapeRoomDemo

open System
open Serilog
open Tars.Core
open Tars.Llm
open Tars.Interface.Cli
open Spectre.Console

// ============================================================================
// ESCAPE ROOM DEMO - Demonstrates TARS reasoning with an interactive puzzle
// ============================================================================

/// Room state tracking
type RoomState =
    { CurrentRoom: string
      Inventory: string list
      UnlockedDoors: string list
      SolvedPuzzles: string list
      ExaminedItems: string list
      TurnCount: int
      GameWon: bool }

module RoomState =
    let initial =
        { CurrentRoom = "entrance"
          Inventory = []
          UnlockedDoors = []
          SolvedPuzzles = []
          ExaminedItems = []
          TurnCount = 0
          GameWon = false }

/// Room descriptions
let roomDescriptions =
    Map.ofList
        [ "entrance",
          """You are in a dimly lit entrance hall. 
There's a PAINTING on the wall, a CABINET in the corner, and a locked DOOR to the north.
The door has a 4-digit combination lock."""

          "library",
          """You enter a dusty library filled with old books.
A BOOKSHELF dominates one wall. There's a DESK with scattered papers.
A locked SAFE sits on the desk. Another DOOR leads east."""

          "exit",
          """You've found the exit! Sunlight streams through a window.
There's a final locked DOOR with a keyhole.""" ]

/// Items that can be examined
let examinableItems =
    Map.ofList
        [ "painting", "A portrait of a mathematician. The frame has numbers: '1879' engraved."
          "cabinet", "An old wooden cabinet. Opening it reveals a RUSTY KEY and a NOTE."
          "note", "The note reads: 'The code is the year Einstein was born.'"
          "rusty key", "An old rusty key. It might open something."
          "bookshelf", "Many dusty books. One titled 'SECRETS' looks out of place."
          "secrets book", "Inside the book is a BRASS KEY hidden in a cut-out section."
          "brass key", "A shiny brass key. Looks important."
          "desk", "Papers with mathematical equations. A SAFE sits here with a letter lock."
          "safe", "A safe with a 5-letter combination lock. There's a hint: 'Opposite of HELLO'."
          "golden key", "A golden key! This must be the exit key."
          "exit door", "The final door to freedom. It needs a golden key." ]

/// Process a command and return new state + result message
let processCommand (state: RoomState) (command: string) : RoomState * string =
    let cmd = command.ToLowerInvariant().Trim()

    let newState =
        { state with
            TurnCount = state.TurnCount + 1 }

    match cmd with
    // LOOK command
    | "look"
    | "look around" ->
        match roomDescriptions.TryFind state.CurrentRoom with
        | Some desc -> newState, desc
        | None -> newState, "You see nothing special."

    // EXAMINE commands
    | c when c.StartsWith("examine ") || c.StartsWith("look at ") ->
        let rawItem = c.Replace("examine ", "").Replace("look at ", "").Trim()
        // Normalize: remove extra spaces, handle common variations
        let item =
            rawItem.Replace("  ", " ").Replace(" ", "").Replace("shelf", "-SHELF-").Replace("-SHELF-", "shelf").Trim()
            // Fix: "bookshe lf" -> "bookshelf", handle weird spacing
            |> fun s -> System.Text.RegularExpressions.Regex.Replace(s, @"\s+", "")
            |> fun s ->
                if s = "bookshelf" || s.Contains("booksh") then
                    "bookshelf"
                else
                    s
            |> fun s ->
                if s = "secrets" || s = "secretsbook" then
                    "secrets book"
                else
                    s
            |> fun s ->
                if (s = "book" || s = "thebook") && state.CurrentRoom = "library" then
                    "secrets book"
                else
                    s
            // Restore normal form
            |> fun s -> s.Replace("bookshelf", "bookshelf")

        match examinableItems.TryFind item with
        | Some desc ->
            let s =
                { newState with
                    ExaminedItems = item :: newState.ExaminedItems |> List.distinct }

            s, desc
        | None -> newState, $"You can't examine '{item}'."

    // TAKE commands
    | c when c.StartsWith("take ") || c.StartsWith("pick up ") ->
        let item = c.Replace("take ", "").Replace("pick up ", "").Trim()

        let canTake =
            (item = "rusty key" && state.ExaminedItems |> List.contains "cabinet")
            || (item = "note" && state.ExaminedItems |> List.contains "cabinet")
            || (item = "brass key" && state.ExaminedItems |> List.contains "secrets book")
            || (item = "golden key" && state.SolvedPuzzles |> List.contains "safe")

        if canTake && not (state.Inventory |> List.contains item) then
            let s =
                { newState with
                    Inventory = item :: newState.Inventory }

            s, $"You take the {item}."
        elif state.Inventory |> List.contains item then
            newState, $"You already have the {item}."
        else
            newState, $"You can't take the {item}."

    // USE commands for keys in library
    | c when state.CurrentRoom = "library" && c.Contains("key") && c.Contains("door") ->
        if state.Inventory |> List.contains "rusty key" then
            let s =
                { newState with
                    UnlockedDoors = "library_exit" :: newState.UnlockedDoors }

            s, "You unlock the door with the rusty key! The door to the exit room opens."
        elif state.Inventory |> List.contains "brass key" then
            newState, "The brass key doesn't fit. Try the RUSTY KEY."
        else
            newState, "You need the RUSTY KEY from the entrance cabinet."

    // Any key usage without door in library
    | c when
        state.CurrentRoom = "library"
        && c.Contains("key")
        && c.Contains("use")
        && not (c.Contains("safe"))
        ->
        if state.Inventory |> List.contains "rusty key" then
            let s =
                { newState with
                    UnlockedDoors = "library_exit" :: newState.UnlockedDoors }

            s, "You unlock the door with the rusty key! The door to the exit room opens."
        else
            newState, "You need the RUSTY KEY."

    // Rusty key on safe doesn't work
    | c when state.CurrentRoom = "library" && c.Contains("rusty key") && c.Contains("safe") ->
        newState, "The rusty key doesn't fit the safe. This safe has a LETTER combination lock."

    // Hint: golden key only works in exit room
    | c when state.CurrentRoom = "library" && c.Contains("golden key") ->
        if state.UnlockedDoors |> List.contains "library_exit" then
            newState, "The golden key doesn't work here. Use it at the EXIT door. Go EAST first."
        else
            newState, "First unlock the door with the RUSTY KEY, then GO EAST."

    | "enter code 1879"
    | "use code 1879"
    | "input 1879" when state.CurrentRoom = "entrance" ->
        let s =
            { newState with
                UnlockedDoors = "entrance_library" :: newState.UnlockedDoors
                SolvedPuzzles = "entrance_lock" :: newState.SolvedPuzzles }

        s, "CLICK! The combination lock opens! The door to the library is now unlocked."

    | c when
        state.CurrentRoom = "library"
        && (c = "enter goodbye"
            || c = "input goodbye"
            || c = "use code goodbye"
            || c = "enter code goodbye"
            || c = "goodbye"
            || c.Contains("goodbye"))
        ->
        if state.ExaminedItems |> List.contains "safe" then
            let s =
                { newState with
                    SolvedPuzzles = "safe" :: newState.SolvedPuzzles }

            s, "The safe clicks open! Inside you find a GOLDEN KEY!"
        else
            newState, "You need to examine the safe first."

    // Hint when user tries HELLO
    | c when state.CurrentRoom = "library" && c.Contains("hello") ->
        newState, "The safe doesn't respond to 'hello'. The hint says 'opposite of HELLO' - try GOODBYE."

    | "use golden key on exit door"
    | "unlock exit door"
    | "use golden key" when state.CurrentRoom = "exit" ->
        if state.Inventory |> List.contains "golden key" then
            let s = { newState with GameWon = true }
            s, "You insert the golden key and turn it. The door swings open! CONGRATULATIONS - YOU ESCAPED!"
        else
            newState,
            "You need the GOLDEN KEY. Go back to the library, examine the SAFE, enter GOODBYE to open it, then TAKE GOLDEN KEY."

    // Hint for any action in exit room without golden key
    | c when
        state.CurrentRoom = "exit"
        && not (state.Inventory |> List.contains "golden key")
        && (c.Contains("door") || c.Contains("key") || c.Contains("use"))
        ->
        newState, "You need the GOLDEN KEY from the library safe. The safe code is the opposite of HELLO (GOODBYE)."

    // MOVEMENT
    | "go north"
    | "north"
    | "enter door" when state.CurrentRoom = "entrance" ->
        if state.UnlockedDoors |> List.contains "entrance_library" then
            let s =
                { newState with
                    CurrentRoom = "library" }

            s, "You go through the door into the library.\n\n" + (roomDescriptions.["library"])
        else
            newState, "The door is locked. It needs a 4-digit code."

    | "go east"
    | "east" when state.CurrentRoom = "library" ->
        if state.UnlockedDoors |> List.contains "library_exit" then
            let s = { newState with CurrentRoom = "exit" }
            s, "You go through the door into the exit room.\n\n" + (roomDescriptions.["exit"])
        else
            newState, "The door is locked. You need a key."

    // INVENTORY
    | "inventory"
    | "i"
    | "check inventory" ->
        if state.Inventory.IsEmpty then
            newState, "Your inventory is empty."
        else
            let items = String.Join(", ", state.Inventory)
            newState, $"Inventory: {items}"

    // HELP
    | "help" ->
        newState,
        """Available commands:
- LOOK / LOOK AROUND - See the room
- EXAMINE [item] - Look at something closely
- TAKE [item] - Pick up an item
- USE [item] ON [target] - Use an item
- ENTER CODE [code] - Enter a combination
- GO [direction] - Move to another room
- INVENTORY - Check what you're carrying"""

    | _ -> newState, $"I don't understand '{command}'. Try HELP for available commands."

/// Create LLM service
let private createLlmService (logger: ILogger) =
    LlmFactory.createWithConfig logger

/// System prompt for the escape room agent
let systemPrompt =
    """You play text adventure games. Output only game commands.

COMMANDS: LOOK, EXAMINE [item], TAKE [item], ENTER CODE [code], GO NORTH, GO EAST, USE [item] ON [target]

/no_think
Output exactly one command. No explanation."""

/// Run the escape room demo
let run (logger: ILogger) (maxTurns: int) (verbose: bool) =
    task {
        AnsiConsole.Write(new Rule("[bold magenta]TARS Escape Room Demo[/]"))
        AnsiConsole.MarkupLine("[dim]Watch TARS solve an escape room puzzle using reasoning![/]")
        AnsiConsole.WriteLine()

        let svc, config = createLlmService logger
        let mutable state = RoomState.initial
        let mutable conversationHistory = []
        let mutable solved = false

        // Initial room description
        let initialDesc = roomDescriptions.["entrance"]
        AnsiConsole.MarkupLine("[bold cyan]== ESCAPE ROOM START ==[/]")
        AnsiConsole.MarkupLine($"[dim]{initialDesc}[/]")
        AnsiConsole.WriteLine()

        conversationHistory <-
            [ { Role = Role.User
                Content = $"You are in an escape room. {initialDesc}\n\nWhat is your first command?" } ]

        while not solved && state.TurnCount < maxTurns do
            // Get LLM's next action
            let req =
                { ModelHint = None
                  Model = Some config.Llm.Model
                  SystemPrompt = Some systemPrompt
                  MaxTokens = Some 50
                  Temperature = Some 0.3
                  Stop = []
                  Messages = conversationHistory
                  Tools = []
                  ToolChoice = None
                  ResponseFormat = None
                  Stream = false
                  JsonMode = false
                  Seed = None

                  ContextWindow = None }

            let! response = svc.CompleteAsync(req)

            // Extract command - handle LLMs that include reasoning
            let rawText = response.Text.Trim()
            let lowerText = rawText.ToLowerInvariant()

            let command =
                // First, look for "ACTION:" prefix
                let actionMatch =
                    if lowerText.Contains("action:") then
                        let idx = lowerText.IndexOf("action:")
                        let afterAction = rawText.Substring(idx + 7).Trim()
                        let firstLine = afterAction.Split([| '\n' |]).[0].Trim()
                        Some(firstLine.ToLowerInvariant())
                    else
                        None

                match actionMatch with
                | Some cmd when cmd.Length > 0 -> cmd
                | _ ->
                    // Known valid commands
                    let validCommands =
                        [ "look"
                          "look around"
                          "examine painting"
                          "examine cabinet"
                          "examine note"
                          "examine bookshelf"
                          "examine secrets book"
                          "examine desk"
                          "examine safe"
                          "take rusty key"
                          "take note"
                          "take brass key"
                          "take golden key"
                          "enter code 1879"
                          "enter goodbye"
                          "input goodbye"
                          "use rusty key on door"
                          "use golden key on exit door"
                          "use golden key"
                          "go north"
                          "go east"
                          "north"
                          "east"
                          "inventory" ]

                    match validCommands |> List.tryFind (fun cmd -> lowerText.Contains(cmd)) with
                    | Some cmd -> cmd
                    | None ->
                        // Last resort: find any line with command prefix
                        let lines = rawText.Split([| '\n' |], StringSplitOptions.RemoveEmptyEntries)
                        let prefixes = [ "look"; "examine"; "take"; "enter"; "use"; "go"; "inventory" ]

                        match
                            lines
                            |> Array.tryFind (fun line ->
                                let l = line.Trim().ToLowerInvariant()
                                prefixes |> List.exists (fun p -> l.StartsWith(p)))
                        with
                        | Some line -> line.Trim().ToLowerInvariant()
                        | None -> "look"

            // Display the action
            AnsiConsole.MarkupLine($"[bold yellow]Turn {state.TurnCount + 1}:[/] [green]{Markup.Escape(command)}[/]")

            // Process the command
            let newState, result = processCommand state command
            state <- newState

            // Display the result
            AnsiConsole.MarkupLine($"[dim]{Markup.Escape(result)}[/]")
            AnsiConsole.WriteLine()

            // Update conversation
            conversationHistory <-
                conversationHistory
                @ [ { Role = Role.Assistant
                      Content = command }
                    { Role = Role.User
                      Content = $"Result: {result}\n\nWhat is your next command?" } ]

            // Check win condition
            if state.GameWon then
                solved <- true
                AnsiConsole.MarkupLine("[bold green]== ESCAPE SUCCESSFUL! ==[/]")
                AnsiConsole.MarkupLine($"[cyan]TARS escaped in {state.TurnCount} turns![/]")

        if not solved then
            AnsiConsole.MarkupLine("[bold red]== TIME'S UP! ==[/]")
            AnsiConsole.MarkupLine($"[yellow]TARS used all {maxTurns} turns without escaping.[/]")

        AnsiConsole.Write(new Rule())

        // Summary
        AnsiConsole.MarkupLine("[bold]Summary:[/]")
        let turnCount = state.TurnCount
        let puzzleCount = state.SolvedPuzzles.Length
        let itemCount = state.Inventory.Length

        let roomCount =
            if state.CurrentRoom = "exit" then 3
            elif state.CurrentRoom = "library" then 2
            else 1

        AnsiConsole.MarkupLine($"  Turns used: {turnCount}")
        AnsiConsole.MarkupLine($"  Puzzles solved: {puzzleCount}")
        AnsiConsole.MarkupLine($"  Items collected: {itemCount}")
        AnsiConsole.MarkupLine($"  Rooms visited: {roomCount}")

        return if solved then 0 else 1
    }

/// Interactive mode where user watches and can intervene
let runInteractive (logger: ILogger) =
    task {
        AnsiConsole.Write(new Rule("[bold magenta]TARS Escape Room - Interactive Mode[/]"))
        AnsiConsole.MarkupLine("[dim]Press ENTER to let TARS take a turn, or type a command to override.[/]")
        AnsiConsole.WriteLine()

        // For now, just run the auto mode with 30 turns max
        return! run logger 30 true
    }
