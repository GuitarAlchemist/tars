namespace Tars.Evolution

open System
open Tars.Core

/// Enhanced console visualization for Evolution Demo
module DemoVisualization =

    // ANSI color codes for rich console output
    let private cyan = "\u001b[36m"
    let private green = "\u001b[32m"
    let private yellow = "\u001b[33m"
    let private magenta = "\u001b[35m"
    let private blue = "\u001b[34m"
    let private red = "\u001b[31m"
    let private bold = "\u001b[1m"
    let private dim = "\u001b[2m"
    let private reset = "\u001b[0m"

    /// Get emoji and color for performative type
    let private getPerformativeStyle (perf: Performative) =
        match perf with
        | Performative.Request -> ("🎯", cyan)
        | Performative.Inform -> ("📢", green)
        | Performative.Propose -> ("💡", yellow)
        | Performative.Query -> ("❓", blue)
        | Performative.Refuse -> ("🚫", red)
        | Performative.Failure -> ("❌", red)
        | Performative.NotUnderstood -> ("❓", yellow)
        | Performative.Event -> ("📣", magenta)

    /// Format a message endpoint for display
    let private formatEndpoint (endpoint: MessageEndpoint) =
        match endpoint with
        | MessageEndpoint.System -> "System"
        | MessageEndpoint.Agent id -> $"Agent:%s{id.ToString().Substring(0, 8)}"
        | MessageEndpoint.User -> "User"
        | MessageEndpoint.Alias name -> $"%s{name}"

    /// Display a semantic message in the console
    let showSemanticMessage (msg: Message) (verbose: bool) =
        let (emoji, color) = getPerformativeStyle msg.Performative
        let sender = formatEndpoint msg.Sender

        let receiver =
            msg.Receiver |> Option.map formatEndpoint |> Option.defaultValue "Broadcast"

        let perfName = msg.Performative.ToString().ToUpper()

        // Compact format - show more content
        let preview =
            let firstLine = msg.Content.Split('\n').[0]

            if firstLine.Length > 100 then
                firstLine.Substring(0, 97) + "..."
            else
                firstLine

        printfn $"{color}{emoji} [{sender} → {receiver}] {bold}{perfName}{reset}{color}: {preview}{reset}"

        if verbose then
            printfn ""
            printfn $"{dim}╔════════════════════════════════════════════════════════════════════════════════╗{reset}"
            printfn $"{dim}║{reset} {bold}Semantic Message{reset}"
            printfn $"{dim}╟────────────────────────────────────────────────────────────────────────────────╢{reset}"
            printfn $"{dim}║{reset} {cyan}From:{reset}   {sender}"
            printfn $"{dim}║{reset} {cyan}To:{reset}     {receiver}"
            printfn $"{dim}║{reset} {cyan}Type:{reset}   {color}{msg.Performative}{reset}"

            match msg.Intent with
            | Some intent -> printfn $"{dim}║{reset} {cyan}Intent:{reset} {yellow}{intent}{reset}"
            | None -> ()

            printfn $"{dim}╟────────────────────────────────────────────────────────────────────────────────╢{reset}"

            // Wrap content at 76 chars to fit in box, show more lines
            let wrapWidth = 76

            let lines =
                msg.Content.Split([| '\n' |], StringSplitOptions.None)
                |> Array.collect (fun line ->
                    if line.Length <= wrapWidth then
                        [| line |]
                    else
                        // Word-wrap at spaces when possible
                        let rec wrapLine (remaining: string) (acc: string list) : string[] =
                            if remaining.Length <= wrapWidth then
                                List.rev (remaining :: acc) |> Array.ofList
                            else
                                let breakPoint =
                                    let spaceIdx =
                                        remaining.LastIndexOf(' ', Math.Min(wrapWidth, remaining.Length - 1))

                                    if spaceIdx > 20 then spaceIdx else wrapWidth

                                let (chunk, rest) =
                                    remaining.Substring(0, breakPoint), remaining.Substring(breakPoint).TrimStart()

                                wrapLine rest (chunk :: acc)

                        wrapLine line [])
                |> Array.truncate 25

            for line in lines do
                printfn $"{dim}║{reset} {line}"

            if msg.Content.Split('\n').Length > 25 || msg.Content.Length > 1500 then
                printfn $"{dim}║{reset} {dim}... (content truncated - {msg.Content.Length} chars total){reset}"

            printfn $"{dim}╚════════════════════════════════════════════════════════════════════════════════╝{reset}"
            printfn ""

    /// Show reflection progress
    let showReflection (iteration: int) (maxIterations: int) (feedback: string option) =
        let progressBar =
            String.replicate iteration "█"
            + String.replicate (maxIterations - iteration) "░"

        printfn $"{yellow}🔄 REFLECTION [{progressBar}] {iteration}/{maxIterations}{reset}"

        match feedback with
        | Some fb when fb.Length > 0 ->
            let preview = if fb.Length > 80 then fb.Substring(0, 77) + "..." else fb
            printfn $"   {dim}Feedback: {preview}{reset}"
        | _ -> ()

    /// Show improvement after reflection
    let showImprovement (description: string) =
        printfn $"{green}   ✨ Improvement: {description}{reset}"

    /// Show budget status
    let showBudgetStatus (tokensUsed: int) (tokensRemaining: int option) =
        let bar =
            match tokensRemaining with
            | Some remaining when remaining > 0 ->
                let total = tokensUsed + remaining
                let usedPercent = float tokensUsed / float total
                let usedBlocks = int (usedPercent * 20.0)
                let remainingBlocks = 20 - usedBlocks

                let barColor =
                    if usedPercent > 0.9 then red
                    elif usedPercent > 0.7 then yellow
                    else green

                let usedBar = String.replicate usedBlocks "█"
                let remainBar = String.replicate remainingBlocks "░"
                $"%s{barColor}%s{usedBar}%s{dim}%s{remainBar}%s{reset}"
            | _ -> $"%s{dim}[unlimited]%s{reset}"

        printfn $"%s{blue}💰 Budget: %s{bar} %d{tokensUsed} tokens used%s{reset}"

    /// Show task being worked on
    let showTaskStart (goal: string) (constraints: string list) =
        printfn ""
        printfn $"{bold}{magenta}╔══════════════════════════════════════════════════════════════╗{reset}"
        printfn $"{bold}{magenta}║{reset} {bold}📋 NEW TASK{reset}"
        printfn $"{magenta}╟──────────────────────────────────────────────────────────────╢{reset}"

        let goalPreview =
            if goal.Length > 60 then
                goal.Substring(0, 57) + "..."
            else
                goal

        printfn $"{magenta}║{reset} Goal: {goalPreview}"

        if not constraints.IsEmpty then
            printfn $"{magenta}║{reset} Constraints:"

            for c in constraints |> List.truncate 3 do
                let cPreview = if c.Length > 55 then c.Substring(0, 52) + "..." else c
                printfn $"{magenta}║{reset}   • {cPreview}"

        printfn $"{magenta}╚══════════════════════════════════════════════════════════════╝{reset}"
        printfn ""

    /// Show task completion with generated solution
    let showTaskComplete (success: bool) (output: string) (duration: TimeSpan) =
        let (icon, statusColor) = if success then ("✅", green) else ("❌", red)
        let status = if success then "SUCCESS" else "FAILED"

        printfn ""

        printfn
            $"{bold}{statusColor}╔════════════════════════════════════════════════════════════════════════════════╗{reset}"

        printfn
            $"{bold}{statusColor}║{reset} {icon} {bold}Task {status}{reset} {dim}(completed in {duration.TotalSeconds:F1}s){reset}"

        printfn
            $"{statusColor}╟────────────────────────────────────────────────────────────────────────────────╢{reset}"

        if String.IsNullOrWhiteSpace(output) then
            printfn $"{statusColor}║{reset} {dim}(no output){reset}"
        else
            // Display the solution/output
            printfn $"{statusColor}║{reset} {cyan}Generated Solution:{reset}"

            printfn
                $"{statusColor}╟────────────────────────────────────────────────────────────────────────────────╢{reset}"

            // Show up to 30 lines of output
            let lines = output.Split([| '\n' |], StringSplitOptions.None) |> Array.truncate 30

            for line in lines do
                let displayLine =
                    if line.Length > 78 then
                        line.Substring(0, 75) + "..."
                    else
                        line

                printfn $"{statusColor}║{reset} {displayLine}"

            if output.Split('\n').Length > 30 then
                printfn $"{statusColor}║{reset} {dim}... ({output.Split('\n').Length - 30} more lines){reset}"

        printfn
            $"{statusColor}╚════════════════════════════════════════════════════════════════════════════════╝{reset}"

        printfn ""

    /// Show evolution summary header
    let showSummaryHeader () =
        printfn ""
        printfn $"{bold}═══════════════════════════════════════════════════════════════{reset}"
        printfn $"{bold}                    EVOLUTION SUMMARY                          {reset}"
        printfn $"{bold}═══════════════════════════════════════════════════════════════{reset}"

    /// Show final stats
    let showFinalStats (generation: int) (tasksCompleted: int) (tokensUsed: int) (beliefs: int) =
        printfn $"  📊 Generation:      {cyan}{generation}{reset}"
        printfn $"  ✅ Tasks Completed: {green}{tasksCompleted}{reset}"
        printfn $"  💰 Tokens Used:     {yellow}{tokensUsed}{reset}"
        printfn $"  🧠 Beliefs Formed:  {magenta}{beliefs}{reset}"
        printfn $"{bold}═══════════════════════════════════════════════════════════════{reset}"
        printfn ""
