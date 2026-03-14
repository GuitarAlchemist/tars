// Simplified streaming section for PuzzleDemo.fs
// Replace lines 196-253 with this:

let sw = System.Diagnostics.Stopwatch.StartNew()

AnsiConsole.MarkupLine("[bold cyan]🧠 TARS Thinking (streaming tokens)...[/]")
AnsiConsole.WriteLine()

// Stream tokens directly to console - NO PANEL, NO INTERFERENCE
let streamRequest = { request with Stream = true }

let! _response =
    llmService.CompleteStreamAsync(
        streamRequest,
        fun token ->
            tokenCountRef := !tokenCountRef + 1
            answerRef := !answerRef + token
            Console.Write(token) // RAW OUTPUT - YOU SEE EVERYTHING!
    )

sw.Stop()
AnsiConsole.WriteLine()
AnsiConsole.WriteLine()
let finalTokPerSec = float !tokenCountRef / sw.Elapsed.TotalSeconds

AnsiConsole.MarkupLine(
    $"[green]✅ Complete![/] [grey]({!tokenCountRef} tokens, {finalTokPerSec:F1} tok/s, {sw.Elapsed.TotalSeconds:F1}s)[/]"
)
