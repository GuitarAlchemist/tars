namespace Tars.Tests

open System
open System.IO
open Xunit
open Tars.Tools.Standard

/// The line counts `analyze_file_complexity` reports.
///
/// Checked through the tool rather than the private counter, because the wrong
/// numbers were wrong in the report a caller reads, and that is the thing worth
/// holding still.
module CodeAnalysisLineCountTests =

    /// Write `content` to a temporary .fs file, run the tool on it, delete the file.
    let private report (content: string) =
        if not (TestHelpers.requireTools ()) then
            None
        else
            let path = Path.Combine(Path.GetTempPath(), $"loc_{Guid.NewGuid():N}.fs")

            try
                File.WriteAllText(path, content)

                let answer =
                    CodeAnalysisTools.analyzeFileComplexity (sprintf """{"path":%s}""" (System.Text.Json.JsonSerializer.Serialize path))
                    |> Async.AwaitTask
                    |> Async.RunSynchronously

                Some answer
            finally
                try
                    File.Delete path
                with _ ->
                    ()

    /// The number in a "| Metric | N |" row.
    let private metric (name: string) (answer: string) =
        let marker = $"| {name} | "

        match answer.IndexOf(marker, StringComparison.Ordinal) with
        | -1 -> failwith $"the report has no '{name}' row:\n{answer}"
        | at ->
            let rest = answer.Substring(at + marker.Length)
            let value = rest.Substring(0, rest.IndexOf(' '))
            int value

    [<Fact>]
    let ``blank lines are counted, not deleted before counting`` () =
        // Six lines, two of them empty and one whitespace-only.
        //
        // The old split used RemoveEmptyEntries, which deletes every empty line before
        // anything is measured: "Total Lines" became the count of non-empty lines, and
        // "Blank Lines" could only ever match a line of spaces, because a genuinely
        // empty line was already gone. This file reported 3 total and 1 blank.
        let content = "let a = 1\n\nlet b = 2\n   \n\nlet c = 3\n"

        match report content with
        | None -> () // WDAC blocked Tars.Tools
        | Some answer ->
            Assert.Equal(6, metric "Total Lines" answer)
            Assert.Equal(3, metric "Blank Lines" answer)

    [<Fact>]
    let ``total lines is total lines, whatever is on them`` () =
        match report "a\nb\nc" with
        | None -> ()
        | Some answer ->
            Assert.Equal(3, metric "Total Lines" answer)
            Assert.Equal(0, metric "Blank Lines" answer)

    [<Fact>]
    let ``the newline a file ends with does not invent a line`` () =
        match report "a\nb\n" with
        | None -> ()
        | Some answer -> Assert.Equal(2, metric "Total Lines" answer)

    [<Fact>]
    let ``CRLF and LF count the same`` () =
        match report "let a = 1\r\n\r\nlet b = 2\r\n", report "let a = 1\n\nlet b = 2\n" with
        | Some crlf, Some lf ->
            Assert.Equal(metric "Total Lines" lf, metric "Total Lines" crlf)
            Assert.Equal(metric "Blank Lines" lf, metric "Blank Lines" crlf)
            Assert.Equal(3, metric "Total Lines" crlf)
        | _ -> ()

    [<Fact>]
    let ``the four counts still add up`` () =
        // Lines of Code is what is left after blanks and comments, so the three plus
        // it must be the total. That held before this change and has to keep holding:
        // it is the only relationship between the numbers a reader can rely on.
        let content = "// a comment\nlet a = 1\n\nlet b = 2\n// another\n   \nlet c = 3\n"

        match report content with
        | None -> ()
        | Some answer ->
            let total = metric "Total Lines" answer
            let blank = metric "Blank Lines" answer
            let comment = metric "Comment Lines" answer
            let loc = metric "Lines of Code" answer

            Assert.Equal(7, total)
            Assert.Equal(total, blank + comment + loc)
