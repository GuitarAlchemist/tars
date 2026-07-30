namespace Tars.Tests

open System
open System.Collections.Generic
open System.IO
open System.Text
open System.Text.RegularExpressions
open Xunit
open Tars.DSL.Wot

/// Item 4 slice C — grammars/wot.ebnf must describe the language WotParser
/// actually accepts.
///
/// Before this, the two had no surface syntax in common at all: the grammar
/// specified `META { k: "v" }`, `REASON id { input: a }` and `EDGE a -> b when
/// x == "y"`, while the parser accepts `meta { k = "v" }`, `node "id"
/// kind="reason" { input = ["a"] }` and `edge "a" -> "b"`. Every string the old
/// grammar generated was unparseable. Nothing caught it because nothing loaded
/// the file — `withNamedGrammar dir "wot"` would have worked, but no call site
/// ever asked for it.
///
/// The gate here is SOUNDNESS, not equivalence: everything the grammar
/// generates must parse under WotParser. The reverse is deliberately untested —
/// WotParser ignores unrecognised lines and tolerates unquoted values, so
/// asserting the converse would mean encoding its tolerance for garbage.
module WotGrammarConformanceTests =

    // ────────────────────────────────────────────────────────────────────────
    // A recognizer for the EBNF subset grammars/wot.ebnf uses.
    //
    // Deliberately small, with two documented limitations that this grammar
    // does not exercise:
    //   * left recursion yields [] rather than looping (guarded, not supported)
    //   * a repetition with min >= 2 may under-report when the inner expression
    //     is ambiguous; only ? * + appear here, whose min is 0 or 1
    // Anything outside the subset raises instead of silently passing.
    // ────────────────────────────────────────────────────────────────────────
    module private Ebnf =

        type Expr =
            | Lit of string
            | Cls of ranges: (char * char) list * negated: bool
            | Ref of string
            | Cat of Expr list
            | Alt of Expr list
            | Rep of Expr * min: int * max: int option

        /// Strips (* ... *) comments. Non-nesting, and it would corrupt a
        /// literal containing "(*" — wot.ebnf has none, and a stray strip
        /// surfaces as a parse failure rather than a silent pass.
        let private stripComments (text: string) =
            let sb = StringBuilder()
            let mutable i = 0

            while i < text.Length do
                if i + 1 < text.Length && text[i] = '(' && text[i + 1] = '*' then
                    let mutable j = i + 2

                    while j + 1 < text.Length && not (text[j] = '*' && text[j + 1] = ')') do
                        j <- j + 1

                    i <- if j + 1 < text.Length then j + 2 else text.Length
                else
                    sb.Append(text[i]) |> ignore
                    i <- i + 1

            sb.ToString()

        let private parseRhs (name: string) (src: string) : Expr =
            let mutable i = 0
            let skipWs () = while i < src.Length && Char.IsWhiteSpace src[i] do i <- i + 1

            let decodeEscape (c: char) =
                match c with
                | 'n' -> '\n'
                | 't' -> '\t'
                | 'r' -> '\r'
                | other -> other

            let readLiteral (closing: char) =
                let sb = StringBuilder()

                while i < src.Length && src[i] <> closing do
                    if src[i] = '\\' && i + 1 < src.Length then
                        sb.Append(decodeEscape src[i + 1]) |> ignore
                        i <- i + 2
                    else
                        sb.Append(src[i]) |> ignore
                        i <- i + 1

                if i >= src.Length then
                    failwithf "%s: unterminated literal" name

                i <- i + 1
                sb.ToString()

            let readClass () =
                i <- i + 1 // consume '['
                let negated = i < src.Length && src[i] = '^'
                if negated then i <- i + 1
                let ranges = ResizeArray<char * char>()

                let readChar () =
                    if src[i] = '\\' && i + 1 < src.Length then
                        let c = decodeEscape src[i + 1]
                        i <- i + 2
                        c
                    else
                        let c = src[i]
                        i <- i + 1
                        c

                while i < src.Length && src[i] <> ']' do
                    let a = readChar ()

                    if i + 1 < src.Length && src[i] = '-' && src[i + 1] <> ']' then
                        i <- i + 1
                        ranges.Add(a, readChar ())
                    else
                        ranges.Add(a, a)

                if i >= src.Length then
                    failwithf "%s: unterminated character class" name

                i <- i + 1
                Cls(List.ofSeq ranges, negated)

            let rec parseAlt () =
                let first = parseCat ()
                let alts = ResizeArray<Expr>([ first ])
                skipWs ()

                while i < src.Length && src[i] = '|' do
                    i <- i + 1
                    alts.Add(parseCat ())
                    skipWs ()

                if alts.Count = 1 then first else Alt(List.ofSeq alts)

            and parseCat () =
                let parts = ResizeArray<Expr>()
                let mutable go = true

                while go do
                    skipWs ()

                    if i >= src.Length || src[i] = '|' || src[i] = ')' then
                        go <- false
                    else
                        parts.Add(parsePostfix ())

                if parts.Count = 1 then parts[0] else Cat(List.ofSeq parts)

            and parsePostfix () =
                let a = parseAtom ()

                if i < src.Length then
                    match src[i] with
                    | '?' ->
                        i <- i + 1
                        Rep(a, 0, Some 1)
                    | '*' ->
                        i <- i + 1
                        Rep(a, 0, None)
                    | '+' ->
                        i <- i + 1
                        Rep(a, 1, None)
                    | _ -> a
                else
                    a

            and parseAtom () =
                skipWs ()

                if i >= src.Length then
                    failwithf "%s: unexpected end of production" name

                match src[i] with
                | '(' ->
                    i <- i + 1
                    let e = parseAlt ()
                    skipWs ()

                    if i >= src.Length || src[i] <> ')' then
                        failwithf "%s: expected ')'" name

                    i <- i + 1
                    e
                | '"' ->
                    i <- i + 1
                    Lit(readLiteral '"')
                | '\'' ->
                    i <- i + 1
                    Lit(readLiteral '\'')
                | '[' -> readClass ()
                | c when Char.IsLetter c || c = '_' ->
                    let start = i

                    while i < src.Length && (Char.IsLetterOrDigit src[i] || src[i] = '_') do
                        i <- i + 1

                    Ref(src.Substring(start, i - start))
                | c -> failwithf "%s: unsupported character '%c' in production" name c

            let e = parseAlt ()
            skipWs ()

            if i < src.Length then
                failwithf "%s: trailing input at offset %d: %s" name i (src.Substring i)

            e

        /// name ::= rhs, with continuation lines folded into the previous rule.
        let parse (text: string) : Map<string, Expr> =
            let rxHead = Regex(@"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*::=(.*)$")

            let lines = (stripComments text).Replace("\r\n", "\n").Split('\n')
            let order = ResizeArray<string>()
            let bodies = Dictionary<string, StringBuilder>()

            for line in lines do
                let m = rxHead.Match line

                if m.Success then
                    let n = m.Groups[1].Value

                    if bodies.ContainsKey n then
                        failwithf "duplicate rule '%s'" n

                    order.Add n
                    bodies[n] <- StringBuilder(m.Groups[2].Value)
                elif order.Count > 0 && not (String.IsNullOrWhiteSpace line) then
                    bodies[order[order.Count - 1]].Append(' ').Append(line) |> ignore

            order
            |> Seq.map (fun n -> n, parseRhs n (bodies[n].ToString()))
            |> Map.ofSeq

        /// Rules referenced but never defined — a typo here would otherwise
        /// only show up as a mysterious rejection.
        let undefinedRefs (g: Map<string, Expr>) : string list =
            let rec refs e =
                match e with
                | Ref n -> [ n ]
                | Cat xs
                | Alt xs -> xs |> List.collect refs
                | Rep(x, _, _) -> refs x
                | Lit _
                | Cls _ -> []

            g
            |> Map.toList
            |> List.collect (snd >> refs)
            |> List.distinct
            |> List.filter (fun n -> not (g.ContainsKey n))
            |> List.sort

        /// Returns whether `start` derives the whole of `text`, plus the
        /// furthest offset any terminal matched. On rejection that offset is
        /// where the derivation ran out of options — the practical way to find
        /// which line a grammar/parser divergence sits on.
        let acceptsWithFurthest (g: Map<string, Expr>) (start: string) (text: string) : bool * int =
            let memo = Dictionary<struct (string * int), int list>()
            let mutable steps = 0
            let mutable furthest = 0

            let rec ends (e: Expr) (pos: int) : int list =
                steps <- steps + 1

                if steps > 5_000_000 then
                    failwith "EBNF recognizer step budget exhausted — the grammar is pathologically ambiguous"

                match e with
                | Lit t ->
                    if pos + t.Length <= text.Length && String.CompareOrdinal(text, pos, t, 0, t.Length) = 0 then
                        if pos + t.Length > furthest then furthest <- pos + t.Length
                        [ pos + t.Length ]
                    else
                        []
                | Cls(ranges, negated) ->
                    if pos >= text.Length then
                        []
                    else
                        let c = text[pos]
                        let inside = ranges |> List.exists (fun (a, b) -> c >= a && c <= b)

                        if inside <> negated then
                            if pos + 1 > furthest then furthest <- pos + 1
                            [ pos + 1 ]
                        else
                            []
                | Ref n ->
                    let key = struct (n, pos)

                    match memo.TryGetValue key with
                    | true, cached -> cached
                    | _ ->
                        memo[key] <- [] // left-recursion guard

                        let body =
                            match g.TryFind n with
                            | Some b -> b
                            | None -> failwithf "undefined rule '%s'" n

                        let r = ends body pos
                        memo[key] <- r
                        r
                | Cat parts ->
                    parts
                    |> List.fold (fun acc p -> acc |> List.collect (ends p) |> List.distinct) [ pos ]
                | Alt alts -> alts |> List.collect (fun a -> ends a pos) |> List.distinct
                | Rep(inner, mn, mx) ->
                    let acc = HashSet<int>()
                    if mn = 0 then acc.Add pos |> ignore
                    let seen = HashSet<int>([ pos ])
                    let mutable frontier = [ pos ]
                    let mutable k = 0
                    let mutable go = true

                    while go && not frontier.IsEmpty do
                        k <- k + 1

                        // `seen` filters both zero-width matches and revisits,
                        // which is what guarantees termination.
                        let next =
                            frontier |> List.collect (ends inner) |> List.distinct |> List.filter seen.Add

                        if k >= mn then
                            for p in next do
                                acc.Add p |> ignore

                        frontier <- next

                        match mx with
                        | Some m when k >= m -> go <- false
                        | _ -> ()

                    acc |> List.ofSeq |> List.sortDescending

            let ok = ends (Ref start) 0 |> List.contains text.Length
            ok, furthest

        /// True when `start` derives the whole of `text`.
        let accepts (g: Map<string, Expr>) (start: string) (text: string) : bool =
            acceptsWithFurthest g start text |> fst

    // ── loading ─────────────────────────────────────────────────────────────

    let private baseDir = AppContext.BaseDirectory

    let private grammarPath = Path.Combine(baseDir, "grammars", "wot.ebnf")

    let private fixturePath (name: string) =
        Path.Combine(baseDir, "WorkflowOfThought", "fixtures", name)

    /// Both adjustments mirror File.ReadAllLines, which is how WotParser reads
    /// input, so the grammar models post-ReadAllLines text:
    ///   * carriage returns are stripped before any rule sees them
    ///   * a final line with no trailing newline is still a line
    /// The second matters: 3 of the 25 corpus plans end with `}` and no
    /// newline. Weakening `eol` to `nl?` in the grammar would have admitted
    /// them too, but at the cost of making every newline optional — which is
    /// exactly the line orientation this grammar exists to pin down.
    let private normalize (s: string) =
        let lf = s.Replace("\r\n", "\n").Replace("\r", "\n")
        if lf.EndsWith "\n" || lf = "" then lf else lf + "\n"

    let private grammar =
        lazy
            (Assert.True(File.Exists grammarPath, $"grammar not copied to output: {grammarPath}")
             Ebnf.parse (File.ReadAllText grammarPath))

    let private grammarAccepts (text: string) =
        Ebnf.accepts grammar.Value "file" (normalize text)

    /// "<line>: <text>" for the furthest point the grammar reached — where the
    /// derivation ran out of options. Without this a rejection is just `false`,
    /// and finding the offending construct in a 70-line workflow is guesswork.
    let private rejectionSite (text: string) =
        let t = normalize text
        let _, furthest = Ebnf.acceptsWithFurthest grammar.Value "file" t
        let consumed = t.Substring(0, min furthest t.Length)
        let line = 1 + (consumed |> Seq.filter ((=) '\n') |> Seq.length)
        let lines = t.Split('\n')
        let offending = if line - 1 < lines.Length then lines[line - 1].Trim() else "<end of input>"
        $"line {line}: {offending}"

    let private parserAccepts (text: string) =
        let lines = (normalize text).Split('\n') |> Array.toList
        match WotParser.parseLines lines with
        | Ok _ -> true
        | Error _ -> false

    // ── the grammar itself is well-formed ───────────────────────────────────

    [<Fact>]
    let ``wot.ebnf parses and defines every rule it references`` () =
        let g = grammar.Value
        Assert.NotEmpty(g)
        let missing = Ebnf.undefinedRefs g

        Assert.True(
            List.isEmpty missing,
            $"""wot.ebnf references undefined rules: {String.Join(", ", missing)}"""
        )

        Assert.True(g.ContainsKey "file", "wot.ebnf must define the `file` start rule")

    // ── soundness: what the grammar admits, the parser accepts ──────────────

    [<Theory>]
    [<InlineData("sample.wot.trsx")>]
    [<InlineData("full_surface.wot.trsx")>]
    let ``fixture is accepted by both the grammar and WotParser`` (fixture: string) =
        let path = fixturePath fixture
        Assert.True(File.Exists path, $"missing fixture: {path}")
        let text = File.ReadAllText path

        // Order matters for the failure message: a grammar that is too narrow
        // is the likely defect, so report it first and distinctly.
        Assert.True(
            grammarAccepts text,
            $"grammars/wot.ebnf rejects {fixture}, which WotParser accepts — {rejectionSite text}"
        )

        Assert.True(parserAccepts text, $"WotParser rejects {fixture}")

    /// The construct-by-construct soundness check. Each snippet is a complete
    /// workflow the grammar admits; WotParser must accept every one.
    [<Theory>]
    [<InlineData("goal", "    goal = \"do the thing\"")>]
    [<InlineData("output", "    output = \"result\"")>]
    [<InlineData("input list", "    input = [\"a\", \"b\"]")>]
    [<InlineData("input scalar", "    input = \"a\"")>]
    [<InlineData("invariants", "    invariants = [\"keep_it\"]")>]
    [<InlineData("constraints", "    constraints = [\"be_brief\"]")>]
    [<InlineData("tool", "    tool = \"web_search\"")>]
    [<InlineData("verdict", "    verdict = \"accept\"")>]
    [<InlineData("condition", "    condition = \"ready\"")>]
    [<InlineData("kind in body", "    kind = \"work\"")>]
    [<InlineData("args", "    args = { \"query\": \"x\" }")>]
    [<InlineData("comment", "    // just a comment")>]
    let ``node field the grammar admits is accepted by WotParser`` (label: string) (field: string) =
        let text =
            String.Join(
                "\n",
                [ "meta {"; "  name = \"t\""; "}"; "workflow \"t\" {"; "  node \"n\" {"; field; "  }"; "}"; "" ]
            )

        Assert.True(grammarAccepts text, $"grammar rejects its own construct: {label}")
        Assert.True(parserAccepts text, $"WotParser rejects grammar-legal construct: {label}")

    // ── non-vacuity: the grammar must reject what the parser rejects ─────────

    /// The exact syntax the old wot.ebnf specified. If this ever passes again,
    /// the file has been reverted to describing a language nothing accepts.
    [<Fact>]
    let ``grammar rejects the superseded META/REASON/EDGE syntax`` () =
        let old =
            String.Join(
                "\n",
                [ "META {"
                  "  name: \"old\""
                  "}"
                  "REASON think title: \"Think\" {"
                  "  input: a, b"
                  "  transform: Refine"
                  "}"
                  "EDGE think -> act when score >= 0.5"
                  "" ]
            )

        Assert.False(grammarAccepts old, "grammar still admits the superseded syntax")

    /// `sp` is load-bearing, not decoration: WotParser's node header regex uses
    /// `\s+` between `node` and the id, so a decoder allowed to omit that space
    /// would emit unparseable output. Both must reject it.
    [<Fact>]
    let ``grammar and parser agree that the space after node is required`` () =
        let text =
            String.Join("\n", [ "meta {"; "  name = \"t\""; "}"; "workflow {"; "  node\"n\"{"; "  }"; "}"; "" ])

        Assert.False(grammarAccepts text, "grammar admits `node\"n\"{`, which WotParser cannot parse")
        Assert.False(parserAccepts text, "WotParser unexpectedly accepted `node\"n\"{`")

    /// Line orientation is the property the old grammar most badly missed.
    [<Fact>]
    let ``grammar rejects a workflow collapsed onto one line`` () =
        let oneLine = "meta { name = \"t\" } workflow { node \"n\" { goal = \"g\" } }\n"
        Assert.False(grammarAccepts oneLine, "grammar admits a single-line workflow")

    [<Fact>]
    let ``grammar rejects a closing brace that shares its line`` () =
        // WotParser only closes a block on `line = "}"` after trimming, so
        // trailing content on the brace line leaves the node open.
        let text =
            String.Join("\n", [ "meta {"; "  name = \"t\""; "}"; "workflow {"; "  node \"n\" {"; "  } edge \"n\" -> \"n\""; "}"; "" ])

        Assert.False(grammarAccepts text, "grammar admits a closing brace sharing its line")

    [<Fact>]
    let ``grammar rejects an unknown node kind`` () =
        // An unrecognised kind silently degrades to `reason` in WotParser, so
        // the grammar must not let a decoder emit one.
        let text =
            String.Join(
                "\n",
                [ "meta {"; "  name = \"t\""; "}"; "workflow {"; "  node \"n\" kind=\"ponder\" {"; "  }"; "}"; "" ]
            )

        Assert.False(grammarAccepts text, "grammar admits kind=\"ponder\"")

    // ── drift detection against the parser's own source ─────────────────────

    /// Reads the keys WotParser matches literally (`Some("goal", v)`) straight
    /// out of its source and requires each to appear in the grammar. This is
    /// the direction a behavioural test cannot cover: a key added to the parser
    /// but not the grammar is invisible until a decoder never emits it.
    [<Fact>]
    let ``every key WotParser matches literally appears in the grammar`` () =
        let src = Path.Combine(baseDir, "sources", "WotParser.fs")
        Assert.True(File.Exists src, $"WotParser.fs not copied to output: {src}")

        let parserKeys =
            Regex.Matches(File.ReadAllText src, @"Some\(\s*""([a-z_][a-z0-9_]*)""\s*,")
            |> Seq.map (fun m -> m.Groups[1].Value)
            |> Set.ofSeq

        Assert.True(parserKeys.Count > 10, $"extracted only {parserKeys.Count} keys — the regex has drifted")

        let grammarText = File.ReadAllText grammarPath

        let missing =
            parserKeys
            |> Set.filter (fun k -> not (grammarText.Contains($"\"{k}\"")))
            |> Set.toList

        Assert.True(
            List.isEmpty missing,
            $"""WotParser recognises keys absent from grammars/wot.ebnf: {String.Join(", ", missing)}"""
        )
