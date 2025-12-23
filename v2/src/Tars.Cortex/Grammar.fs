namespace Tars.Cortex.Grammar

open System

type TaskItem = 
    { Name: string
      Description: string option }

type Goal = 
    { Name: string
      Tasks: TaskItem list }

module internal Ebnf =
    type Expr =
        | Terminal of string
        | NonTerminal of string
        | Sequence of Expr list
        | ZeroOrMore of Expr
        | BuiltIn of string

    type Rule = { Name: string; Body: Expr }

    let parseEbnf (input: string) : Rule list =
        let rec tokenize (chars: char list) acc =
            match chars with
            | [] -> List.rev acc
            | ' '::rest | '\t'::rest | '\r'::rest | '\n'::rest -> tokenize rest acc
            | ':'::':'::'='::rest -> tokenize rest ("::=" :: acc)
            | '*'::rest -> tokenize rest ("*" :: acc)
            | '"'::rest ->
                let rec readStr cs s =
                    match cs with
                    | '"'::rem -> (s, rem)
                    | c::rem -> readStr rem (s + string c)
                    | [] -> (s, [])
                let (s, rest2) = readStr rest ""
                tokenize rest2 ( $"\"%s{s}\"" :: acc)
            | c::rest when Char.IsLetterOrDigit(c) || c = '_' ->
                let rec readIdent cs s =
                    match cs with
                    | c2::rem when Char.IsLetterOrDigit(c2) || c2 = '_' -> readIdent rem (s + string c2)
                    | _ -> (s, cs)
                let (s, rest2) = readIdent rest (string c)
                tokenize rest2 (s :: acc)
            | c::rest -> tokenize rest (string c :: acc)

        let tokens = tokenize (Seq.toList input) []

        let rec parseExpr toks =
            match toks with
            | [] -> ([], [])
            | t::rest when t = "*" -> failwith "Unexpected *"
            | t::rest ->
                let (term, rem) =
                    if t.StartsWith("\"") then 
                        (Terminal(t.Trim('"')), rest)
                    else if t = "string_literal" then
                        (BuiltIn("string_literal"), rest)
                    else if t = "{" || t = "}" then 
                         (Terminal(t), rest)
                    else
                        (NonTerminal(t), rest)
                
                let (finalTerm, rem2) =
                    match rem with
                    | "*"::r -> (ZeroOrMore(term), r)
                    | _ -> (term, rem)
                
                // Check for start of next rule (Lookahead 2)
                let isNextRule = 
                    match rem2 with
                    | _::"::="::_ -> true
                    | _ -> false

                match rem2 with
                | [] -> ([finalTerm], [])
                | _ when isNextRule -> ([finalTerm], rem2)
                | "::="::_ -> ([finalTerm], rem2)
                | next::_ -> 
                     let (nextExprs, rem3) = parseExpr rem2
                     (finalTerm :: nextExprs, rem3)

        let rec parseRules toks acc =
            match toks with
            | [] -> List.rev acc
            | name::"::="::rest ->
                let (exprs, rem) = parseExpr rest
                let body = if exprs.Length = 1 then exprs[0] else Sequence exprs
                parseRules rem ({ Name = name; Body = body } :: acc)
            | _ -> failwith "Invalid EBNF syntax"

        parseRules tokens []

module internal Interpreter =
    open Ebnf

    type InputToken = 
        | TIdentifier of string
        | TString of string
        | TSymbol of string

    type ParseNode =
        | Node of string * ParseNode list
        | Leaf of InputToken

    let tokenizeInput (source: string) =
        let len = source.Length
        let rec loop i acc =
            if i >= len then List.rev acc
            else
                match source[i] with
                | c when Char.IsWhiteSpace(c) -> loop (i + 1) acc
                | '{' -> loop (i + 1) (TSymbol "{" :: acc)
                | '}' -> loop (i + 1) (TSymbol "}" :: acc)
                | '"' ->
                    let start = i + 1
                    let mutable endP = start
                    while endP < len && source[endP] <> '"' do endP <- endP + 1
                    if endP >= len then List.rev acc
                    else 
                        let s = source.Substring(start, endP - start)
                        loop (endP + 1) (TString s :: acc)
                | c when Char.IsLetter(c) ->
                    let start = i
                    let mutable endP = start
                    while endP < len && Char.IsLetter(source[endP]) do endP <- endP + 1
                    let word = source.Substring(start, endP - start)
                    loop endP (TIdentifier word :: acc)
                | _ -> loop (i + 1) acc
        loop 0 []

    let parse (rules: Rule list) (rootRule: string) (input: string) =
        let tokens = tokenizeInput input
        
        let ruleMap = rules |> List.map (fun r -> r.Name, r) |> Map.ofList

        let rec matchExpr expr toks =
            match expr with
            | Terminal s ->
                match toks with
                | TIdentifier val' :: rest when val' = s -> Some (Leaf(TIdentifier val'), rest)
                | TSymbol val' :: rest when val' = s -> Some (Leaf(TSymbol val'), rest)
                | _ -> None
            | BuiltIn "string_literal" ->
                match toks with
                | TString s :: rest -> Some (Leaf(TString s), rest)
                | _ -> None
            | BuiltIn _ -> None
            | NonTerminal name ->
                match Map.tryFind name ruleMap with
                | Some rule -> 
                    match matchExpr rule.Body toks with
                    | Some (node, rest) -> Some (Node(name, [node]), rest)
                    | None -> None
                | None -> failwithf $"Unknown rule %s{name}"
            | Sequence exprs ->
                let rec matchSeq es ts acc =
                    match es with
                    | [] -> Some (List.rev acc, ts)
                    | e::eres ->
                        match matchExpr e ts with
                        | Some (node, rest) -> matchSeq eres rest (node::acc)
                        | None -> None
                match matchSeq exprs toks [] with
                | Some (nodes, rest) -> Some (Node("seq", nodes), rest)
                | None -> None
            | ZeroOrMore sub ->
                let rec loop acc ts =
                    match matchExpr sub ts with
                    | Some (node, rest) -> loop (node::acc) rest
                    | None -> (List.rev acc, ts)
                let (nodes, rest) = loop [] toks
                Some (Node("list", nodes), rest)

        matchExpr (NonTerminal rootRule) tokens

module Parser =
    open Interpreter
    open Ebnf

    // Matches cortex.ebnf
    let private DefaultGrammar = """
program ::= goal*
goal    ::= "goal" string_literal "{" task* "}"
task    ::= "task" string_literal
"""

    let private mapToDomain (node: ParseNode) : Goal list =
        let rec extractGoals n =
            match n with
            | Node("program", [Node("list", goals)]) -> goals |> List.map extractGoal
            | Node("program", [child]) -> extractGoals child
            | _ -> []
            
        and extractGoal n =
            match n with
            | Node("goal", [Node("seq", [_; Leaf(TString name); _; Node("list", taskNodes); _])]) ->
                { Name = name; Tasks = taskNodes |> List.map extractTask }
            | _ -> failwith "Invalid goal structure"
            
        and extractTask n =
            match n with
            | Node("task", [Node("seq", [_; Leaf(TString name)])]) ->
                { Name = name; Description = None }
            | _ -> failwith "Invalid task structure"

        extractGoals node

    let parse (input: string) =
        let rules = Ebnf.parseEbnf DefaultGrammar
        match Interpreter.parse rules "program" input with
        | Some (node, []) -> mapToDomain node
        | Some (_, rest) -> 
            // For robustness, if rest is just whitespace, we might be ok, but our tokenizer removes whitespace.
            // So rest contains unparsed tokens.
            failwithf $"Parsing incomplete. Remaining tokens: %A{rest}"
        | None -> failwith "Parsing failed"
