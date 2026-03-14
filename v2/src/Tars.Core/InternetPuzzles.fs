namespace Tars.Core

open System
open System.Net.Http
open System.Text.Json
open System.Text.Json.Serialization
open System.Threading.Tasks

/// Internet puzzle sources and fetcher
module InternetPuzzles =

    // ========================================
    // Open Trivia Database (opentdb.com)
    // ========================================
    
    /// Response from Open Trivia DB API
    type OpenTriviaResponse = {
        [<JsonPropertyName("response_code")>] ResponseCode: int
        [<JsonPropertyName("results")>] Results: OpenTriviaQuestion array
    }
    and OpenTriviaQuestion = {
        [<JsonPropertyName("type")>] Type: string
        [<JsonPropertyName("difficulty")>] Difficulty: string
        [<JsonPropertyName("category")>] Category: string
        [<JsonPropertyName("question")>] Question: string
        [<JsonPropertyName("correct_answer")>] CorrectAnswer: string
        [<JsonPropertyName("incorrect_answers")>] IncorrectAnswers: string array
    }

    // ========================================
    // Hugging Face Datasets API
    // ========================================

    /// Row from Hugging Face AI2 ARC dataset
    type HuggingFaceArcRow = {
        [<JsonPropertyName("row_idx")>] RowIdx: int
        [<JsonPropertyName("row")>] Row: ArcQuestion
        [<JsonPropertyName("truncated_cells")>] TruncatedCells: string array
    }
    and ArcQuestion = {
        [<JsonPropertyName("id")>] Id: string
        [<JsonPropertyName("question")>] Question: string
        [<JsonPropertyName("choices")>] Choices: ArcChoices
        [<JsonPropertyName("answerKey")>] AnswerKey: string
    }
    and ArcChoices = {
        [<JsonPropertyName("text")>] Text: string array
        [<JsonPropertyName("label")>] Label: string array
    }

    type HuggingFaceArcResponse = {
        [<JsonPropertyName("dataset")>] Dataset: string
        [<JsonPropertyName("config")>] Config: string
        [<JsonPropertyName("split")>] Split: string
        [<JsonPropertyName("rows")>] Rows: HuggingFaceArcRow array
    }

    // ========================================
    // GSM8K Dataset (Multi-step Math Problems)
    // ========================================

    /// Row from GSM8K dataset
    type Gsm8kRow = {
        [<JsonPropertyName("row_idx")>] RowIdx: int
        [<JsonPropertyName("row")>] Row: Gsm8kQuestion
        [<JsonPropertyName("truncated_cells")>] TruncatedCells: string array
    }
    and Gsm8kQuestion = {
        [<JsonPropertyName("question")>] Question: string
        [<JsonPropertyName("answer")>] Answer: string
    }

    type HuggingFaceGsm8kResponse = {
        [<JsonPropertyName("dataset")>] Dataset: string
        [<JsonPropertyName("config")>] Config: string
        [<JsonPropertyName("split")>] Split: string
        [<JsonPropertyName("rows")>] Rows: Gsm8kRow array
    }

    // ========================================
    // Puzzle Sources
    // ========================================

    type PuzzleSource =
        | OpenTriviaDB
        | HuggingFaceARC
        | HuggingFaceARCEasy
        | GSM8K  // Grade School Math - multi-step word problems

    type InternetPuzzle = {
        Source: PuzzleSource
        SourceId: string
        Question: string
        Choices: (string * string) list  // (label, text)
        CorrectAnswer: string
        Category: string
        Difficulty: string
    }

    // ========================================
    // HTML Entity Decoder
    // ========================================

    let decodeHtmlEntities (text: string) =
        System.Net.WebUtility.HtmlDecode(text)

    // ========================================
    // Fetchers
    // ========================================

    /// Fetch a random math/science question from Open Trivia DB
    let fetchFromOpenTriviaDB (http: HttpClient) (category: int option) (count: int) : Task<InternetPuzzle list> =
        task {
            // Categories: 17=Science, 18=Computers, 19=Math, 9=General Knowledge
            let categoryParam = 
                match category with
                | Some c -> $"&category={c}"
                | None -> ""
            
            let url = $"https://opentdb.com/api.php?amount={count}&type=multiple{categoryParam}"
            
            try
                let! response = http.GetStringAsync(url)
                let parsed = JsonSerializer.Deserialize<OpenTriviaResponse>(response)
                
                if parsed.ResponseCode = 0 then
                    return parsed.Results
                    |> Array.toList
                    |> List.map (fun q ->
                        let allAnswers = 
                            Array.append [| q.CorrectAnswer |] q.IncorrectAnswers
                            |> Array.mapi (fun i a -> (string (char (65 + i)), decodeHtmlEntities a))
                            |> Array.toList
                        
                        { Source = OpenTriviaDB
                          SourceId = Guid.NewGuid().ToString("N").[..7]
                          Question = decodeHtmlEntities q.Question
                          Choices = allAnswers
                          CorrectAnswer = "A"  // Correct is always first before shuffle
                          Category = q.Category
                          Difficulty = q.Difficulty })
                else
                    return []
            with ex ->
                printfn $"Error fetching from OpenTriviaDB: {ex.Message}"
                return []
        }

    /// Fetch ARC Challenge questions from Hugging Face
    let fetchFromHuggingFaceARC (http: HttpClient) (challenge: bool) (count: int) : Task<InternetPuzzle list> =
        task {
            let config = if challenge then "ARC-Challenge" else "ARC-Easy"
            let url = $"https://datasets-server.huggingface.co/first-rows?dataset=allenai/ai2_arc&config={config}&split=test"
            
            try
                let! response = http.GetStringAsync(url)
                let parsed = JsonSerializer.Deserialize<HuggingFaceArcResponse>(response)
                
                // Shuffle and take requested count
                let rng = Random()
                let shuffled = 
                    parsed.Rows 
                    |> Array.sortBy (fun _ -> rng.Next())
                    |> Array.take (min count parsed.Rows.Length)
                
                return shuffled
                |> Array.toList
                |> List.map (fun row ->
                    let q = row.Row
                    let choices = 
                        Array.zip q.Choices.Label q.Choices.Text
                        |> Array.map (fun (l, t) -> (l, t))
                        |> Array.toList
                    
                    { Source = if challenge then HuggingFaceARC else HuggingFaceARCEasy
                      SourceId = q.Id
                      Question = q.Question
                      Choices = choices
                      CorrectAnswer = q.AnswerKey
                      Category = "Science"
                      Difficulty = if challenge then "hard" else "easy" })
            with ex ->
                printfn $"Error fetching from Hugging Face: {ex.Message}"
                return []
        }

    /// Fetch GSM8K (Grade School Math) from Hugging Face - multi-step math word problems
    let fetchFromGSM8K (http: HttpClient) (count: int) : Task<InternetPuzzle list> =
        task {
            let url = "https://datasets-server.huggingface.co/first-rows?dataset=openai/gsm8k&config=main&split=test"
            
            try
                let! response = http.GetStringAsync(url)
                let parsed = JsonSerializer.Deserialize<HuggingFaceGsm8kResponse>(response)
                
                // Shuffle and take requested count
                let rng = Random()
                let shuffled = 
                    parsed.Rows 
                    |> Array.sortBy (fun _ -> rng.Next())
                    |> Array.take (min count parsed.Rows.Length)
                
                return shuffled
                |> Array.toList
                |> List.mapi (fun i row ->
                    let q = row.Row
                    // Extract the final numeric answer (after ####)
                    let finalAnswer = 
                        let parts = q.Answer.Split("####")
                        if parts.Length > 1 then parts.[1].Trim()
                        else ""
                    
                    { Source = GSM8K
                      SourceId = $"GSM8K-{row.RowIdx}"
                      Question = q.Question
                      Choices = []  // GSM8K is open-ended, not multiple choice
                      CorrectAnswer = finalAnswer
                      Category = "Multi-Step Math"
                      Difficulty = "hard" })  // GSM8K requires 2-8 steps
            with ex ->
                printfn $"Error fetching from GSM8K: {ex.Message}"
                return []
        }

    // ========================================
    // Convert to TARS Puzzle
    // ========================================

    /// Convert an internet puzzle to our internal Puzzle type
    let toPuzzle (puzzle: InternetPuzzle) : Puzzle =
        let sourceName = 
            match puzzle.Source with
            | OpenTriviaDB -> "OpenTriviaDB"
            | HuggingFaceARC -> "HuggingFace ARC-Challenge"
            | HuggingFaceARCEasy -> "HuggingFace ARC-Easy"
            | GSM8K -> "GSM8K"
        
        // GSM8K is open-ended, others are multiple choice
        let isMultipleChoice = puzzle.Choices.Length > 0
        
        let choicesText = 
            if isMultipleChoice then
                puzzle.Choices
                |> List.map (fun (label, text) -> $"{label}. {text}")
                |> String.concat "\n"
            else ""
        
        let difficultyNum =
            match puzzle.Difficulty.ToLower() with
            | "easy" -> 2
            | "medium" -> 3
            | "hard" -> 4
            | _ -> 3
        
        let prompt =
            if isMultipleChoice then
                $"""{puzzle.Question}

Choose the correct answer:
{choicesText}

State your answer as a single letter (A, B, C, or D) followed by your explanation."""
            else
                // GSM8K format - open-ended math problem
                $"""{puzzle.Question}

Solve this step by step. Show your work and end with the final numeric answer."""
        let validator : string -> bool = 
            if isMultipleChoice then
                fun answer ->
                    let upper = answer.ToUpperInvariant()
                    upper.Contains(puzzle.CorrectAnswer) ||
                    (puzzle.Choices 
                     |> List.tryFind (fun (label, _) -> label = puzzle.CorrectAnswer)
                     |> Option.map (fun (_, text) -> upper.Contains(text.ToUpperInvariant()))
                     |> Option.defaultValue false)
            else
                fun answer ->
                    let correctNum = puzzle.CorrectAnswer.Replace(",", "").Trim()
                    answer.Contains(correctNum)

        { Name = $"{sourceName}: {puzzle.SourceId}"
          Type = MathWord
          Difficulty = difficultyNum
          Description = $"From {sourceName} - {puzzle.Category}"
          Prompt = prompt
          ExpectedAnswer = puzzle.CorrectAnswer
          Hints = if puzzle.Source = GSM8K then ["Show step-by-step work"] else []
          Validator = validator }

    // ========================================
    // Combined Fetcher
    // ========================================

    /// Fetch puzzles from multiple sources
    let fetchPuzzles (http: HttpClient) (sources: PuzzleSource list) (countPerSource: int) : Task<Puzzle list> =
        task {
            let mutable allPuzzles = []
            
            for source in sources do
                let! puzzles = 
                    match source with
                    | OpenTriviaDB -> 
                        fetchFromOpenTriviaDB http (Some 19) countPerSource  // Math category
                    | HuggingFaceARC ->
                        fetchFromHuggingFaceARC http true countPerSource
                    | HuggingFaceARCEasy ->
                        fetchFromHuggingFaceARC http false countPerSource
                    | GSM8K ->
                        fetchFromGSM8K http countPerSource
                
                allPuzzles <- allPuzzles @ (puzzles |> List.map toPuzzle)
            
            return allPuzzles
        }

    /// Available puzzle sources with descriptions
    let availableSources = [
        (OpenTriviaDB, "Open Trivia Database - Math & Science questions")
        (HuggingFaceARC, "AI2 Reasoning Challenge - Hard science QA")
        (HuggingFaceARCEasy, "AI2 Reasoning Challenge - Easy science QA")
        (GSM8K, "GSM8K - Multi-step arithmetic word problems (2-8 steps)")
    ]
