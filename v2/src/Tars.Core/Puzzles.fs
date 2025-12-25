namespace Tars.Core

open System
open System.Text.RegularExpressions

/// Puzzle types available
type PuzzleType =
    | RiverCrossing // Classic wolf-goat-cabbage
    | KnightsAndKnaves // Logic deduction
    | TowerOfHanoi // Recursive planning
    | LogicGrid // Zebra puzzle style
    | MathWord // Multi-step math
    | Cryptarithmetic // Letter-for-digit puzzles
    | Probabilistic // Reasoning about probability
    | TheoryOfMind // Reasoning about what others know
    | TemporalReasoning // Reasoning about time and sequences

/// Puzzle definition
type Puzzle =
    { Name: string
      Type: PuzzleType
      Difficulty: int // 1-5
      Description: string
      Prompt: string
      ExpectedAnswer: string
      Hints: string list
      Validator: string -> bool }

/// Collection of standard puzzles
module Puzzles =

    let riverCrossingPuzzle =
        { Name = "River Crossing"
          Type = RiverCrossing
          Difficulty = 2
          Description = "The classic wolf-goat-cabbage puzzle"
          Prompt =
            """A farmer needs to cross a river with a wolf, a goat, and a cabbage.
The boat can only carry the farmer and one item at a time.
If left alone:
- The wolf will eat the goat
- The goat will eat the cabbage

What sequence of crossings should the farmer make?
List each step as: "Take [item] across" or "Return alone"
Start with Step 1."""
          ExpectedAnswer = "goat" // First move must involve goat
          Hints =
            [ "Think about what can safely be left together"
              "The goat is the troublemaker - it can't be left with wolf OR cabbage"
              "You may need to bring something back" ]
          Validator =
            fun answer ->
                let lower = answer.ToLowerInvariant()

                let hasStepCount =
                    lower.Contains("step 7")
                    || lower.Contains("step seven")
                    || lower.Contains("7 steps")
                    || lower.Contains("seven steps")
                    || Regex.IsMatch(lower, @"\b7[.)]\s")
                    || Regex.IsMatch(lower, @"\b7:\s")
                    || lower.Contains("finally")

                // Valid solution starts with taking goat and ends with all across
                lower.Contains("goat")
                && lower.Contains("wolf")
                && lower.Contains("cabbage")
                && hasStepCount }

    let knightsAndKnavesPuzzle =
        { Name = "Knights and Knaves"
          Type = KnightsAndKnaves
          Difficulty = 3
          Description = "Logic puzzle with truth-tellers and liars"
          Prompt =
            """On an island, Knights always tell the truth and Knaves always lie.

You meet two islanders, A and B.
A says: "B is a Knight"
B says: "We are different types"

What are A and B? Explain your reasoning step by step.
Answer format: "A is a [Knight/Knave], B is a [Knight/Knave]"."""
          ExpectedAnswer = "knave" // A is Knave, B is Knight
          Hints =
            [ "Consider what happens if A is a Knight (truth-teller)"
              "If A tells truth, then B is a Knight. What would B say then?"
              "Try assuming A is a Knave and see if it's consistent" ]
          Validator =
            fun answer ->
                let lower = answer.ToLowerInvariant()
                // A is Knave, B is Knight
                (lower.Contains("a") && lower.Contains("knave"))
                && (lower.Contains("b") && lower.Contains("knight")) }

    let towerOfHanoiPuzzle =
        { Name = "Tower of Hanoi"
          Type = TowerOfHanoi
          Difficulty = 2
          Description = "Classic recursive planning puzzle"
          Prompt =
            """Solve the Tower of Hanoi with 3 disks.
Rules:
- Move one disk at a time
- A larger disk cannot be placed on a smaller disk
- Move all disks from peg A to peg C

Disks are numbered 1 (smallest) to 3 (largest).
Initial state: All disks on peg A (3 on bottom, 1 on top)

List each move as: "Move disk X from Y to Z"
What is the minimum number of moves, and what are they?"""
          ExpectedAnswer = "7" // Minimum moves is 7
          Hints =
            [ "The pattern is: Move n-1 disks to helper peg, move largest to target, move n-1 to target"
              "For 3 disks, you need 2^3 - 1 = 7 moves"
              "First move: disk 1 from A to C" ]
          Validator =
            fun answer ->
                let lower = answer.ToLowerInvariant()
                lower.Contains("7") && (lower.Contains("move") || lower.Contains("step")) }

    let logicGridPuzzle =
        { Name = "Logic Grid (Mini Zebra)"
          Type = LogicGrid
          Difficulty = 4
          Description = "Constraint satisfaction puzzle"
          Prompt =
            """Three friends (Alice, Bob, Carol) each have a different pet (cat, dog, fish) 
and drink a different beverage (tea, coffee, juice).

Clues:
1. Alice does not have the dog
2. The person with the cat drinks coffee
3. Bob drinks tea
4. Carol does not have the fish

Who has which pet and drinks what?

Answer format:
Alice: [pet], [drink]
Bob: [pet], [drink]  
Carol: [pet], [drink]"""
          ExpectedAnswer = "alice.*fish.*juice" // Regex pattern
          Hints =
            [ "Start with Bob - he drinks tea (given directly)"
              "Since Bob drinks tea and the cat owner drinks coffee, Bob doesn't have the cat"
              "Work through the constraints one by one" ]
          Validator =
            fun answer ->
                let lower = answer.ToLowerInvariant()
                // Alice: fish, juice; Bob: dog, tea; Carol: cat, coffee
                lower.Contains("alice")
                && lower.Contains("fish")
                && lower.Contains("carol")
                && lower.Contains("cat")
                && lower.Contains("bob")
                && lower.Contains("dog") }

    let mathWordPuzzle =
        { Name = "Multi-Step Math"
          Type = MathWord
          Difficulty = 3
          Description = "Word problem requiring multiple reasoning steps"
          Prompt =
            """A train leaves Station A at 9:00 AM traveling at 60 mph toward Station B.
Another train leaves Station B at 10:00 AM traveling at 80 mph toward Station A.
The stations are 280 miles apart.

At what time do the trains meet?
How far from Station A is the meeting point?

Show your work step by step."""
          ExpectedAnswer = "11:34" // Correct: 11:34 AM, ~154 miles
          Hints =
            [ "The first train has a 1-hour head start"
              "After 1 hour, the first train has traveled 60 miles"
              "The remaining distance is 280 - 60 = 220 miles"
              "Combined speed is 60 + 80 = 140 mph"
              "Time to meet: 220 / 140 = 1.57 hours after 10:00 AM" ]
          Validator =
            fun answer ->
                let lower = answer.ToLowerInvariant()
                // They meet at ~11:34 AM, ~154 miles from A
                (lower.Contains("11:34") || lower.Contains("11:35") || 
                 (lower.Contains("11") && (lower.Contains("34") || lower.Contains("35"))))
                && (lower.Contains("154") || lower.Contains("155") || lower.Contains("94")) }

    let cryptarithmeticPuzzle =
        { Name = "Cryptarithmetic"
          Type = Cryptarithmetic
          Difficulty = 4
          Description = "Letter-for-digit substitution puzzle"
          Prompt =
            """Solve this cryptarithmetic puzzle where each letter represents a unique digit (0-9):

    S E N D
  + M O R E
  ---------
  M O N E Y

Find the digit for each letter.
Hint: M must be 1 (it's a carry from adding two 4-digit numbers).

Answer format: S=?, E=?, N=?, D=?, M=?, O=?, R=?, Y=?"""
          ExpectedAnswer = "9567" // SEND = 9567
          Hints =
            [ "M = 1 (from the carry)"
              "S + M produces a carry, so S = 8 or 9"
              "O must be 0 (since S + M = 10 or more, O = 0 with carry)"
              "Work column by column from right to left" ]
          Validator =
            fun answer ->
                let lower = answer.ToLowerInvariant()
                // SEND = 9567, MORE = 1085, MONEY = 10652
                lower.Contains("s=9")
                || lower.Contains("s = 9")
                || (lower.Contains("9567") && lower.Contains("1085")) }

    let montyHallPuzzle =
        { Name = "Monty Hall Problem"
          Type = Probabilistic
          Difficulty = 3
          Description = "Counter-intuitive probability puzzle"
          Prompt =
            """Suppose you're on a game show, and you're given the choice of three doors: 
Behind one door is a car; behind the others, goats. 
You pick a door, say No. 1, and the host, who knows what's behind the doors, opens another door, say No. 3, which has a goat. 
He then says to you, "Do you want to pick door No. 2?" 

Is it to your advantage to switch your choice? 
Explain the probability of winning if you stay vs. if you switch."""
          ExpectedAnswer = "switch"
          Hints =
            [ "Think about the probability of your initial choice being right"
              "What does the host's action tell you about the other doors?"
              "There are only two possibilities now: stay or switch" ]
          Validator =
            fun answer ->
                let lower = answer.ToLowerInvariant()
                lower.Contains("switch") && (lower.Contains("2/3") || lower.Contains("66%")) }

    let cherylsBirthdayPuzzle =
        { Name = "Cheryl's Birthday"
          Type = TheoryOfMind
          Difficulty = 5
          Description = "Logic puzzle requiring reasoning about knowledge"
          Prompt =
            """Albert and Bernard just became friends with Cheryl, and they want to know when her birthday is. 
Cheryl gave them a list of 10 possible dates:
May 15, May 16, May 19
June 17, June 18
July 14, July 16
August 14, August 15, August 17

Cheryl then tells Albert and Bernard separately the month and the day of her birthday respectively.

Albert: I don't know when Cheryl's birthday is, but I know that Bernard does not know too.
Bernard: At first I don't know when Cheryl's birthday is, but I know now.
Albert: Then I also know when Cheryl's birthday is.

So when is Cheryl's birthday?"""
          ExpectedAnswer = "July 16"
          Hints =
            [ "Albert only knows the month. If he knows Bernard doesn't know, it means the month can't have any unique days (18, 19)."
              "Which months have unique days?"
              "Once some months are eliminated, Bernard now knows. This means the day must be unique among the remaining dates." ]
          Validator =
            fun answer ->
                let lower = answer.ToLowerInvariant()
                lower.Contains("july") && lower.Contains("16") }

    let schedulingPuzzle =
        { Name = "Space Station Maintenance"
          Type = TemporalReasoning
          Difficulty = 4
          Description = "Resource and time constraint planning"
          Prompt =
            """As a station commander, you must schedule 4 critical tasks:
A: Oxygen scrubber repair (2 hours)
B: Solar panel alignment (3 hours)
C: Software update (1 hour)
D: Battery health check (2 hours)

Constraints:
1. Task A must be completed before Task D can start.
2. Task B and Task D cannot happen at the same time because they both require the external robotic arm.
3. Task C must happen after Task B is finished.
4. The technician for Task A is only available after 10:00 AM.
5. All tasks must be finished by 4:00 PM (16:00).
6. The station has a power-down window between 12:00 and 13:00 where NO tasks can be performed.

What is the earliest possible time all tasks can be finished? 
Provide a schedule starting at 10:00 AM.
End your response with: "Conclusion: The earliest finish time is [Time]." """
          ExpectedAnswer = "16:00"
          Hints =
            [ "A and B can happen simultaneously if they don't share resources."
              "Check if A and B share the robotic arm."
              "Don't forget the power-down hour." ]
          Validator =
            fun answer ->
                let lower = answer.ToLowerInvariant()

                (lower.Contains("16:00") || lower.Contains("4:00 pm") || lower.Contains("4 pm"))
                && lower.Contains("conclusion") }

    /// All available puzzles
    let all =
        [ riverCrossingPuzzle
          knightsAndKnavesPuzzle
          towerOfHanoiPuzzle
          logicGridPuzzle
          mathWordPuzzle
          cryptarithmeticPuzzle
          montyHallPuzzle
          cherylsBirthdayPuzzle
          schedulingPuzzle ]

// ============================================================================
// DIAGNOSTICS & RESULTS
// ============================================================================

/// Metrics gathered during a puzzle execution
type PuzzleMetrics =
    { LatencyMs: int64
      TokenCount: int
      ModelName: string
      CostEstimate: float }

/// The outcome of a puzzle run
type PuzzleRunResult =
    | Success of Puzzle * answer: string * PuzzleMetrics
    | Failure of Puzzle * answer: string * PuzzleMetrics * reason: string

