Start a Ralph loop for the GA chatbot — a Demerzel-governed iterative improvement cycle.

Usage: /ga-ralph [subcommand] [args...]

Subcommands:
- (none)         — start the Ralph loop (iterative improvement)
- status         — show current chatbot skill inventory and test results
- test           — run GA chatbot tests only

## Full Loop (default)

When invoked without a subcommand (or with a goal string), start a real Ralph loop using the ralph-loop plugin.

Invoke the `/ralph-loop` skill with this prompt:

```
/ralph-loop "You are working on the GA chatbot at C:/Users/spare/source/repos/ga. Your mission: make the chatbot a fully functional music companion. Each iteration: 1) Read GaPlugin.cs to understand current skill inventory. 2) Run: cd C:/Users/spare/source/repos/ga && dotnet test Tests/Common/GA.Business.ML.Tests/GA.Business.ML.Tests.csproj --filter 'FullyQualifiedName~SkillRoutingTests' --verbosity quiet. 3) Identify the highest-impact missing capability or failing test. 4) Implement one focused improvement (new skill, bug fix, or test). 5) Build and test to verify. 6) If all tests pass and the chatbot covers: session context, practice routines, interval quizzes, chord quizzes, scale practice, and progress tracking — output <promise>RALPH COMPLETE</promise>. Key files: GaPlugin.cs (skill registry), Common/GA.Business.ML/Agents/Skills/ (skills), Tests/Common/GA.Business.ML.Tests/Skills/SkillRoutingTests.cs (tests). $ARGUMENTS" --max-iterations 15 --completion-promise "RALPH COMPLETE"
```

## Status

When `status` is passed:

1. Read the skill registry:
```bash
cd C:/Users/spare/source/repos/ga && grep -E "AddS(ingleton|coped)<IOrchestratorSkill" Common/GA.Business.Core.Orchestration/Plugins/GaPlugin.cs
```

2. Run tests:
```bash
cd C:/Users/spare/source/repos/ga && dotnet test Tests/Common/GA.Business.ML.Tests/GA.Business.ML.Tests.csproj --filter "FullyQualifiedName~SkillRoutingTests" --verbosity quiet
```

3. Report: skill count, hook count, test pass/fail, and coverage gaps.

## Test

When `test` is passed:
```bash
cd C:/Users/spare/source/repos/ga && dotnet test Tests/Common/GA.Business.ML.Tests/GA.Business.ML.Tests.csproj --filter "FullyQualifiedName~SkillRoutingTests" --verbosity quiet
```

Report pass/fail counts.
