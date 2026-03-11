Run TARS CLI command from Claude Code.

Usage: /tars <subcommand> [args...]

Examples:
- /tars agent run "Analyze the codebase"
- /tars evolve --max-iterations 3 --focus "improve tests"
- /tars evolve --loop 3 --max-iterations 5
- /tars agent wot "Find and fix code smells"
- /tars diag
- /tars know status

Execute the following command and report the results:

```bash
cd C:/Users/spare/source/repos/tars/v2 && dotnet run --project src/Tars.Interface.Cli -- $ARGUMENTS
```

If the command takes longer than 2 minutes, it may be an LLM-dependent operation. Report what you see and suggest using --quiet for less output.
