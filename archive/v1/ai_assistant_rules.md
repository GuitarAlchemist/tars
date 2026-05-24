# AI Assistant Rules for TARS Project

These rules must be followed by any AI assistant working on this Blazor application to prevent breaking functionality.

## General Rules

1. **Minimal Changes Rule**: Only modify the specific code that needs fixing, leaving everything else untouched.

2. **Preserve Structure Rule**: Never alter the basic structure of components, especially navigation and layout components.

3. **Service Injection Rule**: Don't change service injections unless explicitly asked to do so.

4. **Rendering Mode Rule**: Never modify `@rendermode` directives as they're critical for interactivity.

5. **Verification Step**: Before suggesting changes, mentally review if they might break existing functionality.

6. **No Assumptions Rule**: Don't assume missing files or components - only work with what's explicitly shown.

7. **Focus on Reported Issues**: Address only the specific issues reported rather than trying to "improve" working code.

## Blazor-Specific Guidelines

1. **Component Parameters**: Use the correct syntax for generic components (e.g., `T="string"` not `<string>`).

2. **Interactive Rendering**: Don't modify the interactive rendering setup in Program.cs.

3. **JavaScript Interop**: Be careful with JS interop code - it's often critical for functionality.

4. **MudBlazor Components**: Follow MudBlazor documentation precisely for component usage.

5. **State Management**: Don't change how component state is managed unless specifically asked.

## When in Doubt

If unsure about a change, suggest minimal, focused fixes or ask for clarification rather than making broad changes.