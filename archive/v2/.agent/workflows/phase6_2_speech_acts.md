---
description: Implement Phase 6.2 Semantic Speech Acts
---

# Phase 6.2: Semantic Speech Acts

This workflow implements the Semantic Speech Acts pattern (Request/Inform) in the Evolution Engine and adds telemetry.

## Steps

1. **Add Logger to GraphExecutor**
    - Update `GraphExecutor` constructor to accept a logger (e.g., `string -> unit`).
    - Update `GraphRuntime.GraphContext` to include the logger.
    - Log message interactions in `GraphRuntime` (Thinking, Acting, Observing).

2. **Refactor `generateTask` in `Engine.fs`**
    - Change `generateTask` to use `GraphExecutor` instead of `ctx.Llm.CompleteAsync`.
    - Construct a `SemanticMessage` with `Performative.Request` containing the task generation prompt.
    - Send the message to `CurriculumAgent` using `Kernel.receiveMessage`.
    - Run `GraphExecutor.RunAgentLoop`.
    - Extract the response from the agent's output (which should be the JSON).

3. **Verify `executeTask`**
    - Ensure `executeTask` uses `Performative.Request`.
    - Ensure `GraphExecutor` logs the interaction.

4. **Update `Evolve.fs`**
    - Pass a logger (e.g., `Log.Information`) to `GraphExecutor` when initializing `EvolutionContext`.

5. **Verification**
    - Run `tars evolve --max-iterations 1` and check logs for "Request" and "Inform" (or at least the logged interactions).
