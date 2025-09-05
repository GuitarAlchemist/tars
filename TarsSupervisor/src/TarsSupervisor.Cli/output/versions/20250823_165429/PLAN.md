### NEXT IMPROVEMENT PLAN

#### Step 1: Implement Robust Validation and Metrics (Rationale: Ensure reliability before adding optional features)
- **Action**: Introduce pluggable validators for .NET build/test.
- **Risk**: Potential integration issues with existing build processes.

#### Step 2: Add Metrics Collection (Rationale: Monitor performance and stability)
- **Action**: Emit metrics.jsonl per iteration.
- **Risk**: Data collection might impact performance, especially in large iterations.

#### Step 3: Integrate Local LLM via Ollama (Rationale: Enhance AI capabilities for iterative improvements)
- **Action**: Add directive to integrate local LLM.
- **Risk**: Integration complexity and potential errors during initial setup.

#### Step 4: Implement Optional Auto-Apply Feature (Rationale: Enable automated application of validated changes)
- **Action**: Refine the rollback directive to include auto-apply logic if validation passes.
- **Risk**: Potential unintended side effects from auto-application.