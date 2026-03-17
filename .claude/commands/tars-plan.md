Use the TARS cognitive compiler bridge to execute a goal.

TARS compiles goals into structured execution plans (Workflow-of-Thought DAGs). You provide the reasoning intelligence. TARS handles pattern selection, tool dispatch, validation, and learning from outcomes.

## Workflow

### Step 1 — Compile the plan

Call the `tars_compile_plan` MCP tool:

```json
{"goal": "$ARGUMENTS"}
```

TARS analyzes the goal, selects the best reasoning pattern (Chain-of-Thought, ReAct, Graph-of-Thought, Tree-of-Thought) based on goal characteristics and past success history, and returns a plan manifest.

### Step 2 — Read the manifest

The manifest contains:
- `plan_id` — unique identifier for this execution
- `pattern` — the reasoning pattern TARS selected
- `nodes` — a map of node objects, each with an `id`, `type`, and `prompt` or `tool_spec`
- `entryNode` — the node to start from
- `edges` — maps each node id to its `next` node(s)

Node types:
- **Reason** — requires your reasoning (analysis, synthesis, planning)
- **Tool** — requires calling a TARS tool or external action
- **Validate** — requires checking an output against acceptance criteria
- **Memory** — requires querying or storing knowledge

### Step 3 — Walk the DAG

Start at `entryNode`. For each node, based on its type:

**Reason nodes:** Read the `prompt` field. Use your own reasoning to produce an answer. This is where your intelligence drives the plan.

**Tool nodes:** Call `tars_execute_step`:
```json
{"plan_id": "<id>", "node_id": "<node_id>", "input": "<your input or prior node output>"}
```

**Validate nodes:** Call `tars_validate_step`:
```json
{"plan_id": "<id>", "node_id": "<node_id>", "content": "<the output to validate>"}
```
If validation fails, re-examine the upstream node output and retry.

**Memory nodes:** Call `tars_memory_op`:
```json
{"plan_id": "<id>", "node_id": "<node_id>", "op": "query|store", "key": "<key>", "value": "<value if storing>"}
```

### Step 4 — Follow edges

After completing a node, look up its id in `edges` to find the next node(s). If multiple next nodes are listed, execute them in order (they may be independent branches). Continue until no further edges remain.

### Step 5 — Complete the plan

When all nodes are done, call `tars_complete_plan`:
```json
{"plan_id": "<id>", "output": "<final synthesized result>"}
```

## Example

Goal: "Find unused functions in src/Tars.Core"

**1. Compile:**
Call `tars_compile_plan` with `{"goal": "Find unused functions in src/Tars.Core"}`.

Returns:
```json
{
  "plan_id": "p-a1b2c3",
  "pattern": "ReAct",
  "entryNode": "n1",
  "nodes": {
    "n1": {"id": "n1", "type": "Tool", "tool_spec": {"name": "codebase_search", "args": {"scope": "src/Tars.Core", "query": "public function definitions"}}},
    "n2": {"id": "n2", "type": "Reason", "prompt": "Given the function list and call graph, identify functions with zero callers outside their own module."},
    "n3": {"id": "n3", "type": "Validate", "prompt": "Confirm each candidate is truly unused — check for reflection, serialization, or interface implementations."}
  },
  "edges": {"n1": ["n2"], "n2": ["n3"]}
}
```

**2. Execute n1 (Tool):**
Call `tars_execute_step` with `{"plan_id": "p-a1b2c3", "node_id": "n1", "input": ""}`.
TARS returns a list of function definitions and their call sites.

**3. Execute n2 (Reason):**
Read the prompt. Analyze the tool output. Produce a list of candidate unused functions with reasoning for each.

**4. Execute n3 (Validate):**
Call `tars_validate_step` with `{"plan_id": "p-a1b2c3", "node_id": "n3", "content": "<your candidate list>"}`.
TARS checks against golden traces and acceptance criteria. Returns pass/fail per candidate.

**5. Complete:**
Call `tars_complete_plan` with the validated final list.

## Automatic behaviors

You do not need to manage these — TARS handles them behind the scenes:
- **Pattern selection:** TARS picks the optimal reasoning pattern based on goal analysis and historical success rates. The manifest reflects this choice.
- **Regression detection:** Validation nodes compare results against golden traces from prior successful runs. Regressions are flagged automatically.
- **Learning:** Plan outcomes (success, failure, timing) feed back into pattern selection weights so future plans improve.
