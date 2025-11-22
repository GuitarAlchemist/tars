id: dynamic-plan
# Dynamic Plan Coordination Spec

The dynamic plan focuses on hierarchical leadership with adaptive patterns.

```metascript
SPAWN QRE 2 HIERARCHICAL
SPAWN ML 1 FRACTAL
CONNECT leader agent-1 directive
CONNECT agent-1 agent-2 support
METRIC innovation 0.85
METRIC stability 0.72
REPEAT adaptive 3
```

```expectations
rules=7
max_depth=3
spawn_count=2
connection_count=2
pattern=adaptive
metric.innovation=0.85
metric.stability=0.72
```
