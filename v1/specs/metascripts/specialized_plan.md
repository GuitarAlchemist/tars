id: specialized-plan
# Specialized Coordination Spec

Combines hierarchical leaders with specialized roles.

```metascript
SPAWN CH 1 HIERARCHICAL
SPAWN CE 2 SPECIALIZED
CONNECT agent-1 agent-2 mentor
CONNECT agent-0 agent-2 broadcast
METRIC accuracy 1.20
METRIC latency 0.45
```

```expectations
rules=6
max_depth=1
spawn_count=2
connection_count=2
metric.accuracy=1.20
metric.latency=0.45
```
