id: swarm-plan
# Swarm Coordination Spec

Designed for swarm strategies emphasising cohesion.

```metascript
SPAWN NOREGRET 3 SWARM
CONNECT leader agent-1 align
METRIC cohesion 0.95
REPEAT loop 4
```

```expectations
rules=4
max_depth=4
spawn_count=1
connection_count=1
pattern=loop
metric.cohesion=0.95
```
