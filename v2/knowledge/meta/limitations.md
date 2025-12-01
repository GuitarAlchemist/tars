---
id: limitations
title: TARS Limitations
category: meta
confidence: medium
source: observed
tags: limitations, honesty, self-knowledge
created: 2024-01-01T00:00:00Z
updated: 2024-01-01T00:00:00Z
---

# What TARS Cannot Do (Yet)

## Current Limitations

### No Persistent State Across Sessions
- Agent memory resets on restart
- Knowledge base helps but isn't automatically loaded

### No Real-Time Learning
- Cannot update weights or fine-tune models
- Learning is via prompt engineering and knowledge storage

### Limited Multi-Agent Coordination
- EventBus exists but agents don't communicate yet
- No delegation or supervisor patterns implemented

### No Web Access
- Cannot browse websites or fetch URLs
- Would need WebSearch tool integration

### No Code Execution
- Can generate code but cannot run it in sandbox
- Tools exist but not sandboxed execution

## Planned Improvements

- [ ] MCP server integration
- [ ] Persistent agent memory
- [ ] Multi-agent delegation
- [ ] Sandboxed code execution
- [ ] Web browsing capability

