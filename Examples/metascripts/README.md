# TARS Metascript Examples

This directory contains example metascripts for the TARS system.

## Examples

### hello_world.tars

A simple hello world example that demonstrates the basic features of the TARS metascript system:
- Variables
- Actions (log)
- Conditional logic (IF/ELSE)

### tars_augment_collaboration.tars

A more complex example that demonstrates TARS-Augment collaboration via MCP:
- MCP integration (mcp_send, mcp_receive)
- Variable substitution
- Prompts
- Actions (log, mcp_send)
- Conditional logic (IF/ELSE)

## Running the Examples

You can run these examples using the TARS CLI:

```
tarscli metascript execute Examples/metascripts/hello_world.tars
tarscli metascript validate Examples/metascripts/tars_augment_collaboration.tars
```

You can also run a demo of the metascript capabilities:

```
tarscli demo --type metascript
```

## Creating Your Own Metascripts

See the [Metascripts documentation](../../docs/Features/Metascripts.md) for information on creating your own metascripts.
