---
id: tool-formats
title: Tool Call Formats
category: learned
confidence: high
source: observed
tags: tools, parsing, llm
created: 2024-01-01T00:00:00Z
updated: 2024-01-01T00:00:00Z
---

# Tool Call Formats Supported by TARS

Different LLMs output tool calls in different formats. TARS supports all of these:

## 1. Legacy Format
```
TOOL:ToolName:input string here
```

Simple colon-separated format. First colon separates name, rest is input.

## 2. JSON Block
```
```tool
{"name": "ToolName", "arguments": {"param": "value"}}
```
```

Fenced code block with `tool` or `json` language hint.

## 3. XML-Style
```
<tool_call>{"name": "ToolName", "arguments": {"param": "value"}}</tool_call>
```

Wraps JSON in XML-like tags. Common in Claude outputs.

## 4. Inline JSON
```
{"name": "ToolName", "arguments": {"param": "value"}}
```

Plain JSON object with `name`/`tool`/`function` key.

## Multiple Tools

When multiple tool calls appear in one response, they're parsed as `MultiToolCall`.
Each is executed in order and results are collected.

