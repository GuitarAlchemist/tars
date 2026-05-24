# Web Ingest and Learn

Goal: fetch a web page, summarize it, and store it so TARS can reuse it in RAG and the knowledge base.

## Prereqs
- Network access enabled (this workspace already allows it).
- `web_fetch` tool is registered via `Tars.Tools.Standard`.
- Any vector store/KB already configured in your run config (Metascript auto-index will use it).

## Minimal Metascript workflow
```json
{
  "name": "web-ingest",
  "steps": [
    {
      "id": "fetch",
      "type": "tool",
      "tool": "web_fetch",
      "params": { "url": "https://example.com" },
      "outputs": ["page"]
    },
    {
      "id": "summarize",
      "type": "agent",
      "agent": "Assistant",
      "instruction": "Summarize the page content and extract 3-5 key facts. Keep under 150 words.",
      "inputs": { "context": "{{fetch.page}}" },
      "outputs": ["summary"]
    }
  ]
}
```

Run it:
```
tars run web-ingest.tars
```

What happens:
- `web_fetch` downloads the page (first 64k chars max) and returns the text.
- The agent step summarizes it. Metascript auto-indexing will embed and store the agent output when RAG auto-index is enabled in your config, making it retrievable for later queries.

## Tips
- Change the URL in the `fetch` step to target any page/API that returns text.
- For multiple URLs, wrap the fetch/summarize in a `loop` step with a list of URLs.
- To also log a KB entry, add a follow-up agent step: `“Create a KB note with title and bullet points for reuse.”`
