namespace Tars.Metascript

open Domain
open Parser

/// <summary>
/// Pre-built workflow templates for common patterns.
/// </summary>
module Templates =

    /// <summary>Simple Q&A workflow with RAG</summary>
    let ragQA (question: string) : Workflow =
        (workflow "RAG Q&A")
            .Description("Answer a question using retrieved context")
            .Version("1.0")
            .Input("question", "string", "The question to answer")
            .Retrieval("retrieve", question)
            .Agent("answer", "assistant",
                "Using the following context:\n{{retrieve.context}}\n\nAnswer this question: " + question)
            .Build()

    /// <summary>Code review workflow</summary>
    let codeReview (code: string) : Workflow =
        (workflow "Code Review")
            .Description("Review code for issues and improvements")
            .Version("1.0")
            .Input("code", "string", "The code to review")
            .Agent("analyze", "code-reviewer",
                "Analyze this code for bugs, security issues, and style problems:\n\n" + code)
            .Agent("suggest", "code-reviewer",
                "Based on the analysis:\n{{analyze.result}}\n\nSuggest specific improvements with code examples.")
            .Build()

    /// <summary>Research workflow with multiple sources</summary>
    let research (topic: string) : Workflow =
        (workflow "Research")
            .Description("Research a topic from multiple angles")
            .Version("1.0")
            .Input("topic", "string", "The topic to research")
            .Retrieval("docs", topic)
            .Agent("summarize", "researcher",
                "Summarize what we know about: " + topic + "\n\nContext:\n{{docs.context}}")
            .Agent("gaps", "researcher",
                "Based on this summary:\n{{summarize.result}}\n\nIdentify knowledge gaps and open questions.")
            .Agent("synthesize", "researcher",
                "Create a comprehensive research brief combining:\n- Summary: {{summarize.result}}\n- Gaps: {{gaps.result}}")
            .Build()

    /// <summary>Task decomposition workflow</summary>
    let taskDecomposition (task: string) : Workflow =
        (workflow "Task Decomposition")
            .Description("Break down a complex task into subtasks")
            .Version("1.0")
            .Input("task", "string", "The task to decompose")
            .Agent("analyze", "planner",
                "Analyze this task and identify its components:\n" + task)
            .Agent("decompose", "planner",
                "Based on the analysis:\n{{analyze.result}}\n\nCreate a numbered list of subtasks with dependencies.")
            .Agent("estimate", "planner",
                "For each subtask:\n{{decompose.result}}\n\nEstimate effort and identify risks.")
            .Build()

    /// <summary>Document summarization workflow</summary>
    let summarize (document: string) : Workflow =
        (workflow "Summarize")
            .Description("Create a multi-level summary of a document")
            .Version("1.0")
            .Input("document", "string", "The document to summarize")
            .Agent("extract", "summarizer",
                "Extract the key points from this document:\n\n" + document)
            .Agent("brief", "summarizer",
                "Create a 2-3 sentence executive summary from:\n{{extract.result}}")
            .Agent("detailed", "summarizer",
                "Create a detailed summary with sections from:\n{{extract.result}}")
            .Build()

    /// <summary>Conversation workflow with memory</summary>
    let conversation (systemPrompt: string) : Workflow =
        (workflow "Conversation")
            .Description("Multi-turn conversation with context")
            .Version("1.0")
            .Input("message", "string", "User message")
            .Retrieval("memory", "{{message}}")
            .Agent("respond", "assistant",
                systemPrompt + "\n\nPrevious context:\n{{memory.context}}\n\nUser: {{message}}")
            .Build()

    /// <summary>Data extraction workflow</summary>
    let extract (schema: string) (text: string) : Workflow =
        (workflow "Extract")
            .Description("Extract structured data from text")
            .Version("1.0")
            .Input("text", "string", "Text to extract from")
            .Input("schema", "string", "Expected output schema")
            .Agent("extract", "extractor",
                $"Extract data matching this schema:\n{schema}\n\nFrom this text:\n{text}\n\nReturn valid JSON only.")
            .Build()

    /// <summary>Translation workflow</summary>
    let translate (targetLang: string) (text: string) : Workflow =
        (workflow "Translate")
            .Description("Translate text to target language")
            .Version("1.0")
            .Input("text", "string", "Text to translate")
            .Input("targetLang", "string", "Target language")
            .Agent("translate", "translator",
                $"Translate the following text to {targetLang}. Preserve formatting and tone:\n\n{text}")
            .Build()

    /// <summary>Comparison workflow</summary>
    let compare (item1: string) (item2: string) : Workflow =
        (workflow "Compare")
            .Description("Compare two items")
            .Version("1.0")
            .Input("item1", "string", "First item")
            .Input("item2", "string", "Second item")
            .Agent("analyze1", "analyst", $"Analyze the key characteristics of:\n{item1}")
            .Agent("analyze2", "analyst", $"Analyze the key characteristics of:\n{item2}")
            .Agent("compare", "analyst",
                "Compare these two analyses:\n\nItem 1:\n{{analyze1.result}}\n\nItem 2:\n{{analyze2.result}}\n\nProvide a structured comparison.")
            .Build()

    /// <summary>Get all available template names</summary>
    let availableTemplates = [
        "ragQA", "Answer questions using RAG"
        "codeReview", "Review code for issues"
        "research", "Research a topic"
        "taskDecomposition", "Break down complex tasks"
        "summarize", "Summarize documents"
        "conversation", "Multi-turn conversation"
        "extract", "Extract structured data"
        "translate", "Translate text"
        "compare", "Compare two items"
    ]

