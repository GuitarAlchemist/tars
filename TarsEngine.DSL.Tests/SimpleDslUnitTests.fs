module TarsEngine.DSL.Tests.SimpleDslUnitTests

open System
open System.IO
open System.Text.Json
open Xunit
open TarsEngine.DSL.SimpleDsl

[<assembly: CollectionBehavior(DisableTestParallelization = true)>]
do ()

let private createTempDirectory prefix =
    let basePath = Path.Combine(".tars", "tests")
    Directory.CreateDirectory(basePath) |> ignore
    let guidSegment = Guid.NewGuid().ToString("N")
    let folderName = $"%s{prefix}_%s{guidSegment}"
    let path = Path.Combine(basePath, folderName)
    Directory.CreateDirectory(path) |> ignore
    Path.GetFullPath(path)

let private deleteDirectoryIfExists path =
    if Directory.Exists(path) then
        Directory.Delete(path, true)

[<Fact>]
let ``Parse simple block`` () =
    let blockText = """DESCRIBE {
    name: "Test Block"
    description: "A test block"
}"""
    
    let block = parseBlock blockText
    
    Assert.Equal(BlockType.Describe, block.Type)
    Assert.Equal(None, block.Name)
    Assert.Equal(2, block.Properties.Count)
    Assert.Equal(StringValue("Test Block"), block.Properties.["name"])
    Assert.Equal(StringValue("A test block"), block.Properties.["description"])
    Assert.Empty(block.NestedBlocks)

[<Fact>]
let ``Parse block with name`` () =
    let blockText = """VARIABLE message {
    value: "Hello, World!"
}"""
    
    let block = parseBlock blockText
    
    Assert.Equal(BlockType.Variable, block.Type)
    Assert.Equal(Some "message", block.Name)
    Assert.Equal(1, block.Properties.Count)
    Assert.Equal(StringValue("Hello, World!"), block.Properties.["value"])
    Assert.Empty(block.NestedBlocks)

[<Fact>]
let ``Parse nested blocks`` () =
    let blockText = """IF {
    condition: "true"
    
    ACTION {
        type: "log"
        message: "This is a test"
    }
}"""
    
    let block = parseBlock blockText
    
    Assert.Equal(BlockType.If, block.Type)
    Assert.Equal(None, block.Name)
    Assert.Equal(1, block.Properties.Count)
    Assert.Equal(StringValue("true"), block.Properties.["condition"])
    Assert.Equal(1, block.NestedBlocks.Length)
    
    let nestedBlock = block.NestedBlocks.[0]
    Assert.Equal(BlockType.Action, nestedBlock.Type)
    Assert.Equal(None, nestedBlock.Name)
    Assert.Equal(2, nestedBlock.Properties.Count)
    Assert.Equal(StringValue("log"), nestedBlock.Properties.["type"])
    Assert.Equal(StringValue("This is a test"), nestedBlock.Properties.["message"])

[<Fact>]
let ``Parse program`` () =
    let programText = """DESCRIBE {
    name: "Test Program"
    description: "A test program"
}

VARIABLE message {
    value: "Hello, World!"
}

ACTION {
    type: "log"
    message: "${message}"
}"""
    
    let program = parseProgram programText
    
    Assert.Equal(3, program.Blocks.Length)
    
    let describeBlock = program.Blocks.[0]
    Assert.Equal(BlockType.Describe, describeBlock.Type)
    
    let variableBlock = program.Blocks.[1]
    Assert.Equal(BlockType.Variable, variableBlock.Type)
    Assert.Equal(Some "message", variableBlock.Name)
    
    let actionBlock = program.Blocks.[2]
    Assert.Equal(BlockType.Action, actionBlock.Type)
    Assert.Equal(StringValue("log"), actionBlock.Properties.["type"])
    Assert.Equal(StringValue("${message}"), actionBlock.Properties.["message"])

[<Fact>]
let ``Execute program with variable substitution`` () =
    let programText = """VARIABLE message {
    value: "Hello, World!"
}

ACTION {
    type: "log"
    message: "${message}"
}"""
    
    let program = parseProgram programText
    let result = executeProgram program
    
    match result with
    | Success value ->
        match value with
        | StringValue message -> Assert.Contains("Hello, World!", message)
        | _ -> Assert.True(false, "Expected StringValue")
    | Error msg -> Assert.True(false, $"Expected Success but got Error: {msg}")

[<Fact>]
let ``Execute program with if condition true`` () =
    let programText = """VARIABLE message {
    value: "Hello, World!"
}

IF {
    condition: "${message == 'Hello, World!'}"
    
    ACTION {
        type: "log"
        message: "Condition is true"
    }
}"""
    
    let program = parseProgram programText
    let result = executeProgram program
    
    match result with
    | Success value ->
        match value with
        | StringValue message -> Assert.Contains("Condition is true", message)
        | _ -> Assert.True(false, "Expected StringValue")
    | Error msg -> Assert.True(false, $"Expected Success but got Error: {msg}")

[<Fact>]
let ``Execute program with if condition false`` () =
    let programText = """VARIABLE message {
    value: "Hello, World!"
}

IF {
    condition: "${message == 'Goodbye, World!'}"
    
    ACTION {
        type: "log"
        message: "Condition is true"
    }
}
ELSE {
    ACTION {
        type: "log"
        message: "Condition is false"
    }
}"""
    
    let program = parseProgram programText
    let result = executeProgram program
    
    match result with
    | Success value ->
        match value with
        | StringValue message -> Assert.Contains("Condition is false", message)
        | _ -> Assert.True(false, "Expected StringValue")
    | Error msg -> Assert.True(false, $"Expected Success but got Error: {msg}")

[<Fact>]
let ``Execute program with mcp_send action`` () =
    let workspace = Directory.GetCurrentDirectory()
    let outboxDir = Path.Combine(workspace, ".tars", "mcp", "outbox")
    Directory.CreateDirectory(outboxDir) |> ignore
    let existing =
        if Directory.Exists(outboxDir) then
            Directory.GetFiles(outboxDir) |> Set.ofArray
        else
            Set.empty

    let programText = """VARIABLE target {
    value: "augment"
}

ACTION {
    type: "mcp_send"
    target: "${target}"
    action: "code_generation"
    result_variable: "response"
}

ACTION {
    type: "log"
    message: "${response}"
}"""
    
    let program = parseProgram programText
    let result = executeProgram program
    try
        match result with
        | Success (StringValue message) ->
            let generatedFiles = Directory.GetFiles(outboxDir) |> Array.filter (fun file -> not (existing.Contains file))
            let filePath = Assert.Single(generatedFiles)
            use doc = JsonDocument.Parse(File.ReadAllText(filePath))
            let root = doc.RootElement
            Assert.Equal("augment", root.GetProperty("target").GetString())
            Assert.Equal("code_generation", root.GetProperty("action").GetString())
            let requestId = root.GetProperty("id").GetString()
            Assert.False(String.IsNullOrWhiteSpace requestId)

            use messageDoc = JsonDocument.Parse(message)
            let messageRoot = messageDoc.RootElement
            Assert.Equal(requestId, messageRoot.GetProperty("requestId").GetString())
            Assert.Equal("augment", messageRoot.GetProperty("target").GetString())
            Assert.Equal("code_generation", messageRoot.GetProperty("action").GetString())
            Assert.Equal(filePath, messageRoot.GetProperty("path").GetString())
        | Success other ->
            Assert.True(false, $"Expected StringValue but received {other}")
        | Error msg ->
            Assert.True(false, $"Expected Success but got Error: {msg}")
    finally
        for file in Directory.GetFiles(outboxDir) do
            if not (existing.Contains file) then
                File.Delete(file)

[<Fact>]
let ``Execute program with mcp_receive action`` () =
    let workspace = Directory.GetCurrentDirectory()
    let inboxDir = Path.Combine(workspace, ".tars", "mcp", "inbox")
    Directory.CreateDirectory(inboxDir) |> ignore
    let testMessageId = Guid.NewGuid().ToString("N")
    let fileName = $"augment_{testMessageId}.json"
    let filePath = Path.Combine(inboxDir, fileName)

    let payload =
        JsonSerializer.Serialize(
            {| id = testMessageId
               source = "augment"
               detail = "integration-test" |},
            JsonSerializerOptions(WriteIndented = true))
    File.WriteAllText(filePath, payload)

    let programText = """ACTION {
    type: "mcp_receive"
    source: "augment"
    timeout: 3
    result_variable: "request"
}

ACTION {
    type: "log"
    message: "${request}"
}"""
    
    let program = parseProgram programText
    let result = executeProgram program
    try
        match result with
        | Success (StringValue message) ->
            Assert.DoesNotContain("timeout", message)
            use messageDoc = JsonDocument.Parse(message)
            let messageRoot = messageDoc.RootElement
            Assert.Equal("augment", messageRoot.GetProperty("source").GetString())
            let payloadProperty = messageRoot.GetProperty("payload")
            Assert.Equal(testMessageId, payloadProperty.GetProperty("id").GetString())
        | Success other ->
            Assert.True(false, $"Expected StringValue but received {other}")
        | Error msg ->
            Assert.True(false, $"Expected Success but got Error: {msg}")
    finally
        if File.Exists(filePath) then
            File.Delete(filePath)

[<Fact>]
let ``Execute program with analyze action`` () =
    let tempDir = createTempDirectory "analyze"
    let filePath = Path.Combine(tempDir, "sample.txt")
    File.WriteAllText(filePath, "first line\nsecond line\n\nthird line")

    let programText = $"""ACTION {{
    type: "analyze"
    target: "{filePath}"
    result_variable: "stats"
}}

ACTION {{
    type: "log"
    message: "${{stats}}"
}}"""

    let program = parseProgram programText
    let result = executeProgram program
    try
        match result with
        | Success (StringValue message) ->
            use doc = JsonDocument.Parse(message)
            let root = doc.RootElement
            Assert.Equal(filePath, root.GetProperty("path").GetString())
            Assert.Equal(3.0, root.GetProperty("nonEmptyLineCount").GetDouble())
            Assert.Equal(4.0, root.GetProperty("lineCount").GetDouble())
        | _ ->
            Assert.True(false, "Expected analyze action to return StringValue")
    finally
        deleteDirectoryIfExists tempDir

[<Fact>]
let ``Execute program with pattern_recognition action`` () =
    let tempDir = createTempDirectory "patterns"
    let filePath = Path.Combine(tempDir, "data.txt")
    File.WriteAllText(filePath, "alpha beta beta gamma gamma gamma")

    let programText = $"""ACTION {{
    type: "pattern_recognition"
    target: "{filePath}"
    result_variable: "patterns"
}}

ACTION {{
    type: "log"
    message: "${{patterns}}"
}}"""

    let program = parseProgram programText
    let result = executeProgram program
    try
        match result with
        | Success (StringValue message) ->
            use doc = JsonDocument.Parse(message)
            let tokens = doc.RootElement.GetProperty("topTokens")
            Assert.True(tokens.GetArrayLength() >= 1)
            let firstToken = tokens.[0]
            Assert.Equal("gamma", firstToken.GetProperty("token").GetString())
            Assert.Equal(3.0, firstToken.GetProperty("count").GetDouble())
        | _ ->
            Assert.True(false, "Expected pattern recognition log output")
    finally
        deleteDirectoryIfExists tempDir

[<Fact>]
let ``Execute program with refactor action`` () =
    let tempDir = createTempDirectory "refactor"
    let filePath = Path.Combine(tempDir, "messy.txt")
    File.WriteAllText(filePath, "value one   \n\nvalue two    \n")

    let programText = $"""ACTION {{
    type: "refactor"
    target: "{filePath}"
    create_backup: true
    result_variable: "refactor_result"
}}

ACTION {{
    type: "log"
    message: "${{refactor_result}}"
}}"""

    let program = parseProgram programText
    let result = executeProgram program

    try
        match result with
        | Success (StringValue message) ->
            Assert.True(File.Exists(filePath + ".bak"))
            let updatedContent = File.ReadAllText(filePath)
            Assert.DoesNotContain("   \n", updatedContent)
            use doc = JsonDocument.Parse(message)
            let root = doc.RootElement
            Assert.Equal(filePath, root.GetProperty("path").GetString())
            Assert.True(root.GetProperty("backupCreated").GetBoolean())
        | _ ->
            Assert.True(false, "Expected refactor log output")
    finally
        deleteDirectoryIfExists tempDir

[<Fact>]
let ``Execute program with get_files action`` () =
    let tempDir = createTempDirectory "getfiles"
    let firstFile = Path.Combine(tempDir, "one.txt")
    let secondFile = Path.Combine(tempDir, "two.md")
    File.WriteAllText(firstFile, "a")
    File.WriteAllText(secondFile, "b")

    let programText = $"""ACTION {{
    type: "get_files"
    directory: "{tempDir}"
    extensions: [".txt", ".md"]
    result_variable: "files"
}}

ACTION {{
    type: "log"
    message: "${{files}}"
}}"""

    let program = parseProgram programText
    let result = executeProgram program
    try
        match result with
        | Success (StringValue message) ->
            use doc = JsonDocument.Parse(message)
            let array = doc.RootElement
            let collected = array.EnumerateArray() |> Seq.map (fun e -> e.GetString()) |> Set.ofSeq
            Assert.Contains(firstFile, collected)
            Assert.Contains(secondFile, collected)
        | _ ->
            Assert.True(false, "Expected get_files log output")
    finally
        deleteDirectoryIfExists tempDir

[<Fact>]
let ``Execute program with generate_report action`` () =
    let tempDir = createTempDirectory "report"
    let reportPath = Path.Combine(tempDir, "summary.md")

    let programText = $"""ACTION {{
    type: "generate_report"
    title: "Integration Report"
    format: "markdown"
    output_file: "{reportPath}"
    content: {{
        total: 5
        status: "ok"
    }}
    result_variable: "report"
}}

ACTION {{
    type: "log"
    message: "${{report}}"
}}"""

    let program = parseProgram programText
    let result = executeProgram program
    try
        match result with
        | Success (StringValue message) ->
            Assert.True(File.Exists(reportPath))
            let reportContent = File.ReadAllText(reportPath)
            Assert.Contains("# Integration Report", reportContent)
            use doc = JsonDocument.Parse(message)
            let root = doc.RootElement
            Assert.Equal(reportPath, root.GetProperty("path").GetString())
            Assert.Equal("markdown", root.GetProperty("format").GetString().ToLowerInvariant())
        | _ ->
            Assert.True(false, "Expected generate_report log output")
    finally
        deleteDirectoryIfExists tempDir
