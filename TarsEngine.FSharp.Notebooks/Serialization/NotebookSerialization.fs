namespace TarsEngine.FSharp.Notebooks.Serialization

open System
open System.Text.Json
open System.Text.Json.Serialization
open Newtonsoft.Json
open Newtonsoft.Json.Linq
open TarsEngine.FSharp.Notebooks.Types

/// <summary>
/// Jupyter notebook serialization and deserialization following nbformat specification
/// </summary>
module NotebookSerialization =
    
    /// JSON serializer options for notebook serialization
    let private jsonOptions = 
        let options = JsonSerializerOptions()
        options.PropertyNamingPolicy <- JsonNamingPolicy.SnakeCaseLower
        options.WriteIndented <- true
        options.DefaultIgnoreCondition <- JsonIgnoreCondition.WhenWritingNull
        options
    
    /// Serialize notebook to JSON string
    let serializeToJson (notebook: JupyterNotebook) : string =
        try
            // Convert to JSON object structure following nbformat
            let notebookJson = JObject()
            
            // Add nbformat version
            notebookJson.["nbformat"] <- JValue(notebook.NbFormat)
            notebookJson.["nbformat_minor"] <- JValue(notebook.NbFormatMinor)
            
            // Add metadata
            let metadataJson = JObject()
            
            // Kernel spec
            match notebook.Metadata.KernelSpec with
            | Some kernelSpec ->
                let kernelJson = JObject()
                kernelJson.["name"] <- JValue(kernelSpec.Name)
                kernelJson.["display_name"] <- JValue(kernelSpec.DisplayName)
                kernelJson.["language"] <- JValue(kernelSpec.Language)
                
                if kernelSpec.InterruptMode.IsSome then
                    kernelJson.["interrupt_mode"] <- JValue(kernelSpec.InterruptMode.Value)
                
                if not kernelSpec.Env.IsEmpty then
                    let envJson = JObject()
                    for kvp in kernelSpec.Env do
                        envJson.[kvp.Key] <- JValue(kvp.Value)
                    kernelJson.["env"] <- envJson
                
                if not kernelSpec.Argv.IsEmpty then
                    let argvJson = JArray()
                    for arg in kernelSpec.Argv do
                        argvJson.Add(JValue(arg))
                    kernelJson.["argv"] <- argvJson
                
                metadataJson.["kernelspec"] <- kernelJson
            | None -> ()
            
            // Language info
            match notebook.Metadata.LanguageInfo with
            | Some langInfo ->
                let langJson = JObject()
                langJson.["name"] <- JValue(langInfo.Name)
                langJson.["version"] <- JValue(langInfo.Version)
                
                if langInfo.MimeType.IsSome then
                    langJson.["mimetype"] <- JValue(langInfo.MimeType.Value)
                if langInfo.FileExtension.IsSome then
                    langJson.["file_extension"] <- JValue(langInfo.FileExtension.Value)
                if langInfo.PygmentsLexer.IsSome then
                    langJson.["pygments_lexer"] <- JValue(langInfo.PygmentsLexer.Value)
                if langInfo.CodeMirrorMode.IsSome then
                    langJson.["codemirror_mode"] <- JValue(langInfo.CodeMirrorMode.Value)
                
                metadataJson.["language_info"] <- langJson
            | None -> ()
            
            // Custom metadata
            for kvp in notebook.Metadata.Custom do
                metadataJson.[kvp.Key] <- JToken.FromObject(kvp.Value)
            
            notebookJson.["metadata"] <- metadataJson
            
            // Add cells
            let cellsJson = JArray()
            
            for cell in notebook.Cells do
                let cellJson = JObject()
                
                match cell with
                | CodeCell codeData ->
                    cellJson.["cell_type"] <- JValue("code")
                    
                    // Source
                    let sourceArray = JArray()
                    for line in codeData.Source do
                        sourceArray.Add(JValue(line))
                    cellJson.["source"] <- sourceArray
                    
                    // Execution count
                    match codeData.ExecutionCount with
                    | Some count -> cellJson.["execution_count"] <- JValue(count)
                    | None -> cellJson.["execution_count"] <- JValue.CreateNull()
                    
                    // Outputs
                    let outputsArray = JArray()
                    for output in codeData.Outputs do
                        let outputJson = serializeOutput output
                        outputsArray.Add(outputJson)
                    cellJson.["outputs"] <- outputsArray
                    
                    // Metadata
                    let cellMetadata = JObject()
                    for kvp in codeData.Metadata do
                        cellMetadata.[kvp.Key] <- JToken.FromObject(kvp.Value)
                    cellJson.["metadata"] <- cellMetadata
                
                | MarkdownCell markdownData ->
                    cellJson.["cell_type"] <- JValue("markdown")
                    
                    // Source
                    let sourceArray = JArray()
                    for line in markdownData.Source do
                        sourceArray.Add(JValue(line))
                    cellJson.["source"] <- sourceArray
                    
                    // Metadata
                    let cellMetadata = JObject()
                    for kvp in markdownData.Metadata do
                        cellMetadata.[kvp.Key] <- JToken.FromObject(kvp.Value)
                    cellJson.["metadata"] <- cellMetadata
                
                | RawCell rawData ->
                    cellJson.["cell_type"] <- JValue("raw")
                    
                    // Source
                    let sourceArray = JArray()
                    for line in rawData.Source do
                        sourceArray.Add(JValue(line))
                    cellJson.["source"] <- sourceArray
                    
                    // Format
                    if rawData.Format.IsSome then
                        cellJson.["format"] <- JValue(rawData.Format.Value)
                    
                    // Metadata
                    let cellMetadata = JObject()
                    for kvp in rawData.Metadata do
                        cellMetadata.[kvp.Key] <- JToken.FromObject(kvp.Value)
                    cellJson.["metadata"] <- cellMetadata
                
                cellsJson.Add(cellJson)
            
            notebookJson.["cells"] <- cellsJson
            
            // Serialize to string
            notebookJson.ToString(Formatting.Indented)
            
        with
        | ex ->
            failwith $"Failed to serialize notebook: {ex.Message}"
    
    /// Serialize notebook output to JSON
    and private serializeOutput (output: NotebookOutput) : JObject =
        let outputJson = JObject()
        
        match output with
        | DisplayData displayData ->
            outputJson.["output_type"] <- JValue("display_data")
            
            let dataJson = JObject()
            for kvp in displayData.Data do
                dataJson.[kvp.Key] <- JToken.FromObject(kvp.Value)
            outputJson.["data"] <- dataJson
            
            let metadataJson = JObject()
            for kvp in displayData.Metadata do
                metadataJson.[kvp.Key] <- JToken.FromObject(kvp.Value)
            outputJson.["metadata"] <- metadataJson
        
        | ExecuteResult executeResult ->
            outputJson.["output_type"] <- JValue("execute_result")
            outputJson.["execution_count"] <- JValue(executeResult.ExecutionCount)
            
            let dataJson = JObject()
            for kvp in executeResult.Data do
                dataJson.[kvp.Key] <- JToken.FromObject(kvp.Value)
            outputJson.["data"] <- dataJson
            
            let metadataJson = JObject()
            for kvp in executeResult.Metadata do
                metadataJson.[kvp.Key] <- JToken.FromObject(kvp.Value)
            outputJson.["metadata"] <- metadataJson
        
        | StreamOutput streamData ->
            outputJson.["output_type"] <- JValue("stream")
            outputJson.["name"] <- JValue(streamData.Name)
            
            let textArray = JArray()
            for line in streamData.Text do
                textArray.Add(JValue(line))
            outputJson.["text"] <- textArray
        
        | ErrorOutput errorData ->
            outputJson.["output_type"] <- JValue("error")
            outputJson.["ename"] <- JValue(errorData.Name)
            outputJson.["evalue"] <- JValue(errorData.Value)
            
            let tracebackArray = JArray()
            for line in errorData.Traceback do
                tracebackArray.Add(JValue(line))
            outputJson.["traceback"] <- tracebackArray
        
        outputJson
    
    /// Deserialize notebook from JSON string
    let deserializeFromJson (json: string) : Result<JupyterNotebook, string> =
        try
            let notebookJson = JObject.Parse(json)
            
            // Parse nbformat version
            let nbFormat = notebookJson.["nbformat"].Value<int>()
            let nbFormatMinor = notebookJson.["nbformat_minor"].Value<int>()
            
            // Parse metadata
            let metadataJson = notebookJson.["metadata"] :?> JObject
            let metadata = parseMetadata metadataJson
            
            // Parse cells
            let cellsJson = notebookJson.["cells"] :?> JArray
            let cells = 
                cellsJson
                |> Seq.map parseCell
                |> List.ofSeq
            
            let notebook = {
                Metadata = metadata
                Cells = cells
                NbFormat = nbFormat
                NbFormatMinor = nbFormatMinor
            }
            
            Ok notebook
            
        with
        | ex ->
            Error $"Failed to deserialize notebook: {ex.Message}"
    
    /// Parse metadata from JSON
    let private parseMetadata (metadataJson: JObject) : NotebookMetadata =
        let kernelSpec = 
            match metadataJson.["kernelspec"] with
            | null -> None
            | kernelJson ->
                let kernel = kernelJson :?> JObject
                Some {
                    Name = kernel.["name"].Value<string>()
                    DisplayName = kernel.["display_name"].Value<string>()
                    Language = kernel.["language"].Value<string>()
                    InterruptMode = 
                        match kernel.["interrupt_mode"] with
                        | null -> None
                        | value -> Some (value.Value<string>())
                    Env = 
                        match kernel.["env"] with
                        | null -> Map.empty
                        | envJson ->
                            let env = envJson :?> JObject
                            env.Properties()
                            |> Seq.map (fun prop -> prop.Name, prop.Value.Value<string>())
                            |> Map.ofSeq
                    Argv = 
                        match kernel.["argv"] with
                        | null -> []
                        | argvJson ->
                            let argv = argvJson :?> JArray
                            argv
                            |> Seq.map (fun token -> token.Value<string>())
                            |> List.ofSeq
                }
        
        let languageInfo = 
            match metadataJson.["language_info"] with
            | null -> None
            | langJson ->
                let lang = langJson :?> JObject
                Some {
                    Name = lang.["name"].Value<string>()
                    Version = lang.["version"].Value<string>()
                    MimeType = 
                        match lang.["mimetype"] with
                        | null -> None
                        | value -> Some (value.Value<string>())
                    FileExtension = 
                        match lang.["file_extension"] with
                        | null -> None
                        | value -> Some (value.Value<string>())
                    PygmentsLexer = 
                        match lang.["pygments_lexer"] with
                        | null -> None
                        | value -> Some (value.Value<string>())
                    CodeMirrorMode = 
                        match lang.["codemirror_mode"] with
                        | null -> None
                        | value -> Some (value.Value<string>())
                }
        
        // Parse custom metadata
        let customMetadata = 
            metadataJson.Properties()
            |> Seq.filter (fun prop -> prop.Name <> "kernelspec" && prop.Name <> "language_info")
            |> Seq.map (fun prop -> prop.Name, prop.Value.ToObject<obj>())
            |> Map.ofSeq
        
        {
            KernelSpec = kernelSpec
            LanguageInfo = languageInfo
            Title = None
            Authors = []
            Created = None
            Modified = None
            Custom = customMetadata
        }
    
    /// Parse cell from JSON
    let private parseCell (cellJson: JToken) : NotebookCell =
        let cell = cellJson :?> JObject
        let cellType = cell.["cell_type"].Value<string>()
        
        // Parse source
        let source = 
            match cell.["source"] with
            | null -> []
            | sourceJson ->
                let sourceArray = sourceJson :?> JArray
                sourceArray
                |> Seq.map (fun token -> token.Value<string>())
                |> List.ofSeq
        
        // Parse metadata
        let metadata = 
            match cell.["metadata"] with
            | null -> Map.empty
            | metadataJson ->
                let meta = metadataJson :?> JObject
                meta.Properties()
                |> Seq.map (fun prop -> prop.Name, prop.Value.ToObject<obj>())
                |> Map.ofSeq
        
        match cellType with
        | "code" ->
            let executionCount = 
                match cell.["execution_count"] with
                | null -> None
                | value when value.Type = JTokenType.Null -> None
                | value -> Some (value.Value<int>())
            
            let outputs = 
                match cell.["outputs"] with
                | null -> []
                | outputsJson ->
                    let outputsArray = outputsJson :?> JArray
                    outputsArray
                    |> Seq.map parseOutput
                    |> List.ofSeq
            
            CodeCell {
                Source = source
                Language = "python" // Default, should be determined from kernel
                Outputs = outputs
                ExecutionCount = executionCount
                Metadata = metadata
            }
        
        | "markdown" ->
            MarkdownCell {
                Source = source
                Metadata = metadata
            }
        
        | "raw" ->
            let format = 
                match cell.["format"] with
                | null -> None
                | value -> Some (value.Value<string>())
            
            RawCell {
                Source = source
                Format = format
                Metadata = metadata
            }
        
        | _ ->
            // Unknown cell type, treat as raw
            RawCell {
                Source = source
                Format = Some cellType
                Metadata = metadata
            }
    
    /// Parse output from JSON
    let private parseOutput (outputJson: JToken) : NotebookOutput =
        let output = outputJson :?> JObject
        let outputType = output.["output_type"].Value<string>()
        
        match outputType with
        | "display_data" ->
            let data = 
                match output.["data"] with
                | null -> Map.empty
                | dataJson ->
                    let dataObj = dataJson :?> JObject
                    dataObj.Properties()
                    |> Seq.map (fun prop -> prop.Name, prop.Value.ToObject<obj>())
                    |> Map.ofSeq
            
            let metadata = 
                match output.["metadata"] with
                | null -> Map.empty
                | metadataJson ->
                    let metaObj = metadataJson :?> JObject
                    metaObj.Properties()
                    |> Seq.map (fun prop -> prop.Name, prop.Value.ToObject<obj>())
                    |> Map.ofSeq
            
            DisplayData {
                Data = data
                Metadata = metadata
            }
        
        | "execute_result" ->
            let executionCount = output.["execution_count"].Value<int>()
            
            let data = 
                match output.["data"] with
                | null -> Map.empty
                | dataJson ->
                    let dataObj = dataJson :?> JObject
                    dataObj.Properties()
                    |> Seq.map (fun prop -> prop.Name, prop.Value.ToObject<obj>())
                    |> Map.ofSeq
            
            let metadata = 
                match output.["metadata"] with
                | null -> Map.empty
                | metadataJson ->
                    let metaObj = metadataJson :?> JObject
                    metaObj.Properties()
                    |> Seq.map (fun prop -> prop.Name, prop.Value.ToObject<obj>())
                    |> Map.ofSeq
            
            ExecuteResult {
                Data = data
                ExecutionCount = executionCount
                Metadata = metadata
            }
        
        | "stream" ->
            let name = output.["name"].Value<string>()
            let text = 
                match output.["text"] with
                | null -> []
                | textJson ->
                    let textArray = textJson :?> JArray
                    textArray
                    |> Seq.map (fun token -> token.Value<string>())
                    |> List.ofSeq
            
            StreamOutput {
                Name = name
                Text = text
            }
        
        | "error" ->
            let name = output.["ename"].Value<string>()
            let value = output.["evalue"].Value<string>()
            let traceback = 
                match output.["traceback"] with
                | null -> []
                | tracebackJson ->
                    let tracebackArray = tracebackJson :?> JArray
                    tracebackArray
                    |> Seq.map (fun token -> token.Value<string>())
                    |> List.ofSeq
            
            ErrorOutput {
                Name = name
                Value = value
                Traceback = traceback
            }
        
        | _ ->
            // Unknown output type, create empty display data
            DisplayData {
                Data = Map.empty
                Metadata = Map.empty
            }
