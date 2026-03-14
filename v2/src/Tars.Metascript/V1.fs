namespace Tars.Metascript

open System
open System.Threading.Tasks

module V1 =

    type MetascriptBlockType =
        | Meta
        | Text
        | Code
        | FSharp
        | CSharp
        | Python
        | JavaScript
        | SQL
        | Markdown
        | HTML
        | CSS
        | JSON
        | XML
        | YAML
        | Command
        | Query
        | Transformation
        | Analysis
        | Reflection
        | Execution
        | Import
        | Export
        | Grammar // New for V2 integration
        | Unknown

    type MetascriptBlockParameter = { Name: string; Value: string }

    type MetascriptBlock =
        { Type: MetascriptBlockType
          Content: string
          LineNumber: int
          ColumnNumber: int
          Parameters: MetascriptBlockParameter list
          Id: string
          ParentId: string option
          Metadata: Map<string, string> }

    type MetascriptVariable =
        { Name: string
          Value: obj
          Type: Type
          Metadata: Map<string, string> }

    type Metascript =
        { Name: string
          Blocks: MetascriptBlock list
          FilePath: string option
          Variables: Map<string, MetascriptVariable>
          Metadata: Map<string, string> }

    type V1MetascriptContext =
        { WorkingDirectory: string
          Variables: Map<string, MetascriptVariable>
          CurrentMetascript: Metascript option
          CurrentBlock: MetascriptBlock option }

    type MetascriptExecutionStatus =
        | Success
        | Failure
        | Partial
        | Pending
        | Running

    type MetascriptBlockExecutionResult =
        { Block: MetascriptBlock
          Status: MetascriptExecutionStatus
          Output: string
          Error: string option
          ReturnValue: obj option
          Variables: Map<string, MetascriptVariable>
          ExecutionTimeMs: float }

    type MetascriptExecutionResult =
        { Metascript: Metascript
          BlockResults: MetascriptBlockExecutionResult list
          Status: MetascriptExecutionStatus
          Output: string
          Error: string option
          ExecutionTimeMs: float
          ReturnValue: obj option
          Variables: Map<string, MetascriptVariable>
          Context: V1MetascriptContext option
          Metadata: Map<string, string> }

    type IBlockHandler =
        abstract member BlockType: MetascriptBlockType
        abstract member Priority: int
        abstract member CanHandle: block: MetascriptBlock -> bool

        abstract member ExecuteBlockAsync:
            block: MetascriptBlock * context: V1MetascriptContext -> Task<MetascriptBlockExecutionResult>

    type IMetascriptExecutor =
        abstract member ExecuteAsync:
            metascript: Metascript * ?context: V1MetascriptContext -> Task<MetascriptExecutionResult>

        abstract member ExecuteBlockAsync:
            block: MetascriptBlock * context: V1MetascriptContext -> Task<MetascriptBlockExecutionResult>
