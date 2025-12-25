namespace Tars.Connectors.Mcp

open System
open System.Text.Json
open System.Text.Json.Serialization

// JSON-RPC 2.0 Types

type JsonRpcRequest =
    { [<JsonPropertyName("jsonrpc")>]
      JsonRpc: string
      [<JsonPropertyName("method")>]
      Method: string
      [<JsonPropertyName("params")>]
      Params: JsonElement option
      [<JsonPropertyName("id")>]
      Id: int option }

type JsonRpcResponse =
    { [<JsonPropertyName("jsonrpc")>]
      JsonRpc: string
      [<JsonPropertyName("result")>]
      Result: JsonElement option
      [<JsonPropertyName("error")>]
      Error: JsonRpcError option
      [<JsonPropertyName("id")>]
      Id: int }

and JsonRpcError =
    { [<JsonPropertyName("code")>]
      Code: int
      [<JsonPropertyName("message")>]
      Message: string
      [<JsonPropertyName("data")>]
      Data: JsonElement option }

// MCP Types

type McpInitializeParams =
    { [<JsonPropertyName("protocolVersion")>]
      ProtocolVersion: string
      [<JsonPropertyName("capabilities")>]
      Capabilities: McpClientCapabilities
      [<JsonPropertyName("clientInfo")>]
      ClientInfo: McpImplementation }

and McpClientCapabilities =
    { [<JsonPropertyName("roots")>]
      Roots: McpRootsCapability option
      [<JsonPropertyName("sampling")>]
      Sampling: obj option }

and McpRootsCapability =
    { [<JsonPropertyName("listChanged")>]
      ListChanged: bool option }

and McpImplementation =
    { [<JsonPropertyName("name")>]
      Name: string
      [<JsonPropertyName("version")>]
      Version: string }

type McpInitializeResult =
    { [<JsonPropertyName("protocolVersion")>]
      ProtocolVersion: string
      [<JsonPropertyName("capabilities")>]
      Capabilities: McpServerCapabilities
      [<JsonPropertyName("serverInfo")>]
      ServerInfo: McpImplementation }

and McpServerCapabilities =
    { [<JsonPropertyName("logging")>]
      Logging: obj option
      [<JsonPropertyName("prompts")>]
      Prompts: obj option
      [<JsonPropertyName("resources")>]
      Resources: obj option
      [<JsonPropertyName("tools")>]
      Tools: obj option }

type McpTool =
    { [<JsonPropertyName("name")>]
      Name: string
      [<JsonPropertyName("description")>]
      Description: string option
      [<JsonPropertyName("inputSchema")>]
      InputSchema: JsonElement }

type McpListToolsResult =
    { [<JsonPropertyName("tools")>]
      Tools: McpTool list
      [<JsonPropertyName("nextCursor")>]
      NextCursor: string option }

type McpCallToolParams =
    { [<JsonPropertyName("name")>]
      Name: string
      [<JsonPropertyName("arguments")>]
      Arguments: Map<string, obj> }

type McpCallToolResult =
    { [<JsonPropertyName("content")>]
      Content: McpContent list
      [<JsonPropertyName("isError")>]
      IsError: bool option
      [<JsonPropertyName("_meta")>]
      Meta: Map<string, obj> option }

and McpContent =
    { [<JsonPropertyName("type")>]
      Type: string // "text" or "image" or "resource"
      [<JsonPropertyName("text")>]
      Text: string option
      [<JsonPropertyName("data")>]
      Data: string option
      [<JsonPropertyName("mimeType")>]
      MimeType: string option
      [<JsonPropertyName("resource")>]
      Resource: obj option }
