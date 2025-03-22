# Implement MCP Server .NET

**User:** Stephane Pareilleux (spareilleux@gmail.com)  
**Created:** 3/10/2025 21:01  
**Updated:** 3/10/2025 21:03  
**Exported:** 3/22/2025 16:32  

## Prompt:
Can a .net core app implement an MCP server (Model context protocol) - https://www.anthropic.com/news/model-context-protocol

## Response:
​Yes, a .NET Core application can implement a Model Context Protocol (MCP) server. The Model Context Protocol is an open standard that standardizes how applications provide context to large language models (LLMs), enabling secure integration between LLMs and various data sources and tools. ​[Dev Blogs+2github.com+2modelcontextprotocol.io+2](https://github.com/PederHP/mcpdotnet)

To facilitate the development of MCP servers in .NET Core, there are libraries available that provide the necessary tools and abstractions:​

1.  **mcpdotnet**: This is a .NET implementation of the Model Context Protocol, enabling .NET applications to connect to and interact with MCP clients and servers. It supports various MCP capabilities, including tools, resources, prompts, sampling, and roots. The library is compatible with .NET 8.0 and later. ​[github.com](https://github.com/PederHP/mcpdotnet)
    
    _Example usage_:
    
    ```csharp
    var builder = Host.CreateApplicationBuilder(args);
    builder.Services.AddMcpServer()
        .WithStdioServerTransport()
        .WithTools();
    
    var app = builder.Build();
    await app.RunAsync();
    ```
    

In this example, an MCP server is set up with standard input/output transport and tool support.

2.  **ModelContextProtocol.NET**: This project provides a .NET implementation of the MCP, including core protocol implementation, server components, and server hosting integration. It facilitates the creation of MCP servers that can handle communication and request processing. ​[github.com](https://github.com/salty-flower/ModelContextProtocol.NET)
    
    _Example usage_:
    
    ```csharp
    var builder = Host.CreateApplicationBuilder();
    builder.Services.AddMcpServer(serverInfo, mcp => {
        mcp.AddStdioTransport();
        // Additional configuration
    }, keepDefaultLogging: false);
    
    var host = builder.Build();
    await host.RunAsync();
    ```
    

This setup integrates the MCP server into a .NET generic host with standard input/output transport.

Both libraries provide the necessary infrastructure to implement an MCP server in a .NET Core application, allowing seamless integration with LLMs and other MCP-compliant clients.​[modelcontextprotocol.io](https://modelcontextprotocol.io/introduction)



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)