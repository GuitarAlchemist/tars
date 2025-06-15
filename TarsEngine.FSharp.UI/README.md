# TARS F# UI - Advanced AI Interface

A modern, comprehensive UI for TARS (Thinking, Autonomous, Reasoning System) built with F#, Fable, Elmish, and advanced web technologies.

## Features

### 🚀 Core Technologies
- **F# with Fable**: Type-safe functional programming compiled to JavaScript
- **Elmish Architecture**: Model-View-Update pattern for predictable state management
- **Material-UI**: Modern, accessible React components
- **Semantic Kernel**: Microsoft's AI orchestration framework for chatbot functionality
- **Monaco Editor**: VS Code editor with Language Server Protocol support
- **WebSocket**: Real-time communication with TARS backend

### 🎯 Key Capabilities

#### Agent Management
- **Hierarchical Agent Tree**: View all TARS agents in a tree structure
- **Real-time Status**: Live updates on agent status and activities
- **Agent Control**: Start, stop, and monitor individual agents
- **Team Coordination**: Visualize agent team interactions

#### Metascript Development
- **Monaco Editor Integration**: Full-featured code editor with syntax highlighting
- **Language Server Protocol**: IntelliSense, error checking, and auto-completion
- **Metascript Browser**: Browse, search, and manage all metascripts
- **Execution Control**: Run, stop, and monitor metascript execution
- **Real-time Feedback**: Live execution status and results

#### Node Monitoring
- **Multi-Node Support**: Monitor multiple TARS nodes
- **Performance Metrics**: CPU, memory, and system health monitoring
- **Running Metascripts**: See what's executing on each node
- **Network Status**: Connection health and heartbeat monitoring

#### AI Chat Interface
- **Semantic Kernel Integration**: Powered by Microsoft's AI framework
- **TARS-Specific Plugins**: Custom functions for agent management, metascript execution
- **Context-Aware**: Chat with full knowledge of system state
- **Multi-Modal**: Support for text, code, and system responses

### 🏗️ Architecture

#### Frontend Stack
```
F# Source Code
    ↓ (Fable Compiler)
JavaScript/React
    ↓ (Webpack)
Optimized Bundle
```

#### Component Hierarchy
```
App.fs (Main Application)
├── Navigation (Sidebar)
├── Dashboard (System Overview)
├── AgentsPage (Agent Management)
├── MetascriptsPage (Metascript Browser)
├── NodesPage (Node Monitoring)
├── ChatPage (AI Interface)
└── MonacoEditor (Code Editor)
```

#### State Management
- **Elmish MVU**: Centralized state with immutable updates
- **Type-Safe Messages**: All state changes through discriminated unions
- **Async Commands**: Non-blocking API calls and WebSocket handling

## Getting Started

### Prerequisites
- .NET 8.0 SDK
- Node.js 18+ and npm
- TARS backend running on localhost:5000

### Installation

1. **Clone and navigate to the UI project**:
   ```bash
   cd TarsEngine.FSharp.UI
   ```

2. **Install .NET dependencies**:
   ```bash
   dotnet restore
   ```

3. **Install npm dependencies**:
   ```bash
   npm install
   ```

4. **Build the F# project**:
   ```bash
   dotnet build
   ```

### Development

1. **Start the development server**:
   ```bash
   npm run dev
   ```

2. **Open your browser**:
   Navigate to `http://localhost:3000`

3. **Hot reload**: Changes to F# files will automatically recompile and reload

### Production Build

1. **Build for production**:
   ```bash
   npm run build
   ```

2. **Serve the built files**:
   The `dist/` folder contains the optimized production build

## Configuration

### API Endpoints
The UI connects to TARS backend APIs:
- **REST API**: `http://localhost:5000/api`
- **WebSocket**: `ws://localhost:5000/ws`
- **Language Server**: `http://localhost:5000/lsp`

### Semantic Kernel
Configure AI models in `Services/SemanticKernelService.fs`:
```fsharp
let config = {
    ModelName = "llama3"
    ApiKey = None  // For local Ollama
    Endpoint = Some "http://localhost:11434/v1"
    MaxTokens = 4000
    Temperature = 0.3
}
```

### Monaco Editor
Language support and themes configured in `public/index.html`:
- **TARS DSL**: Custom language definition
- **F# Support**: Syntax highlighting and IntelliSense
- **Dark/Light Themes**: Automatic theme switching

## Project Structure

```
TarsEngine.FSharp.UI/
├── Types.fs                    # Core type definitions
├── Models/                     # Data models
│   ├── AgentModels.fs
│   ├── MetascriptModels.fs
│   └── NodeModels.fs
├── Services/                   # External service integrations
│   ├── TarsApiService.fs       # TARS REST API client
│   ├── SemanticKernelService.fs # AI chat service
│   └── LanguageServerService.fs # LSP integration
├── Components/                 # Reusable UI components
│   ├── AgentTreeView.fs
│   ├── MetascriptBrowser.fs
│   ├── NodeMonitor.fs
│   ├── ChatInterface.fs
│   └── MonacoEditor.fs
├── Pages/                      # Main application pages
│   ├── Dashboard.fs
│   ├── AgentsPage.fs
│   ├── MetascriptsPage.fs
│   ├── NodesPage.fs
│   └── ChatPage.fs
├── App.fs                      # Main application logic
├── Program.fs                  # Entry point and routing
├── public/                     # Static assets
│   ├── index.html
│   └── style.css
├── package.json                # npm dependencies
├── webpack.config.js           # Build configuration
└── TarsEngine.FSharp.UI.fsproj # F# project file
```

## Advanced Features

### Language Server Protocol
- **Real-time Validation**: Syntax and semantic error checking
- **IntelliSense**: Auto-completion for TARS DSL
- **Go to Definition**: Navigate to metascript definitions
- **Hover Information**: Contextual help and documentation

### WebSocket Integration
- **Real-time Updates**: Live agent status and metascript execution
- **Bidirectional Communication**: Send commands and receive notifications
- **Connection Management**: Automatic reconnection and error handling

### Responsive Design
- **Mobile-First**: Optimized for all screen sizes
- **Material Design**: Consistent, accessible UI components
- **Dark Mode**: Automatic theme detection and switching

## API Integration

### TARS REST API
```fsharp
// Get all agents
let! agents = TarsApiService.Agents.getAgentTree()

// Run a metascript
let! result = TarsApiService.Metascripts.runMetascript(path)

// Get node metrics
let! metrics = TarsApiService.Nodes.getNodeMetrics(nodeId)
```

### Semantic Kernel Chat
```fsharp
// Send message to TARS
let! response = semanticKernelService.SendMessageAsync("Show system status")

// Get available functions
let functions = semanticKernelService.GetAvailableFunctions()
```

## Contributing

1. **Follow F# conventions**: Use functional programming patterns
2. **Type Safety**: Leverage F#'s type system for correctness
3. **Immutable State**: All state changes through Elmish messages
4. **Component Isolation**: Keep components pure and testable
5. **Error Handling**: Use Result types for error management

## Troubleshooting

### Common Issues

1. **Build Errors**: Ensure .NET 8.0 SDK is installed
2. **npm Errors**: Delete `node_modules` and run `npm install`
3. **API Connection**: Verify TARS backend is running on port 5000
4. **WebSocket Issues**: Check firewall and proxy settings

### Debug Mode
Enable detailed logging by setting:
```javascript
window.TARS_CONFIG.debug = true;
```

## License

This project is part of the TARS ecosystem and follows the same licensing terms.

## Related Projects

- **TARS Core**: Main TARS engine and metascript runner
- **TARS Agents**: Agent coordination and team management
- **TARS DSL**: Domain-specific language for metascripts
- **TARS CUDA**: GPU-accelerated vector operations
