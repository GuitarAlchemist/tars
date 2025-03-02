# TARS - AI Assistant Platform

TARS is a modern web application built with Blazor and F# that provides an AI assistant interface with self-improvement capabilities and various service integrations.

## 🚀 Features

- **Interactive Chat Interface**: Communicate with TARS using natural language
- **Weather Integration**: Get real-time weather information for any location
- **Self-Improvement Control**: Monitor and control TARS's learning progress
- **Dark/Light Mode**: Customizable UI theme
- **Responsive Design**: Works on desktop and mobile devices

## 🛠️ Technology Stack

- **Frontend**: Blazor WebAssembly with MudBlazor components
- **Backend Services**: F# for core logic and service integration
- **UI Framework**: MudBlazor
- **Local Storage**: Blazored.LocalStorage
- **LLM Integration**: Mistral-7B (configured for local deployment)

## 📋 Prerequisites

- .NET 9.0 SDK (Preview)
- Modern web browser
- (Optional) Visual Studio 2022 Preview, Rider, or VS Code with C#/F# extensions

## 🏗️ Project Structure

```
Tars/
├── Tars/                 # Main Blazor application
├── TarsEngine/           # C# service interfaces
├── TarsEngineFSharp/     # F# core services implementation
└── SolutionItems/        # Shared configuration and documentation
```

## 🚦 Getting Started

1. **Clone the repository**
   ```bash
   git clone [repository-url]
   cd Tars
   ```

2. **Restore dependencies**
   ```bash
   dotnet restore
   ```

3. **Build the solution**
   ```bash
   dotnet build
   ```

4. **Run the application**
   ```bash
   cd Tars
   dotnet run
   ```

5. **Access the application**
   - HTTP: http://localhost:5055
   - HTTPS: https://localhost:7245

## 🔧 Configuration

The application uses default configurations for development. Key services:
- LLM Service: Expected at `http://localhost:8000`
- Weather Service: Uses ip-api.com for location services

## 🧪 Development Status

Current implementation includes:
- ✅ Basic chat interface
- ✅ Weather service integration
- ✅ Self-improvement control panel
- ✅ Dark/light mode theming
- 🚧 LLM service integration (currently mocked)
- 🚧 Extended chat capabilities

## 📦 Dependencies

Main packages:
- MudBlazor (8.3.0)
- Blazored.LocalStorage (4.5.0)
- FSharp.Core (8.0.100)
- Various F# type providers and utilities

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

[Your License Here]

## 🙏 Acknowledgments

- MudBlazor team for the excellent UI components
- Anthropic for LLM capabilities inspiration