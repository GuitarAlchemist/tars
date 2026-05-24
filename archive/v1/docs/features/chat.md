# TARS Chat Bot

TARS includes a powerful chat bot feature that allows you to interact with AI models in a conversational manner. This document provides a detailed overview of the chat bot, its capabilities, and how to use it effectively.

## Overview

The TARS Chat Bot is designed to:

1. **Provide interactive conversations** with AI models
2. **Support multiple models** for different use cases
3. **Enable speech output** for hands-free interaction
4. **Save conversation history** for future reference
5. **Offer WebSocket support** for integration with other applications

## Using the Chat Bot

### Starting a Chat Session

To start a chat session:

```bash
tarscli chat start
```

This will start an interactive chat session with the default model (llama3).

You can specify a different model and enable speech output:

```bash
tarscli chat start --model llama3 --speech
```

### Chat Commands

During a chat session, you can use the following commands:

- `exit` or `quit`: End the chat session
- `examples`: Show example prompts
- `model <name>`: Change the model (e.g., `model llama3`)
- `speech on` or `speech off`: Enable or disable speech output
- `clear`: Start a new conversation

### Example Prompts

To see example prompts:

```bash
tarscli chat examples
```

This will display a list of example prompts that you can use to get started with the chat bot.

### Viewing Chat History

To view your chat history:

```bash
tarscli chat history
```

By default, this will show the 5 most recent chat sessions. You can specify a different number:

```bash
tarscli chat history --count 10
```

## WebSocket Support

TARS Chat Bot includes WebSocket support, allowing you to integrate it with web applications or other services.

### Starting the WebSocket Server

To start the WebSocket server:

```bash
tarscli chat websocket
```

By default, the server will listen on port 8998. You can specify a different port:

```bash
tarscli chat websocket --port 9000
```

### WebSocket API

The WebSocket API supports the following message types:

#### Client to Server

- **Message**: Send a message to the chat bot
  ```json
  {
    "type": "message",
    "content": "Hello, how are you?"
  }
  ```

- **Model**: Change the model
  ```json
  {
    "type": "model",
    "name": "llama3"
  }
  ```

- **Speech**: Enable or disable speech
  ```json
  {
    "type": "speech",
    "enabled": true
  }
  ```

- **New Conversation**: Start a new conversation
  ```json
  {
    "type": "new_conversation"
  }
  ```

#### Server to Client

- **Welcome**: Sent when a client connects
  ```json
  {
    "type": "welcome",
    "message": "Welcome to TARS Chat Bot!",
    "examples": ["Hello, how are you?", "What is the capital of France?", ...]
  }
  ```

- **Response**: Response to a message
  ```json
  {
    "type": "response",
    "content": "I'm doing well, thank you for asking. How can I help you today?"
  }
  ```

- **Typing**: Indicates that the bot is typing
  ```json
  {
    "type": "typing",
    "status": true
  }
  ```

- **Error**: Error message
  ```json
  {
    "type": "error",
    "message": "Invalid request format"
  }
  ```

## Future Enhancements

The TARS Chat Bot is designed to be extensible and will be enhanced with additional features in the future:

1. **Speech-to-Text**: Allow voice input for hands-free interaction
2. **Multi-turn Context**: Improve the bot's ability to maintain context across multiple turns
3. **Persona Customization**: Allow users to customize the bot's persona
4. **Plugin Support**: Enable plugins to extend the bot's capabilities
5. **Integration with Other TARS Features**: Seamless integration with other TARS features like self-improvement and learning

## Technical Details

The TARS Chat Bot is implemented using the following components:

- **ChatBotService**: Handles the core chat functionality, including sending messages, managing conversation history, and speech output
- **ChatWebSocketService**: Provides WebSocket support for integration with web applications
- **OllamaService**: Interfaces with Ollama to generate responses
- **TarsSpeechService**: Provides text-to-speech capabilities

The chat bot uses a REPL (Read-Eval-Print Loop) style interface for interactive sessions, making it easy to use from the command line.

## Troubleshooting

### Chat Bot Not Responding

If the chat bot is not responding, check that:

1. Ollama is running and accessible
2. The model you're trying to use is available in Ollama
3. You have sufficient system resources (memory, CPU) for the model

### WebSocket Connection Issues

If you're having trouble connecting to the WebSocket server, check that:

1. The server is running (`tarscli chat websocket`)
2. You're connecting to the correct URL (`ws://localhost:8998/`)
3. The port is not being used by another application
4. Your firewall is not blocking the connection

### Speech Not Working

If speech output is not working, check that:

1. You have enabled speech (`speech on` or `--speech`)
2. Your system has a working audio output device
3. The TTS service is properly configured
