import * as vscode from 'vscode';
import { TarsAiService } from '../services/TarsAiService';

export class TarsWebviewProvider implements vscode.WebviewViewProvider {
    public static readonly viewType = 'tarsAiChat';
    private _view?: vscode.WebviewView;

    constructor(
        private readonly _extensionUri: vscode.Uri,
        private readonly _tarsService: TarsAiService
    ) {}

    public resolveWebviewView(
        webviewView: vscode.WebviewView,
        context: vscode.WebviewViewResolveContext,
        _token: vscode.CancellationToken,
    ) {
        this._view = webviewView;

        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this._extensionUri]
        };

        webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);

        webviewView.webview.onDidReceiveMessage(async (data) => {
            switch (data.type) {
                case 'sendMessage':
                    await this.handleChatMessage(data.message);
                    break;
                case 'clearChat':
                    this.clearChat();
                    break;
                case 'getStatus':
                    await this.sendStatus();
                    break;
            }
        });
    }

    private async handleChatMessage(message: string) {
        if (!this._view) return;

        // Add user message to chat
        this._view.webview.postMessage({
            type: 'addMessage',
            message: {
                text: message,
                isUser: true,
                timestamp: new Date().toLocaleTimeString()
            }
        });

        try {
            // Show typing indicator
            this._view.webview.postMessage({
                type: 'showTyping'
            });

            // Get response from TARS AI
            const response = await this._tarsService.chat(message);

            // Hide typing indicator
            this._view.webview.postMessage({
                type: 'hideTyping'
            });

            // Add TARS response to chat
            this._view.webview.postMessage({
                type: 'addMessage',
                message: {
                    text: response,
                    isUser: false,
                    timestamp: new Date().toLocaleTimeString()
                }
            });
        } catch (error) {
            // Hide typing indicator
            this._view.webview.postMessage({
                type: 'hideTyping'
            });

            // Show error message
            this._view.webview.postMessage({
                type: 'addMessage',
                message: {
                    text: `❌ Error: ${error}`,
                    isUser: false,
                    timestamp: new Date().toLocaleTimeString(),
                    isError: true
                }
            });
        }
    }

    private clearChat() {
        if (!this._view) return;
        
        this._view.webview.postMessage({
            type: 'clearMessages'
        });
    }

    private async sendStatus() {
        if (!this._view) return;

        try {
            const status = await this._tarsService.getStatus();
            this._view.webview.postMessage({
                type: 'updateStatus',
                status
            });
        } catch (error) {
            console.error('Failed to get TARS status:', error);
        }
    }

    private _getHtmlForWebview(webview: vscode.Webview) {
        return `<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>TARS AI Chat</title>
            <style>
                body {
                    font-family: var(--vscode-font-family);
                    font-size: var(--vscode-font-size);
                    color: var(--vscode-foreground);
                    background-color: var(--vscode-editor-background);
                    margin: 0;
                    padding: 0;
                    height: 100vh;
                    display: flex;
                    flex-direction: column;
                }

                .header {
                    padding: 10px;
                    background-color: var(--vscode-panel-background);
                    border-bottom: 1px solid var(--vscode-panel-border);
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                }

                .title {
                    font-weight: bold;
                    color: var(--vscode-textLink-foreground);
                }

                .status {
                    font-size: 12px;
                    color: var(--vscode-descriptionForeground);
                }

                .status.online {
                    color: #4caf50;
                }

                .status.offline {
                    color: #f44336;
                }

                .chat-container {
                    flex: 1;
                    overflow-y: auto;
                    padding: 10px;
                    display: flex;
                    flex-direction: column;
                    gap: 10px;
                }

                .message {
                    max-width: 80%;
                    padding: 8px 12px;
                    border-radius: 12px;
                    word-wrap: break-word;
                }

                .message.user {
                    align-self: flex-end;
                    background-color: var(--vscode-button-background);
                    color: var(--vscode-button-foreground);
                }

                .message.tars {
                    align-self: flex-start;
                    background-color: var(--vscode-input-background);
                    border: 1px solid var(--vscode-input-border);
                }

                .message.error {
                    background-color: var(--vscode-inputValidation-errorBackground);
                    border: 1px solid var(--vscode-inputValidation-errorBorder);
                }

                .message-time {
                    font-size: 10px;
                    opacity: 0.7;
                    margin-top: 4px;
                }

                .typing-indicator {
                    align-self: flex-start;
                    padding: 8px 12px;
                    background-color: var(--vscode-input-background);
                    border: 1px solid var(--vscode-input-border);
                    border-radius: 12px;
                    display: none;
                }

                .typing-dots {
                    display: inline-block;
                }

                .typing-dots::after {
                    content: '...';
                    animation: typing 1.5s infinite;
                }

                @keyframes typing {
                    0%, 60% { content: '...'; }
                    20% { content: '.'; }
                    40% { content: '..'; }
                }

                .input-container {
                    padding: 10px;
                    background-color: var(--vscode-panel-background);
                    border-top: 1px solid var(--vscode-panel-border);
                    display: flex;
                    gap: 8px;
                }

                .message-input {
                    flex: 1;
                    padding: 8px;
                    background-color: var(--vscode-input-background);
                    color: var(--vscode-input-foreground);
                    border: 1px solid var(--vscode-input-border);
                    border-radius: 4px;
                    outline: none;
                }

                .message-input:focus {
                    border-color: var(--vscode-focusBorder);
                }

                .send-button, .clear-button {
                    padding: 8px 12px;
                    background-color: var(--vscode-button-background);
                    color: var(--vscode-button-foreground);
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 12px;
                }

                .send-button:hover, .clear-button:hover {
                    background-color: var(--vscode-button-hoverBackground);
                }

                .clear-button {
                    background-color: var(--vscode-button-secondaryBackground);
                    color: var(--vscode-button-secondaryForeground);
                }

                .welcome-message {
                    text-align: center;
                    padding: 20px;
                    color: var(--vscode-descriptionForeground);
                    font-style: italic;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <div class="title">🚀 TARS AI Assistant</div>
                <div class="status" id="status">🔴 Offline</div>
            </div>
            
            <div class="chat-container" id="chatContainer">
                <div class="welcome-message">
                    Welcome to TARS AI! 🤖<br>
                    The world's first self-improving multi-modal AI assistant.<br>
                    Ask me anything about your code!
                </div>
            </div>
            
            <div class="typing-indicator" id="typingIndicator">
                <span class="typing-dots">TARS is thinking</span>
            </div>
            
            <div class="input-container">
                <input type="text" class="message-input" id="messageInput" placeholder="Ask TARS AI anything..." />
                <button class="send-button" id="sendButton">Send</button>
                <button class="clear-button" id="clearButton">Clear</button>
            </div>

            <script>
                const vscode = acquireVsCodeApi();
                const chatContainer = document.getElementById('chatContainer');
                const messageInput = document.getElementById('messageInput');
                const sendButton = document.getElementById('sendButton');
                const clearButton = document.getElementById('clearButton');
                const typingIndicator = document.getElementById('typingIndicator');
                const statusElement = document.getElementById('status');

                // Send message function
                function sendMessage() {
                    const message = messageInput.value.trim();
                    if (message) {
                        vscode.postMessage({
                            type: 'sendMessage',
                            message: message
                        });
                        messageInput.value = '';
                    }
                }

                // Event listeners
                sendButton.addEventListener('click', sendMessage);
                clearButton.addEventListener('click', () => {
                    vscode.postMessage({ type: 'clearChat' });
                });

                messageInput.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter') {
                        sendMessage();
                    }
                });

                // Handle messages from extension
                window.addEventListener('message', event => {
                    const message = event.data;
                    
                    switch (message.type) {
                        case 'addMessage':
                            addMessage(message.message);
                            break;
                        case 'clearMessages':
                            clearMessages();
                            break;
                        case 'showTyping':
                            showTyping();
                            break;
                        case 'hideTyping':
                            hideTyping();
                            break;
                        case 'updateStatus':
                            updateStatus(message.status);
                            break;
                    }
                });

                function addMessage(msg) {
                    const messageDiv = document.createElement('div');
                    messageDiv.className = \`message \${msg.isUser ? 'user' : 'tars'}\${msg.isError ? ' error' : ''}\`;
                    
                    const textDiv = document.createElement('div');
                    textDiv.textContent = msg.text;
                    messageDiv.appendChild(textDiv);
                    
                    const timeDiv = document.createElement('div');
                    timeDiv.className = 'message-time';
                    timeDiv.textContent = msg.timestamp;
                    messageDiv.appendChild(timeDiv);
                    
                    chatContainer.appendChild(messageDiv);
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }

                function clearMessages() {
                    chatContainer.innerHTML = '<div class="welcome-message">Chat cleared. How can I help you?</div>';
                }

                function showTyping() {
                    typingIndicator.style.display = 'block';
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }

                function hideTyping() {
                    typingIndicator.style.display = 'none';
                }

                function updateStatus(status) {
                    if (status.connected) {
                        statusElement.textContent = '🟢 Online';
                        statusElement.className = 'status online';
                    } else {
                        statusElement.textContent = '🔴 Offline';
                        statusElement.className = 'status offline';
                    }
                }

                // Request initial status
                vscode.postMessage({ type: 'getStatus' });
            </script>
        </body>
        </html>`;
    }
}
