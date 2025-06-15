import * as vscode from 'vscode';
import { TarsAiService } from './services/TarsAiService';
import { TarsWebviewProvider } from './providers/TarsWebviewProvider';
import { TarsTreeDataProvider } from './providers/TarsTreeDataProvider';
import { TarsStatusBarManager } from './managers/TarsStatusBarManager';
import { TarsVoiceManager } from './managers/TarsVoiceManager';

let tarsService: TarsAiService;
let statusBarManager: TarsStatusBarManager;
let voiceManager: TarsVoiceManager;

export function activate(context: vscode.ExtensionContext) {
    console.log('🚀 TARS AI Extension is now active!');

    // Initialize TARS AI Service
    tarsService = new TarsAiService();
    statusBarManager = new TarsStatusBarManager();
    voiceManager = new TarsVoiceManager();

    // Register Tree Data Provider
    const tarsTreeProvider = new TarsTreeDataProvider();
    vscode.window.registerTreeDataProvider('tarsAiView', tarsTreeProvider);

    // Register Webview Provider
    const webviewProvider = new TarsWebviewProvider(context.extensionUri, tarsService);
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider('tarsAiChat', webviewProvider)
    );

    // Register Commands
    registerCommands(context);

    // Initialize status bar
    statusBarManager.initialize();

    // Show welcome message
    vscode.window.showInformationMessage(
        '🚀 TARS AI is ready! The world\'s first self-improving multi-modal AI is now integrated into VS Code.',
        'Open TARS Chat', 'Show Status'
    ).then(selection => {
        if (selection === 'Open TARS Chat') {
            vscode.commands.executeCommand('tars.openChat');
        } else if (selection === 'Show Status') {
            vscode.commands.executeCommand('tars.showStatus');
        }
    });
}

function registerCommands(context: vscode.ExtensionContext) {
    // Generate Code Command
    const generateCodeCommand = vscode.commands.registerCommand('tars.generateCode', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active editor found');
            return;
        }

        const prompt = await vscode.window.showInputBox({
            prompt: '🚀 Describe what code you want TARS to generate',
            placeHolder: 'e.g., Create a REST API endpoint for user authentication'
        });

        if (!prompt) return;

        try {
            statusBarManager.showProgress('Generating code with TARS AI...');
            const generatedCode = await tarsService.generateCode(prompt, editor.document.languageId);
            
            const position = editor.selection.active;
            await editor.edit(editBuilder => {
                editBuilder.insert(position, generatedCode);
            });

            statusBarManager.showSuccess('Code generated successfully!');
            vscode.window.showInformationMessage('✅ TARS AI generated code successfully!');
        } catch (error) {
            statusBarManager.showError('Code generation failed');
            vscode.window.showErrorMessage(`❌ Code generation failed: ${error}`);
        }
    });

    // Optimize Code Command
    const optimizeCodeCommand = vscode.commands.registerCommand('tars.optimizeCode', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active editor found');
            return;
        }

        const selection = editor.selection;
        const selectedText = editor.document.getText(selection);
        
        if (!selectedText) {
            vscode.window.showErrorMessage('Please select code to optimize');
            return;
        }

        try {
            statusBarManager.showProgress('Optimizing code with GPU acceleration...');
            const optimizedCode = await tarsService.optimizeCode(selectedText, editor.document.languageId);
            
            await editor.edit(editBuilder => {
                editBuilder.replace(selection, optimizedCode);
            });

            statusBarManager.showSuccess('Code optimized successfully!');
            vscode.window.showInformationMessage('⚡ TARS AI optimized your code with GPU acceleration!');
        } catch (error) {
            statusBarManager.showError('Code optimization failed');
            vscode.window.showErrorMessage(`❌ Code optimization failed: ${error}`);
        }
    });

    // Explain Code Command
    const explainCodeCommand = vscode.commands.registerCommand('tars.explainCode', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active editor found');
            return;
        }

        const selection = editor.selection;
        const selectedText = editor.document.getText(selection);
        
        if (!selectedText) {
            vscode.window.showErrorMessage('Please select code to explain');
            return;
        }

        try {
            statusBarManager.showProgress('Analyzing code with advanced reasoning...');
            const explanation = await tarsService.explainCode(selectedText, editor.document.languageId);
            
            // Show explanation in a new document
            const doc = await vscode.workspace.openTextDocument({
                content: `# TARS AI Code Explanation\n\n## Code:\n\`\`\`${editor.document.languageId}\n${selectedText}\n\`\`\`\n\n## Explanation:\n${explanation}`,
                language: 'markdown'
            });
            await vscode.window.showTextDocument(doc);

            statusBarManager.showSuccess('Code explained successfully!');
        } catch (error) {
            statusBarManager.showError('Code explanation failed');
            vscode.window.showErrorMessage(`❌ Code explanation failed: ${error}`);
        }
    });

    // Debug Code Command
    const debugCodeCommand = vscode.commands.registerCommand('tars.debugCode', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active editor found');
            return;
        }

        const selection = editor.selection;
        const selectedText = editor.document.getText(selection);
        
        if (!selectedText) {
            vscode.window.showErrorMessage('Please select code to debug');
            return;
        }

        try {
            statusBarManager.showProgress('Debugging code with AI agents...');
            const debugInfo = await tarsService.debugCode(selectedText, editor.document.languageId);
            
            // Show debug info in output channel
            const outputChannel = vscode.window.createOutputChannel('TARS AI Debug');
            outputChannel.clear();
            outputChannel.appendLine('🔧 TARS AI Debug Analysis');
            outputChannel.appendLine('=========================');
            outputChannel.appendLine(debugInfo);
            outputChannel.show();

            statusBarManager.showSuccess('Code debugging complete!');
        } catch (error) {
            statusBarManager.showError('Code debugging failed');
            vscode.window.showErrorMessage(`❌ Code debugging failed: ${error}`);
        }
    });

    // Refactor Code Command
    const refactorCodeCommand = vscode.commands.registerCommand('tars.refactorCode', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active editor found');
            return;
        }

        const selection = editor.selection;
        const selectedText = editor.document.getText(selection);
        
        if (!selectedText) {
            vscode.window.showErrorMessage('Please select code to refactor');
            return;
        }

        const refactorType = await vscode.window.showQuickPick([
            'Extract Method',
            'Rename Variables',
            'Optimize Performance',
            'Improve Readability',
            'Add Error Handling',
            'Convert to Modern Syntax'
        ], {
            placeHolder: 'Select refactoring type'
        });

        if (!refactorType) return;

        try {
            statusBarManager.showProgress('Refactoring code with multi-modal AI...');
            const refactoredCode = await tarsService.refactorCode(selectedText, refactorType, editor.document.languageId);
            
            await editor.edit(editBuilder => {
                editBuilder.replace(selection, refactoredCode);
            });

            statusBarManager.showSuccess('Code refactored successfully!');
            vscode.window.showInformationMessage(`🔄 TARS AI refactored your code: ${refactorType}`);
        } catch (error) {
            statusBarManager.showError('Code refactoring failed');
            vscode.window.showErrorMessage(`❌ Code refactoring failed: ${error}`);
        }
    });

    // Voice Programming Command
    const voiceProgrammingCommand = vscode.commands.registerCommand('tars.voiceProgramming', async () => {
        const config = vscode.workspace.getConfiguration('tars');
        if (!config.get('enableVoiceProgramming')) {
            vscode.window.showWarningMessage('Voice programming is disabled. Enable it in settings.');
            return;
        }

        try {
            statusBarManager.showProgress('Listening for voice commands...');
            await voiceManager.startListening();
        } catch (error) {
            statusBarManager.showError('Voice programming failed');
            vscode.window.showErrorMessage(`❌ Voice programming failed: ${error}`);
        }
    });

    // Visual Code Analysis Command
    const visualCodeAnalysisCommand = vscode.commands.registerCommand('tars.visualCodeAnalysis', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active editor found');
            return;
        }

        try {
            statusBarManager.showProgress('Performing visual code analysis...');
            const analysis = await tarsService.visualCodeAnalysis(editor.document.getText(), editor.document.languageId);
            
            // Show analysis in webview
            const panel = vscode.window.createWebviewPanel(
                'tarsVisualAnalysis',
                '👁️ TARS Visual Code Analysis',
                vscode.ViewColumn.Beside,
                { enableScripts: true }
            );

            panel.webview.html = getVisualAnalysisHtml(analysis);
            statusBarManager.showSuccess('Visual analysis complete!');
        } catch (error) {
            statusBarManager.showError('Visual analysis failed');
            vscode.window.showErrorMessage(`❌ Visual analysis failed: ${error}`);
        }
    });

    // Self-Improve Command
    const selfImproveCommand = vscode.commands.registerCommand('tars.selfImprove', async () => {
        const config = vscode.workspace.getConfiguration('tars');
        if (!config.get('enableSelfImprovement')) {
            vscode.window.showWarningMessage('Self-improvement is disabled. Enable it in settings.');
            return;
        }

        try {
            statusBarManager.showProgress('TARS AI is self-improving...');
            const improvementResult = await tarsService.selfImprove();
            
            vscode.window.showInformationMessage(
                `🧬 TARS AI self-improvement complete! ${improvementResult}`,
                'View Details'
            ).then(selection => {
                if (selection === 'View Details') {
                    vscode.commands.executeCommand('tars.showStatus');
                }
            });

            statusBarManager.showSuccess('Self-improvement complete!');
        } catch (error) {
            statusBarManager.showError('Self-improvement failed');
            vscode.window.showErrorMessage(`❌ Self-improvement failed: ${error}`);
        }
    });

    // Show Status Command
    const showStatusCommand = vscode.commands.registerCommand('tars.showStatus', async () => {
        try {
            const status = await tarsService.getStatus();
            
            const panel = vscode.window.createWebviewPanel(
                'tarsStatus',
                '📊 TARS AI Status',
                vscode.ViewColumn.One,
                { enableScripts: true }
            );

            panel.webview.html = getStatusHtml(status);
        } catch (error) {
            vscode.window.showErrorMessage(`❌ Failed to get TARS status: ${error}`);
        }
    });

    // Open Chat Command
    const openChatCommand = vscode.commands.registerCommand('tars.openChat', () => {
        vscode.commands.executeCommand('workbench.view.extension.tars-ai');
    });

    // Add all commands to subscriptions
    context.subscriptions.push(
        generateCodeCommand,
        optimizeCodeCommand,
        explainCodeCommand,
        debugCodeCommand,
        refactorCodeCommand,
        voiceProgrammingCommand,
        visualCodeAnalysisCommand,
        selfImproveCommand,
        showStatusCommand,
        openChatCommand
    );
}

function getVisualAnalysisHtml(analysis: any): string {
    return `
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>TARS Visual Code Analysis</title>
            <style>
                body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 20px; background: #1e1e1e; color: #d4d4d4; }
                .header { color: #569cd6; font-size: 24px; margin-bottom: 20px; }
                .metric { margin: 10px 0; padding: 10px; background: #2d2d30; border-radius: 5px; }
                .metric-label { font-weight: bold; color: #4ec9b0; }
                .metric-value { color: #ce9178; }
            </style>
        </head>
        <body>
            <div class="header">👁️ TARS Visual Code Analysis</div>
            <div class="metric">
                <span class="metric-label">Complexity Score:</span>
                <span class="metric-value">${analysis.complexity || 'N/A'}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Quality Score:</span>
                <span class="metric-value">${analysis.quality || 'N/A'}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Performance Score:</span>
                <span class="metric-value">${analysis.performance || 'N/A'}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Recommendations:</span>
                <div class="metric-value">${analysis.recommendations || 'No recommendations available'}</div>
            </div>
        </body>
        </html>
    `;
}

function getStatusHtml(status: any): string {
    return `
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>TARS AI Status</title>
            <style>
                body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 20px; background: #1e1e1e; color: #d4d4d4; }
                .header { color: #569cd6; font-size: 24px; margin-bottom: 20px; }
                .status-item { margin: 10px 0; padding: 10px; background: #2d2d30; border-radius: 5px; }
                .status-label { font-weight: bold; color: #4ec9b0; }
                .status-value { color: #ce9178; }
                .online { color: #4caf50; }
                .offline { color: #f44336; }
            </style>
        </head>
        <body>
            <div class="header">📊 TARS AI Status Dashboard</div>
            <div class="status-item">
                <span class="status-label">Connection Status:</span>
                <span class="status-value ${status.connected ? 'online' : 'offline'}">
                    ${status.connected ? '🟢 Online' : '🔴 Offline'}
                </span>
            </div>
            <div class="status-item">
                <span class="status-label">GPU Acceleration:</span>
                <span class="status-value">${status.gpuEnabled ? '⚡ Enabled' : '❌ Disabled'}</span>
            </div>
            <div class="status-item">
                <span class="status-label">AI Model:</span>
                <span class="status-value">${status.model || 'Unknown'}</span>
            </div>
            <div class="status-item">
                <span class="status-label">Memory Usage:</span>
                <span class="status-value">${status.memoryUsage || 'N/A'}</span>
            </div>
            <div class="status-item">
                <span class="status-label">Active Agents:</span>
                <span class="status-value">${status.activeAgents || 0}</span>
            </div>
        </body>
        </html>
    `;
}

export function deactivate() {
    console.log('🛑 TARS AI Extension is now deactivated');
    statusBarManager?.dispose();
    voiceManager?.dispose();
}
