import * as vscode from 'vscode';

export class TarsStatusBarManager {
    private statusBarItem: vscode.StatusBarItem;
    private progressTimeout?: NodeJS.Timeout;

    constructor() {
        this.statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
        this.statusBarItem.command = 'tars.showStatus';
    }

    initialize() {
        this.statusBarItem.text = '🚀 TARS AI';
        this.statusBarItem.tooltip = 'TARS AI - Click to view status';
        this.statusBarItem.show();
    }

    showProgress(message: string) {
        this.statusBarItem.text = `$(sync~spin) ${message}`;
        this.statusBarItem.tooltip = message;
        
        // Clear any existing timeout
        if (this.progressTimeout) {
            clearTimeout(this.progressTimeout);
        }
        
        // Auto-hide progress after 30 seconds
        this.progressTimeout = setTimeout(() => {
            this.showReady();
        }, 30000);
    }

    showSuccess(message: string) {
        this.statusBarItem.text = `$(check) ${message}`;
        this.statusBarItem.tooltip = message;
        
        // Clear any existing timeout
        if (this.progressTimeout) {
            clearTimeout(this.progressTimeout);
        }
        
        // Return to ready state after 3 seconds
        setTimeout(() => {
            this.showReady();
        }, 3000);
    }

    showError(message: string) {
        this.statusBarItem.text = `$(error) ${message}`;
        this.statusBarItem.tooltip = message;
        
        // Clear any existing timeout
        if (this.progressTimeout) {
            clearTimeout(this.progressTimeout);
        }
        
        // Return to ready state after 5 seconds
        setTimeout(() => {
            this.showReady();
        }, 5000);
    }

    showReady() {
        this.statusBarItem.text = '🚀 TARS AI';
        this.statusBarItem.tooltip = 'TARS AI - Ready for commands';
        
        // Clear any existing timeout
        if (this.progressTimeout) {
            clearTimeout(this.progressTimeout);
            this.progressTimeout = undefined;
        }
    }

    showConnected() {
        this.statusBarItem.text = '🟢 TARS AI';
        this.statusBarItem.tooltip = 'TARS AI - Connected and ready';
    }

    showDisconnected() {
        this.statusBarItem.text = '🔴 TARS AI';
        this.statusBarItem.tooltip = 'TARS AI - Disconnected';
    }

    updateStatus(status: { connected: boolean; gpuEnabled: boolean; activeAgents: number }) {
        let icon = status.connected ? '🟢' : '🔴';
        let text = `${icon} TARS AI`;
        
        if (status.connected) {
            if (status.gpuEnabled) {
                text += ' ⚡';
            }
            if (status.activeAgents > 0) {
                text += ` (${status.activeAgents} agents)`;
            }
        }
        
        this.statusBarItem.text = text;
        this.statusBarItem.tooltip = status.connected 
            ? `TARS AI - Connected | GPU: ${status.gpuEnabled ? 'Enabled' : 'Disabled'} | Agents: ${status.activeAgents}`
            : 'TARS AI - Disconnected';
    }

    dispose() {
        if (this.progressTimeout) {
            clearTimeout(this.progressTimeout);
        }
        this.statusBarItem.dispose();
    }
}
