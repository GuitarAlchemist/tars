import * as vscode from 'vscode';

export class TarsTreeDataProvider implements vscode.TreeDataProvider<TarsTreeItem> {
    private _onDidChangeTreeData: vscode.EventEmitter<TarsTreeItem | undefined | null | void> = new vscode.EventEmitter<TarsTreeItem | undefined | null | void>();
    readonly onDidChangeTreeData: vscode.Event<TarsTreeItem | undefined | null | void> = this._onDidChangeTreeData.event;

    constructor() {}

    refresh(): void {
        this._onDidChangeTreeData.fire();
    }

    getTreeItem(element: TarsTreeItem): vscode.TreeItem {
        return element;
    }

    getChildren(element?: TarsTreeItem): Thenable<TarsTreeItem[]> {
        if (!element) {
            // Root level items
            return Promise.resolve([
                new TarsTreeItem('🚀 Code Generation', 'Generate code with AI', vscode.TreeItemCollapsibleState.None, 'tars.generateCode'),
                new TarsTreeItem('⚡ GPU Optimization', 'Optimize code with GPU acceleration', vscode.TreeItemCollapsibleState.None, 'tars.optimizeCode'),
                new TarsTreeItem('🧠 Advanced Reasoning', 'Explain code with advanced AI reasoning', vscode.TreeItemCollapsibleState.None, 'tars.explainCode'),
                new TarsTreeItem('🔧 AI Debugging', 'Debug code with AI agents', vscode.TreeItemCollapsibleState.None, 'tars.debugCode'),
                new TarsTreeItem('🔄 Smart Refactoring', 'Refactor code with multi-modal AI', vscode.TreeItemCollapsibleState.None, 'tars.refactorCode'),
                new TarsTreeItem('🎤 Voice Programming', 'Program with voice commands', vscode.TreeItemCollapsibleState.None, 'tars.voiceProgramming'),
                new TarsTreeItem('👁️ Visual Analysis', 'Analyze code visually', vscode.TreeItemCollapsibleState.None, 'tars.visualCodeAnalysis'),
                new TarsTreeItem('🧬 Self-Improvement', 'Improve TARS AI capabilities', vscode.TreeItemCollapsibleState.None, 'tars.selfImprove'),
                new TarsTreeItem('📊 AI Status', 'View TARS AI status and metrics', vscode.TreeItemCollapsibleState.None, 'tars.showStatus'),
                new TarsTreeItem('💬 AI Chat', 'Open TARS AI chat interface', vscode.TreeItemCollapsibleState.None, 'tars.openChat')
            ]);
        }
        
        return Promise.resolve([]);
    }
}

export class TarsTreeItem extends vscode.TreeItem {
    constructor(
        public readonly label: string,
        public readonly tooltip: string,
        public readonly collapsibleState: vscode.TreeItemCollapsibleState,
        public readonly commandName?: string
    ) {
        super(label, collapsibleState);

        this.tooltip = tooltip;
        this.description = '';

        if (commandName) {
            this.command = {
                command: commandName,
                title: label,
                arguments: []
            };
        }

        // Set context value for conditional menu items
        this.contextValue = 'tarsTreeItem';
    }
}
