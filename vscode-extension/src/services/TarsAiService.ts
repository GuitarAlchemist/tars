import axios, { AxiosInstance } from 'axios';
import * as vscode from 'vscode';

export interface TarsAiResponse {
    success: boolean;
    result?: string;
    error?: string;
    executionTime?: number;
    tokensGenerated?: number;
    modelUsed?: string;
}

export interface TarsStatus {
    connected: boolean;
    gpuEnabled: boolean;
    model: string;
    memoryUsage: string;
    activeAgents: number;
    version: string;
    uptime: string;
}

export class TarsAiService {
    private client!: AxiosInstance;
    private baseUrl!: string;

    constructor() {
        this.updateConfiguration();
        
        // Listen for configuration changes
        vscode.workspace.onDidChangeConfiguration(e => {
            if (e.affectsConfiguration('tars')) {
                this.updateConfiguration();
            }
        });
    }

    private updateConfiguration() {
        const config = vscode.workspace.getConfiguration('tars');
        this.baseUrl = config.get('serverUrl', 'http://localhost:7777');
        
        this.client = axios.create({
            baseURL: this.baseUrl,
            timeout: 30000,
            headers: {
                'Content-Type': 'application/json',
                'User-Agent': 'TARS-VSCode-Extension/1.0.0'
            }
        });
    }

    async generateCode(prompt: string, language: string): Promise<string> {
        try {
            const response = await this.client.post('/api/generate-code', {
                prompt,
                language,
                maxTokens: vscode.workspace.getConfiguration('tars').get('maxTokens', 2000),
                temperature: vscode.workspace.getConfiguration('tars').get('temperature', 0.7)
            });

            if (response.data.success) {
                return response.data.result;
            } else {
                throw new Error(response.data.error || 'Code generation failed');
            }
        } catch (error) {
            if (axios.isAxiosError(error)) {
                throw new Error(`Network error: ${error.message}`);
            }
            throw error;
        }
    }

    async optimizeCode(code: string, language: string): Promise<string> {
        try {
            const response = await this.client.post('/api/optimize-code', {
                code,
                language,
                enableGpu: vscode.workspace.getConfiguration('tars').get('enableGpuAcceleration', true)
            });

            if (response.data.success) {
                return response.data.result;
            } else {
                throw new Error(response.data.error || 'Code optimization failed');
            }
        } catch (error) {
            if (axios.isAxiosError(error)) {
                throw new Error(`Network error: ${error.message}`);
            }
            throw error;
        }
    }

    async explainCode(code: string, language: string): Promise<string> {
        try {
            const response = await this.client.post('/api/explain-code', {
                code,
                language,
                useAdvancedReasoning: true
            });

            if (response.data.success) {
                return response.data.result;
            } else {
                throw new Error(response.data.error || 'Code explanation failed');
            }
        } catch (error) {
            if (axios.isAxiosError(error)) {
                throw new Error(`Network error: ${error.message}`);
            }
            throw error;
        }
    }

    async debugCode(code: string, language: string): Promise<string> {
        try {
            const response = await this.client.post('/api/debug-code', {
                code,
                language,
                useAgentSwarm: true
            });

            if (response.data.success) {
                return response.data.result;
            } else {
                throw new Error(response.data.error || 'Code debugging failed');
            }
        } catch (error) {
            if (axios.isAxiosError(error)) {
                throw new Error(`Network error: ${error.message}`);
            }
            throw error;
        }
    }

    async refactorCode(code: string, refactorType: string, language: string): Promise<string> {
        try {
            const response = await this.client.post('/api/refactor-code', {
                code,
                refactorType,
                language,
                useMultiModalAi: true
            });

            if (response.data.success) {
                return response.data.result;
            } else {
                throw new Error(response.data.error || 'Code refactoring failed');
            }
        } catch (error) {
            if (axios.isAxiosError(error)) {
                throw new Error(`Network error: ${error.message}`);
            }
            throw error;
        }
    }

    async visualCodeAnalysis(code: string, language: string): Promise<any> {
        try {
            const response = await this.client.post('/api/visual-analysis', {
                code,
                language,
                enableVisualization: true
            });

            if (response.data.success) {
                return response.data.result;
            } else {
                throw new Error(response.data.error || 'Visual analysis failed');
            }
        } catch (error) {
            if (axios.isAxiosError(error)) {
                throw new Error(`Network error: ${error.message}`);
            }
            throw error;
        }
    }

    async selfImprove(): Promise<string> {
        try {
            const response = await this.client.post('/api/self-improve', {
                enableLearning: true,
                enableEvolution: true
            });

            if (response.data.success) {
                return response.data.result;
            } else {
                throw new Error(response.data.error || 'Self-improvement failed');
            }
        } catch (error) {
            if (axios.isAxiosError(error)) {
                throw new Error(`Network error: ${error.message}`);
            }
            throw error;
        }
    }

    async getStatus(): Promise<TarsStatus> {
        try {
            const response = await this.client.get('/api/status');

            if (response.data.success) {
                return {
                    connected: true,
                    gpuEnabled: response.data.gpuEnabled || false,
                    model: response.data.model || 'TARS-Advanced-AI',
                    memoryUsage: response.data.memoryUsage || 'N/A',
                    activeAgents: response.data.activeAgents || 0,
                    version: response.data.version || '1.0.0',
                    uptime: response.data.uptime || 'N/A'
                };
            } else {
                throw new Error(response.data.error || 'Status check failed');
            }
        } catch (error) {
            return {
                connected: false,
                gpuEnabled: false,
                model: 'Unknown',
                memoryUsage: 'N/A',
                activeAgents: 0,
                version: 'Unknown',
                uptime: 'N/A'
            };
        }
    }

    async chat(message: string): Promise<string> {
        try {
            const response = await this.client.post('/api/chat', {
                message,
                useAdvancedReasoning: true,
                enableMemory: true,
                maxTokens: vscode.workspace.getConfiguration('tars').get('maxTokens', 2000)
            });

            if (response.data.success) {
                return response.data.result;
            } else {
                throw new Error(response.data.error || 'Chat failed');
            }
        } catch (error) {
            if (axios.isAxiosError(error)) {
                throw new Error(`Network error: ${error.message}`);
            }
            throw error;
        }
    }

    async processVoiceCommand(audioData: ArrayBuffer): Promise<string> {
        try {
            const formData = new FormData();
            formData.append('audio', new Blob([audioData], { type: 'audio/wav' }));

            const response = await this.client.post('/api/voice-command', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            });

            if (response.data.success) {
                return response.data.result;
            } else {
                throw new Error(response.data.error || 'Voice command processing failed');
            }
        } catch (error) {
            if (axios.isAxiosError(error)) {
                throw new Error(`Network error: ${error.message}`);
            }
            throw error;
        }
    }

    async testConnection(): Promise<boolean> {
        try {
            const response = await this.client.get('/api/health');
            return response.status === 200;
        } catch (error) {
            return false;
        }
    }

    async getCapabilities(): Promise<string[]> {
        try {
            const response = await this.client.get('/api/capabilities');
            
            if (response.data.success) {
                return response.data.capabilities || [];
            } else {
                return [];
            }
        } catch (error) {
            return [];
        }
    }

    async executeMetascript(script: string): Promise<TarsAiResponse> {
        try {
            const response = await this.client.post('/api/execute-metascript', {
                script,
                enableGpu: vscode.workspace.getConfiguration('tars').get('enableGpuAcceleration', true),
                enableAgents: true
            });

            return {
                success: response.data.success,
                result: response.data.result,
                error: response.data.error,
                executionTime: response.data.executionTime,
                tokensGenerated: response.data.tokensGenerated,
                modelUsed: response.data.modelUsed
            };
        } catch (error) {
            return {
                success: false,
                error: axios.isAxiosError(error) ? `Network error: ${error.message}` : String(error)
            };
        }
    }
}
