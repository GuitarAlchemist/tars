import { AudioResponse, RecordingOptions, AudioPlayer, AudioRecorder } from './types/audio';

class TarsAudioPlayer implements AudioPlayer {
    private audioContext: AudioContext | null = null;
    private currentSource: AudioBufferSourceNode | null = null;

    async play(audioData: ArrayBuffer): Promise<void> {
        if (!audioData || audioData.byteLength === 0) {
            throw new Error("No audio data provided");
        }

        try {
            this.stop();
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            
            console.log("Decoding audio data of size:", audioData.byteLength);
            const audioBuffer = await this.audioContext.decodeAudioData(audioData);
            
            this.currentSource = this.audioContext.createBufferSource();
            this.currentSource.buffer = audioBuffer;
            this.currentSource.connect(this.audioContext.destination);
            
            return new Promise((resolve, reject) => {
                if (!this.currentSource) return reject(new Error("Audio source not initialized"));
                
                this.currentSource.onended = () => {
                    console.log("Audio playback completed");
                    this.cleanup();
                    resolve();
                };
                
                this.currentSource.start(0);
            });
        } catch (error) {
            this.cleanup();
            throw error;
        }
    }

    stop(): void {
        if (this.currentSource) {
            try {
                this.currentSource.stop();
            } catch (e) {
                console.warn("Error stopping audio source:", e);
            }
        }
        this.cleanup();
    }

    isPlaying(): boolean {
        return this.currentSource !== null;
    }

    private cleanup(): void {
        if (this.currentSource) {
            this.currentSource.disconnect();
            this.currentSource = null;
        }
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }
    }
}

class TarsAudioRecorder implements AudioRecorder {
    private mediaRecorder: MediaRecorder | null = null;
    private audioChunks: Blob[] = [];
    private stream: MediaStream | null = null;

    async start(options: RecordingOptions = {}): Promise<void> {
        if (this.isRecording()) {
            throw new Error("Recording is already in progress");
        }

        try {
            this.stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    sampleRate: options.sampleRate,
                    channelCount: options.channels,
                }
            });
            
            this.mediaRecorder = new MediaRecorder(this.stream);
            this.audioChunks = [];

            this.mediaRecorder.ondataavailable = (event: BlobEvent) => {
                this.audioChunks.push(event.data);
            };

            this.mediaRecorder.start();
        } catch (error) {
            this.cleanup();
            throw error;
        }
    }

    async stop(): Promise<Uint8Array> {
        if (!this.isRecording()) {
            throw new Error("No recording in progress");
        }

        return new Promise((resolve, reject) => {
            if (!this.mediaRecorder) return reject(new Error("MediaRecorder not initialized"));

            this.mediaRecorder.onstop = async () => {
                try {
                    const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
                    const arrayBuffer = await audioBlob.arrayBuffer();
                    const uint8Array = new Uint8Array(arrayBuffer);
                    this.cleanup();
                    resolve(uint8Array);
                } catch (error) {
                    reject(error);
                }
            };

            this.mediaRecorder.stop();
        });
    }

    isRecording(): boolean {
        return this.mediaRecorder !== null && this.mediaRecorder.state === 'recording';
    }

    private cleanup(): void {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        this.mediaRecorder = null;
        this.audioChunks = [];
    }
}

// Export singleton instances
export const audioPlayer = new TarsAudioPlayer();
export const audioRecorder = new TarsAudioRecorder();

// Export types for use in other files
export type { AudioResponse, RecordingOptions, AudioPlayer, AudioRecorder };