export interface AudioResponse {
    success: boolean;
    error?: string;
    audioData?: ArrayBuffer;
}

export interface RecordingOptions {
    sampleRate?: number;
    channels?: number;
    format?: 'wav' | 'mp3';
}

export interface AudioPlayer {
    play(audioData: ArrayBuffer): Promise<void>;
    stop(): void;
    isPlaying(): boolean;
}

export interface AudioRecorder {
    start(options?: RecordingOptions): Promise<void>;
    stop(): Promise<Uint8Array>;
    isRecording(): boolean;
}