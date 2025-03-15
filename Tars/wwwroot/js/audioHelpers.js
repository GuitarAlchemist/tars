
class AudioPlayer {
    constructor() {
        this.audioContext = null;
        this.currentSource = null;
    }

    async play(audioData) {
        try {
            // Convert base64 if needed
            let arrayBuffer;
            if (typeof audioData === 'string') {
                // Handle base64 string
                const binaryString = window.atob(audioData);
                const bytes = new Uint8Array(binaryString.length);
                for (let i = 0; i < binaryString.length; i++) {
                    bytes[i] = binaryString.charCodeAt(i);
                }
                arrayBuffer = bytes.buffer;
            } else if (audioData instanceof Uint8Array) {
                arrayBuffer = audioData.buffer;
            } else {
                arrayBuffer = audioData;
            }

            // Initialize audio context if needed
            if (!this.audioContext) {
                this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            }

            // Stop any currently playing audio
            if (this.currentSource) {
                try {
                    this.currentSource.stop();
                    this.currentSource.disconnect();
                } catch (e) {
                    console.warn("Error stopping previous audio:", e);
                }
            }

            console.log("Decoding audio data of size:", arrayBuffer.byteLength);

            // Decode and play
            const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);
            this.currentSource = this.audioContext.createBufferSource();
            this.currentSource.buffer = audioBuffer;
            this.currentSource.connect(this.audioContext.destination);

            return new Promise((resolve) => {
                this.currentSource.onended = () => {
                    console.log("Audio playback completed");
                    resolve();
                };
                this.currentSource.start(0);
                console.log("Audio playback started");
            });
        } catch (error) {
            console.error("Error playing audio:", error);
            throw error;
        }
    }

    stop() {
        if (this.currentSource) {
            try {
                this.currentSource.stop();
                this.currentSource.disconnect();
            } catch (e) {
                console.warn("Error stopping audio:", e);
            }
        }
    }
}

// Create and export a global instance
window.audioPlayer = new AudioPlayer();

// Recording functionality
let mediaRecorder = null;
let audioChunks = [];

window.audioRecorder = {
    async start(options = { sampleRate: 16000, channels: 1 }) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    sampleRate: options.sampleRate,
                    channelCount: options.channels,
                    echoCancellation: true,
                    noiseSuppression: true
                }
            });
            
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];

            mediaRecorder.ondataavailable = (event) => {
                audioChunks.push(event.data);
            };

            mediaRecorder.start();
            console.log("Recording started");
        } catch (error) {
            console.error("Error starting recording:", error);
            throw error;
        }
    },

    async stop() {
        return new Promise((resolve, reject) => {
            try {
                mediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    const arrayBuffer = await audioBlob.arrayBuffer();
                    const uint8Array = new Uint8Array(arrayBuffer);
                    resolve(Array.from(uint8Array));
                };

                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
                console.log("Recording stopped");
            } catch (error) {
                console.error("Error stopping recording:", error);
                reject(error);
            }
        });
    }
};