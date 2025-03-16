
// Global audio objects
let mediaRecorder = null;
let audioChunks = [];
let audioPlayer = null;

window.audioPlayer = {
    init: function() {
        if (!audioPlayer) {
            audioPlayer = new Audio();
        }
        return true;
    },
    
    play: function(audioData) {
        return new Promise((resolve, reject) => {
            try {
                if (!audioPlayer) {
                    audioPlayer = new Audio();
                }
                
                const blob = new Blob([audioData], { type: 'audio/wav' });
                const url = URL.createObjectURL(blob);
                
                audioPlayer.src = url;
                audioPlayer.onended = () => {
                    URL.revokeObjectURL(url);
                    resolve();
                };
                audioPlayer.onerror = (e) => {
                    URL.revokeObjectURL(url);
                    reject(e);
                };
                
                audioPlayer.play();
            } catch (error) {
                console.error("Error playing audio:", error);
                reject(error);
            }
        });
    }
};

// Recording functionality
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

// Add this to your audioHelpers.js file
window.speechService = {
    speak: function(text, voiceName, rate, pitch) {
        return new Promise((resolve, reject) => {
            try {
                const utterance = new SpeechSynthesisUtterance(text);
                
                // Apply voice settings if provided
                if (voiceName) {
                    const voices = window.speechSynthesis.getVoices();
                    const voice = voices.find(v => v.name === voiceName);
                    if (voice) utterance.voice = voice;
                }
                
                if (rate) utterance.rate = rate;
                if (pitch) utterance.pitch = pitch;
                
                utterance.onend = () => resolve();
                utterance.onerror = (event) => reject(new Error(`Speech synthesis error: ${event.error}`));
                window.speechSynthesis.speak(utterance);
            } catch (error) {
                reject(error);
            }
        });
    },
    
    getVoices: function() {
        return window.speechSynthesis.getVoices().map(voice => ({
            name: voice.name,
            lang: voice.lang,
            default: voice.default
        }));
    }
};