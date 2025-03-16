
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

// Enhanced speech synthesis service
window.speechService = {
    speak: function(text, voiceName, rate, pitch) {
        return new Promise((resolve, reject) => {
            try {
                // Cancel any ongoing speech
                window.speechSynthesis.cancel();
                
                const utterance = new SpeechSynthesisUtterance(text);
                
                // Apply voice settings if provided
                if (voiceName) {
                    // Make sure voices are loaded
                    const voices = this.getVoicesSync();
                    const voice = voices.find(v => v.name === voiceName);
                    if (voice) utterance.voice = voice;
                }
                
                // Apply speech parameters with sensible defaults
                utterance.rate = rate || 1.0;
                utterance.pitch = pitch || 1.0;
                utterance.volume = 1.0; // Maximum volume
                
                // Add event handlers
                utterance.onend = () => resolve();
                utterance.onerror = (event) => reject(new Error(`Speech synthesis error: ${event.error}`));
                
                // Start speaking
                window.speechSynthesis.speak(utterance);
                
                // Chrome bug workaround: speech can stop after 15 seconds
                const intervalId = setInterval(() => {
                    if (!window.speechSynthesis.speaking) {
                        clearInterval(intervalId);
                        return;
                    }
                    // Pause and resume to keep it going
                    window.speechSynthesis.pause();
                    window.speechSynthesis.resume();
                }, 14000);
            } catch (error) {
                reject(error);
            }
        });
    },
    
    getVoicesSync: function() {
        return window.speechSynthesis.getVoices();
    },
    
    getVoices: function() {
        return new Promise((resolve) => {
            let voices = window.speechSynthesis.getVoices();
            
            if (voices.length > 0) {
                resolve(voices.map(voice => ({
                    name: voice.name,
                    lang: voice.lang,
                    default: voice.default
                })));
            } else {
                // Wait for voices to be loaded
                window.speechSynthesis.onvoiceschanged = () => {
                    voices = window.speechSynthesis.getVoices();
                    resolve(voices.map(voice => ({
                        name: voice.name,
                        lang: voice.lang,
                        default: voice.default
                    })));
                };
            }
        });
    }
};