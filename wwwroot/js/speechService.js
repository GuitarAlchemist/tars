window.speechService = {
    speak: function (text) {
        return new Promise((resolve, reject) => {
            if (!('speechSynthesis' in window)) {
                reject('Speech synthesis not supported');
                return;
            }

            const utterance = new SpeechSynthesisUtterance(text);
            utterance.onend = () => resolve();
            utterance.onerror = (event) => reject(event.error);
            window.speechSynthesis.speak(utterance);
        });
    }
};