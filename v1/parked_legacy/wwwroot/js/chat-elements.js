// Custom element for rendering markdown in assistant messages
class AssistantMessage extends HTMLElement {
    static get observedAttributes() {
        return ['markdown'];
    }
    
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
    }
    
    connectedCallback() {
        this.render();
    }
    
    attributeChangedCallback(name, oldValue, newValue) {
        if (name === 'markdown' && oldValue !== newValue) {
            this.render();
        }
    }
    
    render() {
        const markdown = this.getAttribute('markdown') || '';
        // Simple markdown rendering (in a real app, use a proper markdown library)
        let html = markdown
            // Convert citations to empty strings to avoid rendering them
            .replace(/<citation.*?<\/citation>/g, '')
            // Basic markdown conversion
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/\n\n/g, '<br><br>')
            .replace(/\n/g, '<br>');
            
        this.shadowRoot.innerHTML = `
            <style>
                :host {
                    display: block;
                }
                code {
                    background-color: rgba(0, 0, 0, 0.05);
                    padding: 0.1em 0.2em;
                    border-radius: 3px;
                    font-family: monospace;
                }
            </style>
            <div>${html}</div>
        `;
    }
}

customElements.define('assistant-message', AssistantMessage);