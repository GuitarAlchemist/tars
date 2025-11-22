// TARS DSL Debugger UI

// Debugger state
let debuggerState = {
    isRunning: false,
    isPaused: false,
    currentFile: '',
    currentLine: 0,
    breakpoints: [],
    watches: [],
    variables: {},
    callStack: []
};

// DOM elements
const elements = {
    fileSelector: document.getElementById('file-selector'),
    startButton: document.getElementById('start-debug'),
    stopButton: document.getElementById('stop-debug'),
    continueButton: document.getElementById('continue'),
    stepNextButton: document.getElementById('step-next'),
    stepIntoButton: document.getElementById('step-into'),
    stepOutButton: document.getElementById('step-out'),
    breakpointsList: document.getElementById('breakpoints-list'),
    addBreakpointButton: document.getElementById('add-breakpoint'),
    breakpointFileInput: document.getElementById('breakpoint-file'),
    breakpointLineInput: document.getElementById('breakpoint-line'),
    watchesList: document.getElementById('watches-list'),
    addWatchButton: document.getElementById('add-watch'),
    watchVariableInput: document.getElementById('watch-variable'),
    variablesList: document.getElementById('variables-list'),
    callStackList: document.getElementById('call-stack-list'),
    codeViewer: document.getElementById('code-viewer'),
    statusBar: document.getElementById('status-bar')
};

// API endpoints
const api = {
    status: '/api/DslDebugger/status',
    start: '/api/DslDebugger/start',
    breakpoints: '/api/DslDebugger/breakpoints',
    clearBreakpoints: '/api/DslDebugger/breakpoints/all',
    watches: '/api/DslDebugger/watches',
    clearWatches: '/api/DslDebugger/watches/all',
    continue: '/api/DslDebugger/continue',
    stepNext: '/api/DslDebugger/step/next',
    stepInto: '/api/DslDebugger/step/into',
    stepOut: '/api/DslDebugger/step/out',
    variables: '/api/DslDebugger/variables/watched',
    callStack: '/api/DslDebugger/callstack'
};

// Initialize the debugger UI
function initDebugger() {
    // Set up event listeners
    elements.startButton.addEventListener('click', startDebugging);
    elements.stopButton.addEventListener('click', stopDebugging);
    elements.continueButton.addEventListener('click', continueExecution);
    elements.stepNextButton.addEventListener('click', stepNext);
    elements.stepIntoButton.addEventListener('click', stepInto);
    elements.stepOutButton.addEventListener('click', stepOut);
    elements.addBreakpointButton.addEventListener('click', addBreakpoint);
    elements.addWatchButton.addEventListener('click', addWatch);
    
    // Disable debug controls initially
    updateUIState();
    
    // Load available DSL files
    loadDslFiles();
    
    // Start polling for debugger status
    setInterval(updateDebuggerStatus, 1000);
}

// Load available DSL files
async function loadDslFiles() {
    try {
        // In a real implementation, this would fetch the list of DSL files from the server
        const files = [
            'examples/metascripts/self_improvement_workflow.tars',
            'examples/metascripts/code_analysis.tars',
            'examples/metascripts/documentation_generator.tars',
            'examples/metascripts/learning_plan_generator.tars',
            'examples/metascripts/multi_agent_collaboration.tars'
        ];
        
        elements.fileSelector.innerHTML = '';
        
        files.forEach(file => {
            const option = document.createElement('option');
            option.value = file;
            option.textContent = file;
            elements.fileSelector.appendChild(option);
        });
    } catch (error) {
        console.error('Error loading DSL files:', error);
        showError('Failed to load DSL files');
    }
}

// Start debugging
async function startDebugging() {
    try {
        const filePath = elements.fileSelector.value;
        
        if (!filePath) {
            showError('Please select a file to debug');
            return;
        }
        
        const response = await fetch(api.start, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                filePath: filePath,
                stepMode: true
            })
        });
        
        if (!response.ok) {
            throw new Error(`Failed to start debugging: ${response.statusText}`);
        }
        
        debuggerState.isRunning = true;
        debuggerState.currentFile = filePath;
        
        updateUIState();
        showSuccess('Debugging started');
        
        // Load the file content
        loadFileContent(filePath);
    } catch (error) {
        console.error('Error starting debugging:', error);
        showError('Failed to start debugging');
    }
}

// Stop debugging
function stopDebugging() {
    debuggerState.isRunning = false;
    debuggerState.isPaused = false;
    updateUIState();
    showSuccess('Debugging stopped');
}

// Continue execution
async function continueExecution() {
    try {
        const response = await fetch(api.continue, {
            method: 'POST'
        });
        
        if (!response.ok) {
            throw new Error(`Failed to continue execution: ${response.statusText}`);
        }
        
        debuggerState.isPaused = false;
        updateUIState();
        showSuccess('Execution continued');
    } catch (error) {
        console.error('Error continuing execution:', error);
        showError('Failed to continue execution');
    }
}

// Step to the next line
async function stepNext() {
    try {
        const response = await fetch(api.stepNext, {
            method: 'POST'
        });
        
        if (!response.ok) {
            throw new Error(`Failed to step to next line: ${response.statusText}`);
        }
        
        showSuccess('Stepped to next line');
        await updateDebuggerStatus();
    } catch (error) {
        console.error('Error stepping to next line:', error);
        showError('Failed to step to next line');
    }
}

// Step into a function
async function stepInto() {
    try {
        const response = await fetch(api.stepInto, {
            method: 'POST'
        });
        
        if (!response.ok) {
            throw new Error(`Failed to step into function: ${response.statusText}`);
        }
        
        showSuccess('Stepped into function');
        await updateDebuggerStatus();
    } catch (error) {
        console.error('Error stepping into function:', error);
        showError('Failed to step into function');
    }
}

// Step out of a function
async function stepOut() {
    try {
        const response = await fetch(api.stepOut, {
            method: 'POST'
        });
        
        if (!response.ok) {
            throw new Error(`Failed to step out of function: ${response.statusText}`);
        }
        
        showSuccess('Stepped out of function');
        await updateDebuggerStatus();
    } catch (error) {
        console.error('Error stepping out of function:', error);
        showError('Failed to step out of function');
    }
}

// Add a breakpoint
async function addBreakpoint() {
    try {
        const file = elements.breakpointFileInput.value;
        const line = parseInt(elements.breakpointLineInput.value);
        
        if (!file || isNaN(line) || line <= 0) {
            showError('Please enter a valid file and line number');
            return;
        }
        
        const response = await fetch(api.breakpoints, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                file: file,
                line: line
            })
        });
        
        if (!response.ok) {
            throw new Error(`Failed to add breakpoint: ${response.statusText}`);
        }
        
        // Add the breakpoint to the list
        debuggerState.breakpoints.push({ file, line });
        updateBreakpointsList();
        
        // Clear the inputs
        elements.breakpointFileInput.value = '';
        elements.breakpointLineInput.value = '';
        
        showSuccess('Breakpoint added');
    } catch (error) {
        console.error('Error adding breakpoint:', error);
        showError('Failed to add breakpoint');
    }
}

// Remove a breakpoint
async function removeBreakpoint(file, line) {
    try {
        const response = await fetch(api.breakpoints, {
            method: 'DELETE',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                file: file,
                line: line
            })
        });
        
        if (!response.ok) {
            throw new Error(`Failed to remove breakpoint: ${response.statusText}`);
        }
        
        // Remove the breakpoint from the list
        debuggerState.breakpoints = debuggerState.breakpoints.filter(bp => 
            bp.file !== file || bp.line !== line);
        updateBreakpointsList();
        
        showSuccess('Breakpoint removed');
    } catch (error) {
        console.error('Error removing breakpoint:', error);
        showError('Failed to remove breakpoint');
    }
}

// Add a watch
async function addWatch() {
    try {
        const variable = elements.watchVariableInput.value;
        
        if (!variable) {
            showError('Please enter a variable name');
            return;
        }
        
        const response = await fetch(api.watches, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                variable: variable
            })
        });
        
        if (!response.ok) {
            throw new Error(`Failed to add watch: ${response.statusText}`);
        }
        
        // Add the watch to the list
        debuggerState.watches.push(variable);
        updateWatchesList();
        
        // Clear the input
        elements.watchVariableInput.value = '';
        
        showSuccess('Watch added');
    } catch (error) {
        console.error('Error adding watch:', error);
        showError('Failed to add watch');
    }
}

// Remove a watch
async function removeWatch(variable) {
    try {
        const response = await fetch(api.watches, {
            method: 'DELETE',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                variable: variable
            })
        });
        
        if (!response.ok) {
            throw new Error(`Failed to remove watch: ${response.statusText}`);
        }
        
        // Remove the watch from the list
        debuggerState.watches = debuggerState.watches.filter(w => w !== variable);
        updateWatchesList();
        
        showSuccess('Watch removed');
    } catch (error) {
        console.error('Error removing watch:', error);
        showError('Failed to remove watch');
    }
}

// Update the debugger status
async function updateDebuggerStatus() {
    if (!debuggerState.isRunning) {
        return;
    }
    
    try {
        const response = await fetch(api.status);
        
        if (!response.ok) {
            throw new Error(`Failed to get debugger status: ${response.statusText}`);
        }
        
        const status = await response.json();
        
        debuggerState.isRunning = status.isRunning;
        debuggerState.isPaused = status.isPaused;
        debuggerState.currentFile = status.currentFile;
        debuggerState.currentLine = status.currentLine;
        debuggerState.breakpoints = status.breakpoints.map(bp => ({ file: bp.item1, line: bp.item2 }));
        debuggerState.watches = status.watches;
        
        updateUIState();
        updateBreakpointsList();
        updateWatchesList();
        
        // If paused, update variables and call stack
        if (debuggerState.isPaused) {
            await updateVariables();
            await updateCallStack();
            highlightCurrentLine();
        }
    } catch (error) {
        console.error('Error updating debugger status:', error);
    }
}

// Update variables
async function updateVariables() {
    try {
        const response = await fetch(api.variables);
        
        if (!response.ok) {
            throw new Error(`Failed to get variables: ${response.statusText}`);
        }
        
        const variables = await response.json();
        
        debuggerState.variables = {};
        variables.forEach(v => {
            debuggerState.variables[v.variable] = v.value;
        });
        
        updateVariablesList();
    } catch (error) {
        console.error('Error updating variables:', error);
    }
}

// Update call stack
async function updateCallStack() {
    try {
        const response = await fetch(api.callStack);
        
        if (!response.ok) {
            throw new Error(`Failed to get call stack: ${response.statusText}`);
        }
        
        debuggerState.callStack = await response.json();
        
        updateCallStackList();
    } catch (error) {
        console.error('Error updating call stack:', error);
    }
}

// Load file content
async function loadFileContent(filePath) {
    try {
        // In a real implementation, this would fetch the file content from the server
        const content = `// Example DSL file content for ${filePath}\n\nDESCRIBE {\n    name: "Example DSL"\n    version: "1.0"\n}\n\nCONFIG {\n    model: "llama3"\n    temperature: 0.7\n}\n\nTARS {\n    PROMPT {\n        text: "Hello, world!"\n    }\n    \n    ACTION {\n        type: "generate"\n    }\n}`;
        
        elements.codeViewer.innerHTML = '';
        
        // Add line numbers and content
        const lines = content.split('\n');
        lines.forEach((line, index) => {
            const lineNumber = index + 1;
            const lineElement = document.createElement('div');
            lineElement.className = 'code-line';
            lineElement.dataset.line = lineNumber;
            lineElement.innerHTML = `<span class="line-number">${lineNumber}</span><span class="line-content">${escapeHtml(line)}</span>`;
            elements.codeViewer.appendChild(lineElement);
            
            // Add click handler for setting breakpoints
            lineElement.addEventListener('click', () => {
                const hasBreakpoint = debuggerState.breakpoints.some(bp => 
                    bp.file === filePath && bp.line === lineNumber);
                
                if (hasBreakpoint) {
                    removeBreakpoint(filePath, lineNumber);
                } else {
                    addBreakpoint(filePath, lineNumber);
                }
            });
        });
        
        // Highlight breakpoints
        highlightBreakpoints();
    } catch (error) {
        console.error('Error loading file content:', error);
        showError('Failed to load file content');
    }
}

// Highlight the current line
function highlightCurrentLine() {
    // Remove previous highlight
    const previousHighlight = elements.codeViewer.querySelector('.current-line');
    if (previousHighlight) {
        previousHighlight.classList.remove('current-line');
    }
    
    // Add highlight to current line
    const currentLine = elements.codeViewer.querySelector(`[data-line="${debuggerState.currentLine}"]`);
    if (currentLine) {
        currentLine.classList.add('current-line');
        currentLine.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
}

// Highlight breakpoints
function highlightBreakpoints() {
    // Remove previous highlights
    const previousHighlights = elements.codeViewer.querySelectorAll('.breakpoint');
    previousHighlights.forEach(el => {
        el.classList.remove('breakpoint');
    });
    
    // Add highlights to breakpoints
    debuggerState.breakpoints.forEach(bp => {
        if (bp.file === debuggerState.currentFile) {
            const lineElement = elements.codeViewer.querySelector(`[data-line="${bp.line}"]`);
            if (lineElement) {
                lineElement.classList.add('breakpoint');
            }
        }
    });
}

// Update the breakpoints list
function updateBreakpointsList() {
    elements.breakpointsList.innerHTML = '';
    
    if (debuggerState.breakpoints.length === 0) {
        const emptyItem = document.createElement('li');
        emptyItem.textContent = 'No breakpoints';
        elements.breakpointsList.appendChild(emptyItem);
        return;
    }
    
    debuggerState.breakpoints.forEach(bp => {
        const item = document.createElement('li');
        item.textContent = `${bp.file}:${bp.line}`;
        
        const removeButton = document.createElement('button');
        removeButton.textContent = 'Remove';
        removeButton.className = 'remove-button';
        removeButton.addEventListener('click', () => removeBreakpoint(bp.file, bp.line));
        
        item.appendChild(removeButton);
        elements.breakpointsList.appendChild(item);
    });
    
    // Highlight breakpoints in the code viewer
    highlightBreakpoints();
}

// Update the watches list
function updateWatchesList() {
    elements.watchesList.innerHTML = '';
    
    if (debuggerState.watches.length === 0) {
        const emptyItem = document.createElement('li');
        emptyItem.textContent = 'No watches';
        elements.watchesList.appendChild(emptyItem);
        return;
    }
    
    debuggerState.watches.forEach(variable => {
        const item = document.createElement('li');
        item.textContent = variable;
        
        const removeButton = document.createElement('button');
        removeButton.textContent = 'Remove';
        removeButton.className = 'remove-button';
        removeButton.addEventListener('click', () => removeWatch(variable));
        
        item.appendChild(removeButton);
        elements.watchesList.appendChild(item);
    });
}

// Update the variables list
function updateVariablesList() {
    elements.variablesList.innerHTML = '';
    
    const variables = Object.entries(debuggerState.variables);
    
    if (variables.length === 0) {
        const emptyItem = document.createElement('li');
        emptyItem.textContent = 'No variables';
        elements.variablesList.appendChild(emptyItem);
        return;
    }
    
    variables.forEach(([name, value]) => {
        const item = document.createElement('li');
        item.innerHTML = `<span class="variable-name">${escapeHtml(name)}</span>: <span class="variable-value">${escapeHtml(value)}</span>`;
        elements.variablesList.appendChild(item);
    });
}

// Update the call stack list
function updateCallStackList() {
    elements.callStackList.innerHTML = '';
    
    if (debuggerState.callStack.length === 0) {
        const emptyItem = document.createElement('li');
        emptyItem.textContent = 'No call stack';
        elements.callStackList.appendChild(emptyItem);
        return;
    }
    
    debuggerState.callStack.forEach(frame => {
        const item = document.createElement('li');
        item.textContent = frame;
        elements.callStackList.appendChild(item);
    });
}

// Update the UI state based on the debugger state
function updateUIState() {
    const isRunning = debuggerState.isRunning;
    const isPaused = debuggerState.isPaused;
    
    elements.fileSelector.disabled = isRunning;
    elements.startButton.disabled = isRunning;
    elements.stopButton.disabled = !isRunning;
    elements.continueButton.disabled = !isRunning || !isPaused;
    elements.stepNextButton.disabled = !isRunning || !isPaused;
    elements.stepIntoButton.disabled = !isRunning || !isPaused;
    elements.stepOutButton.disabled = !isRunning || !isPaused;
    elements.addBreakpointButton.disabled = !isRunning;
    elements.addWatchButton.disabled = !isRunning;
    
    // Update status bar
    if (!isRunning) {
        elements.statusBar.textContent = 'Not debugging';
        elements.statusBar.className = 'status-not-running';
    } else if (isPaused) {
        elements.statusBar.textContent = `Paused at ${debuggerState.currentFile}:${debuggerState.currentLine}`;
        elements.statusBar.className = 'status-paused';
    } else {
        elements.statusBar.textContent = 'Running';
        elements.statusBar.className = 'status-running';
    }
}

// Show a success message
function showSuccess(message) {
    console.log('Success:', message);
    // In a real implementation, this would show a toast or notification
}

// Show an error message
function showError(message) {
    console.error('Error:', message);
    // In a real implementation, this would show a toast or notification
}

// Escape HTML to prevent XSS
function escapeHtml(unsafe) {
    return unsafe
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

// Initialize the debugger when the page loads
document.addEventListener('DOMContentLoaded', initDebugger);
