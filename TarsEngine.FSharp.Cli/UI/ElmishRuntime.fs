namespace TarsEngine.FSharp.Cli.UI

open System
open System.Text.Json
open TarsEngine.FSharp.Cli.UI.TarsElmishDiagnostics

/// ELMISH RUNTIME - Complete HTML + JavaScript + CSS Generation
module ElmishRuntime =

    // JAVASCRIPT RUNTIME FOR ELMISH
    let generateJavaScriptRuntime () =
        """
        <script>
        // ELMISH RUNTIME - Message Dispatch and State Management
        let currentModel = null;
        let autoRefreshTimer = null;

        // Initialize the Elmish application
        function initElmish(initialModel) {
            currentModel = JSON.parse(initialModel);
            console.log('🧠 TARS Elmish Runtime Initialized', currentModel);
            
            // Start auto-refresh if enabled
            if (currentModel.AutoRefresh) {
                startAutoRefresh();
            }
            
            // Add keyboard shortcuts
            document.addEventListener('keydown', handleKeyboard);
        }

        // Message dispatch function
        function dispatch(messageJson) {
            try {
                const message = JSON.parse(messageJson);
                console.log('📨 Dispatching message:', message);
                
                // Send message to F# backend via fetch
                fetch('/tars/elmish/update', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        message: message,
                        currentModel: currentModel
                    })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        updateModel(data.newModel);
                        updateView(data.newHtml);
                    } else {
                        console.error('❌ TARS Update Error:', data.error);
                    }
                })
                .catch(error => {
                    console.error('❌ TARS Communication Error:', error);
                });
                
            } catch (error) {
                console.error('❌ Message Dispatch Error:', error);
            }
        }

        // Update model state
        function updateModel(newModel) {
            const oldAutoRefresh = currentModel ? currentModel.AutoRefresh : false;
            currentModel = newModel;
            
            // Handle auto-refresh state changes
            if (currentModel.AutoRefresh && !oldAutoRefresh) {
                startAutoRefresh();
            } else if (!currentModel.AutoRefresh && oldAutoRefresh) {
                stopAutoRefresh();
            }
            
            console.log('🔄 Model updated:', currentModel);
        }

        // Update view with new HTML
        function updateView(newHtml) {
            const appContainer = document.getElementById('elmish-tars-root');
            if (appContainer) {
                appContainer.innerHTML = newHtml;
                
                // Add smooth transitions
                appContainer.style.opacity = '0';
                setTimeout(() => {
                    appContainer.style.opacity = '1';
                }, 50);
                
                console.log('🎨 View updated');
            }
        }

        // Auto-refresh functionality
        function startAutoRefresh() {
            if (autoRefreshTimer) {
                clearInterval(autoRefreshTimer);
            }
            
            autoRefreshTimer = setInterval(() => {
                if (currentModel && currentModel.AutoRefresh) {
                    dispatch(JSON.stringify('EvolutionTick'));
                }
            }, 5000); // 5 second intervals
            
            console.log('⚡ Auto-refresh started');
        }

        function stopAutoRefresh() {
            if (autoRefreshTimer) {
                clearInterval(autoRefreshTimer);
                autoRefreshTimer = null;
            }
            console.log('⏹️ Auto-refresh stopped');
        }

        // Keyboard shortcuts
        function handleKeyboard(event) {
            if (event.ctrlKey || event.metaKey) {
                switch (event.key) {
                    case 'r':
                        event.preventDefault();
                        dispatch(JSON.stringify('RefreshAll'));
                        break;
                    case 'e':
                        event.preventDefault();
                        dispatch(JSON.stringify('Evolve'));
                        break;
                    case 'm':
                        event.preventDefault();
                        dispatch(JSON.stringify('SelfModify'));
                        break;
                    case 'c':
                        event.preventDefault();
                        dispatch(JSON.stringify('BoostConsciousness'));
                        break;
                    case 'q':
                        event.preventDefault();
                        dispatch(JSON.stringify('QuantumTunnel'));
                        break;
                }
            }
            
            // Number keys for view modes
            if (!event.ctrlKey && !event.metaKey && !event.altKey) {
                switch (event.key) {
                    case '1':
                        dispatch(JSON.stringify({ "Case": "ChangeViewMode", "Fields": ["Overview"] }));
                        break;
                    case '2':
                        dispatch(JSON.stringify({ "Case": "ChangeViewMode", "Fields": ["Architecture"] }));
                        break;
                    case '3':
                        dispatch(JSON.stringify({ "Case": "ChangeViewMode", "Fields": ["Performance"] }));
                        break;
                    case '4':
                        dispatch(JSON.stringify({ "Case": "ChangeViewMode", "Fields": ["Consciousness"] }));
                        break;
                    case '5':
                        dispatch(JSON.stringify({ "Case": "ChangeViewMode", "Fields": ["Evolution"] }));
                        break;
                    case '6':
                        dispatch(JSON.stringify({ "Case": "ChangeViewMode", "Fields": ["Dreams"] }));
                        break;
                    case '7':
                        dispatch(JSON.stringify({ "Case": "ChangeViewMode", "Fields": ["Quantum"] }));
                        break;
                }
            }
        }

        // Utility functions
        function formatTime(timestamp) {
            return new Date(timestamp).toLocaleTimeString();
        }

        function animateMetric(elementId, newValue) {
            const element = document.getElementById(elementId);
            if (element) {
                element.style.transform = 'scale(1.1)';
                setTimeout(() => {
                    element.style.transform = 'scale(1)';
                }, 200);
            }
        }

        // Dark mode functionality
        let isDarkMode = true; // Default to dark mode

        function toggleDarkMode() {
            isDarkMode = !isDarkMode;
            document.body.classList.toggle('dark-mode', isDarkMode);
            localStorage.setItem('tars-dark-mode', isDarkMode);
            console.log('🌙 Dark mode:', isDarkMode ? 'ON' : 'OFF');
        }

        // Subsystem selection functionality
        let selectedSubsystem = null;

        function selectSubsystem(subsystemName) {
            selectedSubsystem = subsystemName;

            // Remove previous selection
            document.querySelectorAll('.subsystem-card').forEach(card => {
                card.classList.remove('selected');
            });

            // Add selection to clicked card
            const clickedCard = document.querySelector(`.subsystem-card.${subsystemName.toLowerCase()}`);
            if (clickedCard) {
                clickedCard.classList.add('selected');
                showSubsystemDetail(subsystemName);
            }

            console.log('🔍 Selected subsystem:', subsystemName);
        }

        function showSubsystemDetail(subsystemName) {
            // Create detailed view (this would be enhanced with real data)
            const detailView = document.createElement('div');
            detailView.className = 'subsystem-detail-view';
            detailView.innerHTML = `
                <div class="detail-header">
                    <h2 class="detail-title">🧠 ${subsystemName}</h2>
                    <div class="detail-status">
                        <span style="color: #28a745">✅ Operational</span>
                        <button class="close-detail-btn" onclick="closeSubsystemDetail()">✕ Close</button>
                    </div>
                </div>
                <div class="detail-metrics-grid">
                    <div class="detail-metric-card">
                        <div class="detail-metric-label">Health Percentage</div>
                        <div class="detail-metric-value">94.2%</div>
                    </div>
                    <div class="detail-metric-card">
                        <div class="detail-metric-label">Active Components</div>
                        <div class="detail-metric-value">47</div>
                    </div>
                    <div class="detail-metric-card">
                        <div class="detail-metric-label">Processing Rate</div>
                        <div class="detail-metric-value">1247.3/s</div>
                    </div>
                    <div class="detail-metric-card">
                        <div class="detail-metric-label">Memory Usage</div>
                        <div class="detail-metric-value">3.2 GB</div>
                    </div>
                </div>
                <h3 class="dependencies-title">🔗 Dependencies</h3>
                <div class="dependencies-section">
                    <span class="dependency-tag">BeliefBus</span>
                    <span class="dependency-tag">VectorStore</span>
                    <span class="dependency-tag">NeuralFabric</span>
                    <span class="dependency-tag">ConsciousnessCore</span>
                </div>
            `;

            // Remove existing detail view
            const existingDetail = document.querySelector('.subsystem-detail-view');
            if (existingDetail) {
                existingDetail.remove();
            }

            // Add new detail view
            const mainContent = document.querySelector('.tars-main-content');
            if (mainContent) {
                mainContent.insertBefore(detailView, mainContent.firstChild);
            }
        }

        function closeSubsystemDetail() {
            const detailView = document.querySelector('.subsystem-detail-view');
            if (detailView) {
                detailView.remove();
            }

            // Remove selection
            document.querySelectorAll('.subsystem-card').forEach(card => {
                card.classList.remove('selected');
            });

            selectedSubsystem = null;
            console.log('❌ Closed subsystem detail');
        }

        // REAL ELMISH BREADCRUMB NAVIGATION FUNCTIONS
        function navigateToHome() {
            console.log('🏠 Navigating to TARS Diagnostics home');
            dispatch(JSON.stringify({ "Case": "NavigateToHome", "Fields": [] }));
        }

        function navigateToView(viewMode) {
            console.log('📊 Navigating to view:', viewMode);
            dispatch(JSON.stringify({ "Case": "NavigateToView", "Fields": [viewMode] }));
        }

        function clearSubsystemSelection() {
            console.log('🔄 Clearing subsystem selection');
            dispatch(JSON.stringify({ "Case": "ClearSubsystemSelection", "Fields": [] }));
        }

        // Make functions globally available for onclick handlers
        window.navigateToHome = navigateToHome;
        window.navigateToView = navigateToView;
        window.clearSubsystemSelection = clearSubsystemSelection;

        // Initialize when DOM is ready
        document.addEventListener('DOMContentLoaded', function() {
            console.log('🚀 TARS Elmish Runtime Ready');

            // Apply dark mode from localStorage
            const savedDarkMode = localStorage.getItem('tars-dark-mode');
            if (savedDarkMode !== null) {
                isDarkMode = savedDarkMode === 'true';
            }
            document.body.classList.toggle('dark-mode', isDarkMode);

            // Add dark mode toggle button
            const header = document.querySelector('.tars-header');
            if (header) {
                const darkModeBtn = document.createElement('button');
                darkModeBtn.className = 'btn dark-mode-toggle';
                darkModeBtn.innerHTML = isDarkMode ? '☀️ Light Mode' : '🌙 Dark Mode';
                darkModeBtn.onclick = () => {
                    toggleDarkMode();
                    darkModeBtn.innerHTML = isDarkMode ? '☀️ Light Mode' : '🌙 Dark Mode';
                };
                header.appendChild(darkModeBtn);
            }
        });
        </script>
        """

    // CSS STYLING FOR TARS THEME
    let generateTarsCSS () =
        """
        <style>
        /* TARS ELMISH STYLING - Enhanced Dark Mode */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
            color: #e0e0e0;
            min-height: 100vh;
            overflow-x: hidden;
        }

        /* Dark mode enhancements */
        .dark-mode {
            background: linear-gradient(135deg, #000000 0%, #0d1117 50%, #161b22 100%);
        }

        .dark-mode .tars-sidebar {
            background: rgba(0, 0, 0, 0.8);
            border: 1px solid rgba(0, 255, 136, 0.3);
        }

        .dark-mode .subsystem-card {
            background: rgba(0, 0, 0, 0.6);
            border: 1px solid rgba(0, 255, 136, 0.2);
        }

        .dark-mode .tars-header {
            background: rgba(0, 0, 0, 0.7);
            border: 2px solid rgba(0, 255, 136, 0.5);
        }

        .functional-elmish-tars-app {
            min-height: 100vh;
            padding: 20px;
            position: relative;
        }

        /* HEADER STYLING */
        .tars-header {
            text-align: center;
            margin-bottom: 30px;
            padding: 40px;
            background: rgba(0, 0, 0, 0.4);
            border-radius: 20px;
            backdrop-filter: blur(15px);
            border: 2px solid rgba(0, 255, 136, 0.4);
            box-shadow: 0 10px 40px rgba(0, 255, 136, 0.1);
        }

        .tars-header h1 {
            font-size: 2.5rem;
            margin-bottom: 20px;
            background: linear-gradient(45deg, #00ff88, #17a2b8, #ff6b6b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .tars-metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }

        .metric-card {
            background: rgba(255, 255, 255, 0.08);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            transition: all 0.3s ease;
            border-left: 4px solid #00ff88;
        }

        .metric-card:hover {
            transform: translateY(-5px);
            background: rgba(255, 255, 255, 0.12);
        }

        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            display: block;
            margin-bottom: 5px;
        }

        .metric-label {
            font-size: 0.9rem;
            opacity: 0.8;
        }

        .status-indicators {
            display: flex;
            justify-content: center;
            gap: 20px;
            flex-wrap: wrap;
        }

        .status-item {
            display: flex;
            align-items: center;
            gap: 5px;
        }

        .status-label {
            opacity: 0.8;
        }

        .status-value {
            font-weight: bold;
        }

        .alert-count {
            background: #dc3545;
            color: white;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.8rem;
        }

        /* VIEW MODE CONTROLS */
        .view-mode-controls {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }

        .view-mode-btn {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: #e0e0e0;
            padding: 10px 20px;
            border-radius: 25px;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .view-mode-btn:hover {
            background: rgba(255, 255, 255, 0.2);
            transform: translateY(-2px);
        }

        .view-mode-btn.active {
            background: linear-gradient(45deg, #00ff88, #17a2b8);
            color: #000;
            font-weight: bold;
        }

        .btn-icon {
            font-size: 1.2rem;
        }

        /* CONTROL PANEL */
        .control-panel {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 15px;
            margin-bottom: 30px;
            flex-wrap: wrap;
            padding: 15px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
        }

        .control-btn {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: #e0e0e0;
            padding: 8px 16px;
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .control-btn:hover {
            background: rgba(255, 255, 255, 0.2);
            transform: scale(1.05);
        }

        .control-btn.active {
            background: #00ff88;
            color: #000;
        }

        .control-btn.refresh:hover { background: #17a2b8; }
        .control-btn.evolve:hover { background: #ff6b6b; }
        .control-btn.self-modify:hover { background: #ffc107; }
        .control-btn.consciousness:hover { background: #e91e63; }
        .control-btn.quantum:hover { background: #9c27b0; }

        .last-update {
            display: flex;
            align-items: center;
            gap: 5px;
            font-size: 0.9rem;
            opacity: 0.8;
        }

        /* SUBSYSTEMS GRID */
        .subsystems-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
            gap: 25px;
            padding: 20px 0;
        }

        .subsystem-card {
            background: rgba(255, 255, 255, 0.08);
            border-radius: 15px;
            padding: 20px;
            cursor: pointer;
            transition: all 0.3s ease;
            border-left: 4px solid #00ff88;
            position: relative;
            overflow: hidden;
        }

        .subsystem-card:hover {
            transform: translateY(-5px);
            background: rgba(255, 255, 255, 0.12);
            box-shadow: 0 10px 30px rgba(0, 255, 136, 0.2);
            cursor: pointer;
        }

        .subsystem-card.selected {
            border-left: 4px solid #00ff88;
            background: rgba(0, 255, 136, 0.15);
            box-shadow: 0 0 20px rgba(0, 255, 136, 0.3);
        }

        .subsystem-card.clickable {
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .subsystem-card.clickable:hover {
            border-left: 4px solid #00ccff;
            background: rgba(0, 204, 255, 0.1);
        }

        .subsystem-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }

        .subsystem-name {
            font-size: 1.3rem;
            margin: 0;
        }

        .status-indicator {
            font-weight: bold;
            display: flex;
            align-items: center;
            gap: 5px;
        }

        .subsystem-description {
            font-size: 0.9rem;
            opacity: 0.8;
            margin-bottom: 15px;
            font-style: italic;
        }

        .subsystem-metrics {
            display: grid;
            gap: 8px;
        }

        .metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .metric-label {
            opacity: 0.8;
        }

        .metric-bar {
            position: relative;
            background: rgba(255, 255, 255, 0.1);
            height: 6px;
            border-radius: 3px;
            overflow: hidden;
            flex: 1;
            margin: 0 10px;
        }

        .metric-fill {
            height: 100%;
            transition: width 0.5s ease;
        }

        .metric-value {
            font-weight: bold;
            min-width: 60px;
            text-align: right;
        }

        /* SUBSYSTEM DETAILS */
        .subsystem-details {
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid rgba(255, 255, 255, 0.2);
        }

        .subsystem-detail-view {
            background: rgba(0, 0, 0, 0.4);
            backdrop-filter: blur(15px);
            border-radius: 15px;
            padding: 30px;
            border: 2px solid rgba(0, 255, 136, 0.4);
            margin-bottom: 20px;
        }

        .detail-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 25px;
            padding-bottom: 15px;
            border-bottom: 2px solid rgba(0, 255, 136, 0.3);
        }

        .detail-title {
            font-size: 2rem;
            background: linear-gradient(45deg, #00ff88, #00ccff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .detail-status {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 1.2rem;
        }

        .detail-metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 25px;
        }

        .detail-metric-card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 20px;
            border-left: 4px solid #00ff88;
        }

        .detail-metric-label {
            font-size: 0.9rem;
            opacity: 0.8;
            margin-bottom: 5px;
        }

        .detail-metric-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #00ff88;
        }

        .close-detail-btn {
            background: rgba(255, 75, 75, 0.8);
            border: none;
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .close-detail-btn:hover {
            background: rgba(255, 75, 75, 1);
            transform: scale(1.05);
        }

        .dependencies-title {
            font-size: 1.3rem;
            margin-bottom: 15px;
            color: #00ccff;
        }

        .dependencies-section {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }

        .dependency-tag {
            background: rgba(0, 204, 255, 0.2);
            color: #00ccff;
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 0.9rem;
            border: 1px solid rgba(0, 204, 255, 0.4);
        }

        .dark-mode-toggle {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: #e0e0e0;
            padding: 8px 16px;
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-left: 15px;
        }

        .dark-mode-toggle:hover {
            background: rgba(255, 255, 255, 0.2);
            transform: scale(1.05);
        }

        /* ENHANCED BREADCRUMB NAVIGATION STYLES */
        .tars-breadcrumbs {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 8px;
            padding: 16px 20px;
            margin-bottom: 20px;
            border: 1px solid rgba(0, 255, 136, 0.2);
            backdrop-filter: blur(10px);
        }

        .breadcrumb-status {
            display: flex;
            align-items: center;
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 1px solid rgba(0, 255, 136, 0.1);
        }

        .status-icon {
            font-size: 16px;
            margin-right: 8px;
        }

        .status-text {
            color: #00ff88;
            font-weight: bold;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .breadcrumb-container {
            display: flex;
            align-items: center;
            font-size: 14px;
            color: #e0e0e0;
        }

        .breadcrumb-item {
            color: #00ccff;
            text-decoration: none;
            padding: 4px 8px;
            border-radius: 4px;
            transition: all 0.3s ease;
            cursor: pointer;
        }

        .breadcrumb-item:hover {
            background: rgba(0, 204, 255, 0.2);
            color: #ffffff;
            transform: scale(1.05);
        }

        .breadcrumb-item.home {
            color: #00ff88;
            font-weight: bold;
        }

        .breadcrumb-item.home:hover {
            background: rgba(0, 255, 136, 0.2);
        }

        .breadcrumb-item.current {
            color: #ffffff;
            background: rgba(0, 255, 136, 0.3);
            cursor: default;
            font-weight: bold;
        }

        .breadcrumb-item.current:hover {
            transform: none;
        }

        .breadcrumb-separator {
            color: #666;
            margin: 0 8px;
            font-weight: bold;
        }

        .breadcrumb-close {
            color: #ff6b6b;
            margin-left: 8px;
            padding: 2px 6px;
            border-radius: 3px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: bold;
        }

        .breadcrumb-close:hover {
            background: rgba(255, 107, 107, 0.2);
            color: #ffffff;
            transform: scale(1.1);
        }

        .breadcrumb-item.view {
            color: #00ccff;
        }

        .breadcrumb-item.view:hover {
            background: rgba(0, 204, 255, 0.2);
        }

        .breadcrumb-item.subsystem {
            color: #ff9500;
        }

        .dependencies-section {
            margin-top: 25px;
        }

        .dependencies-title {
            font-size: 1.3rem;
            margin-bottom: 15px;
            color: #00ccff;
        }

        .dependency-tag {
            display: inline-block;
            background: rgba(0, 204, 255, 0.2);
            color: #00ccff;
            padding: 5px 12px;
            border-radius: 15px;
            margin: 3px;
            font-size: 0.9rem;
            border: 1px solid rgba(0, 204, 255, 0.3);
        }

        .close-detail-btn {
            background: linear-gradient(45deg, #ff6b6b, #ff8e8e);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: bold;
            transition: transform 0.2s;
        }

        .close-detail-btn:hover {
            transform: translateY(-2px);
        }

        .detail-controls {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }

        .detail-toggle, .criticality-indicator, .evolution-indicator {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: #e0e0e0;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.8rem;
            cursor: pointer;
        }

        .detail-toggle:hover {
            background: rgba(255, 255, 255, 0.2);
        }

        .dependencies-list {
            list-style: none;
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
        }

        .dependencies-list li {
            background: rgba(255, 255, 255, 0.1);
            padding: 3px 8px;
            border-radius: 10px;
            font-size: 0.8rem;
        }

        .detailed-metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 8px;
        }

        .detailed-metric {
            display: flex;
            justify-content: space-between;
            background: rgba(255, 255, 255, 0.05);
            padding: 5px 10px;
            border-radius: 8px;
        }

        /* LOADING AND ERROR STATES */
        .loading-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            z-index: 1000;
        }

        .spinner {
            width: 50px;
            height: 50px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-top: 3px solid #00ff88;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .loading-text {
            margin-top: 20px;
            font-size: 1.2rem;
        }

        .error-panel {
            background: rgba(220, 53, 69, 0.2);
            border: 1px solid #dc3545;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .error-dismiss {
            background: #dc3545;
            color: white;
            border: none;
            padding: 5px 15px;
            border-radius: 15px;
            cursor: pointer;
        }

        /* RESPONSIVE DESIGN */
        @media (max-width: 768px) {
            .tars-metrics-grid {
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            }
            
            .subsystems-grid {
                grid-template-columns: 1fr;
            }
            
            .view-mode-controls {
                flex-direction: column;
                align-items: center;
            }
            
            .control-panel {
                flex-direction: column;
            }
        }

        /* ANIMATIONS */
        .functional-elmish-tars-app {
            animation: fadeIn 0.5s ease-in;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .subsystem-card {
            animation: slideIn 0.3s ease-out;
        }

        @keyframes slideIn {
            from { opacity: 0; transform: translateX(-20px); }
            to { opacity: 1; transform: translateX(0); }
        }

        /* MAIN CONTAINER STYLES */
        .tars-diagnostics-elmish {
            min-height: 100vh;
            padding: 20px;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
        }

        /* ENHANCED NAVIGATION SIDEBAR STYLES */
        .tars-layout {
            display: grid;
            grid-template-columns: 300px 1fr;
            gap: 20px;
            min-height: calc(100vh - 200px);
            margin-top: 20px;
        }

        .tars-sidebar {
            background: rgba(0, 0, 0, 0.4);
            backdrop-filter: blur(15px);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(0, 255, 136, 0.2);
            height: fit-content;
            max-height: 80vh;
            overflow-y: auto;
        }

        .sidebar-header {
            text-align: center;
            margin-bottom: 25px;
            padding-bottom: 15px;
            border-bottom: 1px solid rgba(0, 255, 136, 0.3);
        }

        .sidebar-header h3 {
            color: #00ff88;
            font-size: 1.2rem;
            text-shadow: 0 0 10px #00ff88;
        }

        .nav-section {
            margin-bottom: 25px;
        }

        .nav-section h4 {
            color: #00ccff;
            font-size: 1rem;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .nav-menu {
            list-style: none;
        }

        .nav-item {
            padding: 12px 15px;
            margin: 5px 0;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid transparent;
            font-size: 0.9rem;
        }

        .nav-item:hover {
            background: rgba(0, 255, 136, 0.1);
            border-color: rgba(0, 255, 136, 0.3);
            transform: translateX(5px);
        }

        .nav-item.active {
            background: linear-gradient(45deg, #00ff88, #00ccff);
            color: #000;
            font-weight: bold;
            border-color: #00ff88;
        }

        .control-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .status-on {
            color: #00ff88;
            font-weight: bold;
            text-shadow: 0 0 5px #00ff88;
        }

        .status-off {
            color: #ff6b6b;
            font-weight: bold;
        }

        .quick-stats {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .stat-item {
            padding: 8px 12px;
            background: rgba(0, 255, 136, 0.1);
            border-radius: 6px;
            font-size: 0.85rem;
            border-left: 3px solid #00ff88;
        }

        .tars-main-content {
            background: rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            border: 1px solid rgba(0, 255, 136, 0.2);
            overflow-y: auto;
            max-height: 80vh;
        }

        .coming-soon {
            text-align: center;
            padding: 50px;
            color: #00ccff;
        }

        .coming-soon h2 {
            font-size: 2rem;
            margin-bottom: 15px;
            text-shadow: 0 0 20px #00ccff;
        }

        .coming-soon p {
            font-size: 1.1rem;
            opacity: 0.8;
        }

        /* Responsive design for smaller screens */
        @media (max-width: 1200px) {
            .tars-layout {
                grid-template-columns: 250px 1fr;
            }
        }

        @media (max-width: 768px) {
            .tars-layout {
                grid-template-columns: 1fr;
                grid-template-rows: auto 1fr;
            }

            .tars-sidebar {
                max-height: 300px;
            }
        }
        </style>
        """

    // COMPLETE HTML TEMPLATE GENERATOR
    let generateCompleteHtml (model: TarsElmishDiagnostics.TarsDiagnosticsModel) =
        // Create a simplified model for JSON serialization (avoiding F# discriminated unions)
        let simplifiedModel = {|
            allSubsystems = model.AllSubsystems |> List.map (fun s -> {|
                name = s.Name
                status = s.Status.ToString()
                healthPercentage = s.HealthPercentage
                activeComponents = s.ActiveComponents
                processingRate = s.ProcessingRate
                memoryUsage = s.MemoryUsage
                lastActivity = s.LastActivity
                dependencies = s.Dependencies
            |})
            overallTarsHealth = model.OverallTarsHealth
            activeAgents = model.ActiveAgents
            processingTasks = model.ProcessingTasks
            isLoading = model.IsLoading
            error = model.Error
            lastUpdate = model.LastUpdate
            selectedSubsystem = model.SelectedSubsystem
            showDetails = model.ShowDetails
            viewMode = model.ViewMode.ToString()
            autoRefresh = model.AutoRefresh
        |}
        let modelJson = JsonSerializer.Serialize(simplifiedModel, JsonSerializerOptions(PropertyNamingPolicy = JsonNamingPolicy.CamelCase))
        // Create a dummy dispatch for server-side rendering
        let dummyDispatch msg = printfn "Server-side dispatch: %A" msg
        let viewHtml = TarsElmishDiagnostics.view model dummyDispatch |> TarsHtml.render

        let cssContent = generateTarsCSS ()
        let jsContent = generateJavaScriptRuntime ()

        sprintf """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🧠 TARS Consciousness & Subsystem Matrix</title>
    <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🧠</text></svg>">
    %s
</head>
<body>
    <div id="elmish-tars-root">
        %s
    </div>

    %s

    <script>
        // Initialize the Elmish application with the current model
        const initialModel = %s;
        initElmish(JSON.stringify(initialModel));

        // Add some TARS-specific enhancements
        console.log('🧠 TARS Consciousness Matrix Activated');
        console.log('⚡ Subsystems:', initialModel.allSubsystems ? initialModel.allSubsystems.length : 0);
        console.log('🧬 Overall Health:', initialModel.overallTarsHealth + '%%');
        console.log('💭 Active Agents:', initialModel.activeAgents);

        // Initialize dark mode
        document.body.classList.add('dark-mode');
        localStorage.setItem('tars-dark-mode', 'true');
        console.log('🌙 Dark mode initialized');

        // Add visual effects
        document.body.style.background = 'linear-gradient(135deg, #0a0a0a 0%%, #1a1a2e 50%%, #16213e 100%%)';

        // Add particle effect (optional)
        function createParticle() {
            const particle = document.createElement('div');
            particle.style.position = 'fixed';
            particle.style.width = '2px';
            particle.style.height = '2px';
            particle.style.background = '#00ff88';
            particle.style.borderRadius = '50%%';
            particle.style.pointerEvents = 'none';
            particle.style.opacity = '0.7';
            particle.style.left = Math.random() * window.innerWidth + 'px';
            particle.style.top = window.innerHeight + 'px';
            particle.style.zIndex = '-1';

            document.body.appendChild(particle);

            const animation = particle.animate([
                { transform: 'translateY(0px)', opacity: 0.7 },
                { transform: 'translateY(-' + (window.innerHeight + 100) + 'px)', opacity: 0 }
            ], {
                duration: 3000 + Math.random() * 2000,
                easing: 'linear'
            });

            animation.onfinish = () => particle.remove();
        }

        // Create particles occasionally
        setInterval(createParticle, 500);

        // Add consciousness pulse effect
        function pulseConsciousness() {
            const consciousnessElements = document.querySelectorAll('.consciousness');
            consciousnessElements.forEach(el => {
                el.style.transform = 'scale(1.05)';
                setTimeout(() => {
                    el.style.transform = 'scale(1)';
                }, 200);
            });
        }

        setInterval(pulseConsciousness, 3000);

        // Keyboard shortcuts help
        console.log('⌨️ Keyboard Shortcuts:');
        console.log('  Ctrl+R: Refresh All');
        console.log('  Ctrl+E: Evolve');
        console.log('  Ctrl+M: Self-Modify');
        console.log('  Ctrl+C: Boost Consciousness');
        console.log('  Ctrl+Q: Quantum Tunnel');
        console.log('  1-7: Switch View Modes');
    </script>
</body>
</html>""" cssContent viewHtml jsContent modelJson

    // REAL ELMISH PROGRAM RUNNER - True MVU Architecture
    let runElmishProgram () =
        // Create the real Elmish program
        let program = TarsElmishDiagnostics.createTarsElmishProgram ()

        // Initialize the TARS model with comprehensive subsystems
        let initialModel = program.Init ()
        let subsystems = TarsElmishDiagnostics.generateComprehensiveTarsSubsystems ()
        let loadedModel = program.Update (TarsElmishDiagnostics.TarsSubsystemsLoaded subsystems) initialModel

        // Create a real dispatch function
        let mutable currentModel = loadedModel
        let rec dispatch msg =
            let newModel = program.Update msg currentModel
            currentModel <- newModel
            // In a real implementation, this would trigger a re-render
            printfn "🔄 Model updated via message: %A" msg

        // Generate initial view with real dispatch
        let initialView = program.View loadedModel dispatch

        (TarsHtml.render initialView, dispatch, loadedModel)

    // REAL ELMISH MESSAGE HANDLER - True MVU with Dispatch
    let handleElmishUpdate (messageJson: string) (modelJson: string) =
        try
            let program = TarsElmishDiagnostics.createTarsElmishProgram ()
            let message = JsonSerializer.Deserialize<TarsElmishDiagnostics.TarsMsg>(messageJson)
            let model = JsonSerializer.Deserialize<TarsElmishDiagnostics.TarsDiagnosticsModel>(modelJson, JsonSerializerOptions(PropertyNamingPolicy = JsonNamingPolicy.CamelCase))

            // Use the real Elmish program update and view functions
            let newModel = program.Update message model

            // Create a dummy dispatch for view rendering (server-side)
            let dummyDispatch msg = printfn "Server-side dispatch: %A" msg
            let newHtml = program.View newModel dummyDispatch |> TarsHtml.render

            {|
                success = true
                newModel = newModel
                newHtml = newHtml
                error = ""
            |}
        with
        | ex ->
            {|
                success = false
                newModel = TarsElmishDiagnostics.init()
                newHtml = ""
                error = ex.Message
            |}
