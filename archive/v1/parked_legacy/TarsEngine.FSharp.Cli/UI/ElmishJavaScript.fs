namespace TarsEngine.FSharp.Cli.UI

/// JavaScript runtime generation for Elmish applications
module ElmishJavaScript =

    /// Generate core JavaScript runtime for Elmish
    let generateCoreRuntime () =
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

        // REAL CLIENT-SIDE Elmish dispatch with immediate DOM updates
        function dispatch(messageType, payload = null) {
            console.log('📨 Dispatching REAL client-side message:', messageType, payload);

            // Update model based on message type (client-side update function)
            switch(messageType) {
                case 'SelectSubsystem':
                    currentModel.SelectedSubsystem = payload;
                    console.log('🔧 Selected subsystem:', payload);
                    break;

                case 'ClearSubsystemSelection':
                    currentModel.SelectedSubsystem = null;
                    console.log('🏠 Cleared subsystem selection');
                    break;

                case 'ChangeViewMode':
                    currentModel.ViewMode = payload;
                    currentModel.SelectedSubsystem = null;
                    console.log('📊 Changed view mode to:', payload);
                    break;

                case 'ToggleAutoRefresh':
                    currentModel.AutoRefresh = !currentModel.AutoRefresh;
                    console.log('🔄 Toggled auto refresh:', currentModel.AutoRefresh);
                    break;

                case 'ToggleDetails':
                    currentModel.ShowDetails = !currentModel.ShowDetails;
                    console.log('📋 Toggled details:', currentModel.ShowDetails);
                    break;

                default:
                    console.log('❓ Unknown message type:', messageType);
                    return;
            }

            // Immediately re-render the UI with updated model
            renderElmishUI();
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

        // Make functions globally available
        window.dispatch = dispatch;
        window.updateModel = updateModel;
        window.updateView = updateView;
        window.initElmish = initElmish;
        </script>
        """

    /// Generate keyboard handling JavaScript
    let generateKeyboardHandling () =
        """
        <script>
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
        </script>
        """

    /// Generate utility functions JavaScript
    let generateUtilities () =
        """
        <script>
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

        // Status utility functions
        function getStatusColor(status) {
            switch(status) {
                case 'Operational': return '#00ff88';
                case 'Evolving': return '#ffaa00';
                case 'Critical': return '#ff4444';
                case 'Offline': return '#666666';
                default: return '#ffffff';
            }
        }

        function getStatusIcon(status) {
            switch(status) {
                case 'Operational': return '✅';
                case 'Evolving': return '🔄';
                case 'Critical': return '⚠️';
                case 'Offline': return '❌';
                default: return '❓';
            }
        }

        function formatMetricValue(value) {
            if (typeof value === 'number') {
                if (value > 1000000) {
                    return (value / 1000000).toFixed(1) + 'M';
                } else if (value > 1000) {
                    return (value / 1000).toFixed(1) + 'K';
                } else {
                    return value.toFixed(1);
                }
            }
            return value.toString();
        }

        // Make functions globally available
        window.toggleDarkMode = toggleDarkMode;
        window.formatTime = formatTime;
        window.animateMetric = animateMetric;
        window.getStatusColor = getStatusColor;
        window.getStatusIcon = getStatusIcon;
        window.formatMetricValue = formatMetricValue;
        </script>
        """
