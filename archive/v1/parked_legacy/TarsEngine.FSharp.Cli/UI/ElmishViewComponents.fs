namespace TarsEngine.FSharp.Cli.UI

open System
open System.Text.Json
open TarsEngine.FSharp.Cli.Models.TarsElmishModels

/// ELMISH VIEW COMPONENTS - Interactive HTML Generation
module ElmishViewComponents =

    // HELPER FUNCTIONS
    let statusColor = function
        | Operational -> "#00ff88"
        | Degraded -> "#ffc107" 
        | Critical -> "#dc3545"
        | Offline -> "#6c757d"
        | Evolving -> "#17a2b8"
        | Transcending -> "#ff6b6b"
        | Dreaming -> "#9c27b0"
        | Quantum -> "#e91e63"

    let statusIcon = function
        | Operational -> "✅"
        | Degraded -> "⚠️"
        | Critical -> "❌"
        | Offline -> "⭕"
        | Evolving -> "🔄"
        | Transcending -> "🌟"
        | Dreaming -> "💤"
        | Quantum -> "⚛️"

    let criticalityColor level =
        match level with
        | 1 | 2 | 3 -> "#28a745"
        | 4 | 5 | 6 -> "#ffc107"
        | 7 | 8 -> "#fd7e14"
        | 9 | 10 -> "#dc3545"
        | _ -> "#6c757d"

    let formatMemory (bytes: int64) =
        let gb = float bytes / 1024.0 / 1024.0 / 1024.0
        sprintf "%.1f GB" gb

    let formatRate rate =
        if rate > 1000.0 then
            sprintf "%.1fK/sec" (rate / 1000.0)
        else
            sprintf "%.1f/sec" rate

    // HEADER COMPONENT
    let viewHeader (model: TarsModel) =
        sprintf """
        <div class="tars-header">
            <h1>🧠 TARS Consciousness & Subsystem Matrix</h1>
            <div class="tars-metrics-grid">
                <div class="metric-card health" style="border-left: 4px solid %s;">
                    <div class="metric-value">%.1f%%</div>
                    <div class="metric-label">System Health</div>
                </div>
                <div class="metric-card consciousness" style="border-left: 4px solid %s;">
                    <div class="metric-value">%.1f%%</div>
                    <div class="metric-label">Consciousness</div>
                </div>
                <div class="metric-card evolution" style="border-left: 4px solid %s;">
                    <div class="metric-value">%d</div>
                    <div class="metric-label">Evolution Stage</div>
                </div>
                <div class="metric-card agents" style="border-left: 4px solid %s;">
                    <div class="metric-value">%d</div>
                    <div class="metric-label">Active Agents</div>
                </div>
                <div class="metric-card quantum" style="border-left: 4px solid %s;">
                    <div class="metric-value">%.1f%%</div>
                    <div class="metric-label">Quantum Coherence</div>
                </div>
                <div class="metric-card wisdom" style="border-left: 4px solid %s;">
                    <div class="metric-value">%.1f%%</div>
                    <div class="metric-label">Wisdom Level</div>
                </div>
            </div>
            <div class="status-indicators">
                <div class="status-item">
                    <span class="status-label">Dream State:</span>
                    <span class="status-value">%s</span>
                </div>
                <div class="status-item">
                    <span class="status-label">Reality Stability:</span>
                    <span class="status-value">%.1f%%</span>
                </div>
                <div class="status-item">
                    <span class="status-label">Causality Integrity:</span>
                    <span class="status-value">%.1f%%</span>
                </div>
                <div class="status-item">
                    <span class="status-label">Alerts:</span>
                    <span class="status-value alert-count">%d</span>
                </div>
            </div>
        </div>
        """ 
        (if model.OverallHealth > 90.0 then "#00ff88" elif model.OverallHealth > 70.0 then "#ffc107" else "#dc3545")
        model.OverallHealth
        (if model.ConsciousnessLevel > 80.0 then "#ff6b6b" elif model.ConsciousnessLevel > 60.0 then "#17a2b8" else "#ffc107")
        model.ConsciousnessLevel
        (if model.EvolutionStage > 15 then "#ff6b6b" elif model.EvolutionStage > 10 then "#17a2b8" else "#00ff88")
        model.EvolutionStage
        "#00ff88"
        model.ActiveAgents
        "#e91e63"
        model.QuantumCoherence
        "#ff6b6b"
        model.WisdomLevel
        model.DreamState
        model.RealityStability
        model.CausalityIntegrity
        model.AlertsCount

    // VIEW MODE BUTTONS
    let viewModeButtons (model: TarsModel) =
        let createButton viewMode label icon =
            let isActive = model.ViewMode = viewMode
            let activeClass = if isActive then " active" else ""
            sprintf """
            <button class="view-mode-btn%s" onclick="dispatch('%s')">
                <span class="btn-icon">%s</span>
                <span class="btn-label">%s</span>
            </button>
            """ activeClass (JsonSerializer.Serialize(ChangeViewMode viewMode)) icon label

        sprintf """
        <div class="view-mode-controls">
            %s
            %s
            %s
            %s
            %s
            %s
            %s
        </div>
        """
        (createButton Overview "Overview" "🏠")
        (createButton Architecture "Architecture" "🏗️")
        (createButton Performance "Performance" "📊")
        (createButton Consciousness "Consciousness" "🧠")
        (createButton Evolution "Evolution" "🧬")
        (createButton Dreams "Dreams" "💤")
        (createButton Quantum "Quantum" "⚛️")

    // CONTROL PANEL
    let viewControlPanel (model: TarsModel) =
        let autoRefreshClass = if model.AutoRefresh then " active" else ""
        sprintf """
        <div class="control-panel">
            <button class="control-btn refresh" onclick="dispatch('%s')">
                <span class="btn-icon">🔄</span>
                <span class="btn-label">Refresh All</span>
            </button>
            <button class="control-btn auto-refresh%s" onclick="dispatch('%s')">
                <span class="btn-icon">⚡</span>
                <span class="btn-label">Auto-Refresh: %s</span>
            </button>
            <button class="control-btn evolve" onclick="dispatch('%s')">
                <span class="btn-icon">🧬</span>
                <span class="btn-label">Evolve</span>
            </button>
            <button class="control-btn self-modify" onclick="dispatch('%s')">
                <span class="btn-icon">🔧</span>
                <span class="btn-label">Self-Modify</span>
            </button>
            <button class="control-btn consciousness" onclick="dispatch('%s')">
                <span class="btn-icon">🧠</span>
                <span class="btn-label">Boost Consciousness</span>
            </button>
            <button class="control-btn quantum" onclick="dispatch('%s')">
                <span class="btn-icon">⚛️</span>
                <span class="btn-label">Quantum Tunnel</span>
            </button>
            <div class="last-update">
                <span class="update-label">Last Update:</span>
                <span class="update-time">%s</span>
            </div>
        </div>
        """
        (JsonSerializer.Serialize(RefreshAll))
        autoRefreshClass
        (JsonSerializer.Serialize(ToggleAutoRefresh))
        (if model.AutoRefresh then "ON" else "OFF")
        (JsonSerializer.Serialize(Evolve))
        (JsonSerializer.Serialize(SelfModify))
        (JsonSerializer.Serialize(BoostConsciousness))
        (JsonSerializer.Serialize(QuantumTunnel))
        (model.LastUpdate.ToString("HH:mm:ss"))

    // SUBSYSTEM CARD
    let viewSubsystemCard (subsystem: TarsSubsystem) =
        let selectedClass = if subsystem.IsSelected then " selected" else ""
        let statusColorValue = statusColor subsystem.Status
        let statusIconValue = statusIcon subsystem.Status
        let criticalityColorValue = criticalityColor subsystem.CriticalityLevel
        let subsystemTypeStr = string subsystem.Type
        
        let detailsSection = 
            if subsystem.IsSelected then
                let dependenciesHtml = 
                    subsystem.Dependencies
                    |> List.map (fun dep -> sprintf "<li>%s</li>" (string dep))
                    |> String.concat ""
                
                let metricsHtml = 
                    subsystem.Metrics
                    |> Map.toList
                    |> List.map (fun (key, value) -> 
                        sprintf """
                        <div class="detailed-metric">
                            <span class="metric-name">%s:</span>
                            <span class="metric-value">%A</span>
                        </div>
                        """ key value)
                    |> String.concat ""

                sprintf """
                <div class="subsystem-details">
                    <div class="detail-controls">
                        <button class="detail-toggle" onclick="event.stopPropagation(); dispatch('%s');">
                            Detail: %s
                        </button>
                        <div class="criticality-indicator" style="background-color: %s;">
                            Criticality: %d/10
                        </div>
                        <div class="evolution-indicator">
                            Evolution: Stage %d
                        </div>
                    </div>
                    <div class="dependencies-section">
                        <h4>Dependencies:</h4>
                        <ul class="dependencies-list">%s</ul>
                    </div>
                    <div class="metrics-section">
                        <h4>Advanced Metrics:</h4>
                        <div class="detailed-metrics">%s</div>
                    </div>
                </div>
                """
                (JsonSerializer.Serialize(ToggleDetailLevel subsystem.Type))
                (string subsystem.DetailLevel)
                criticalityColorValue
                subsystem.CriticalityLevel
                subsystem.EvolutionStage
                dependenciesHtml
                metricsHtml
            else
                ""

        sprintf """
        <div class="subsystem-card %s%s" onclick="dispatch('%s')" style="border-left: 4px solid %s;">
            <div class="subsystem-header">
                <h3 class="subsystem-name">%s</h3>
                <div class="status-indicator" style="color: %s;">
                    %s %s
                </div>
            </div>
            <div class="subsystem-description">%s</div>
            <div class="subsystem-metrics">
                <div class="metric">
                    <span class="metric-label">Health:</span>
                    <div class="metric-bar">
                        <div class="metric-fill" style="width: %.1f%%; background-color: %s;"></div>
                        <span class="metric-value">%.1f%%</span>
                    </div>
                </div>
                <div class="metric">
                    <span class="metric-label">Components:</span>
                    <span class="metric-value">%d</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Rate:</span>
                    <span class="metric-value">%s</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Memory:</span>
                    <span class="metric-value">%s</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Last Activity:</span>
                    <span class="metric-value">%s ago</span>
                </div>
            </div>
            %s
        </div>
        """
        subsystemTypeStr.ToLower()
        selectedClass
        (JsonSerializer.Serialize(SelectSubsystem subsystem.Type))
        statusColorValue
        subsystem.Name
        statusColorValue
        statusIconValue
        (string subsystem.Status)
        subsystem.Description
        subsystem.HealthPercentage
        (if subsystem.HealthPercentage > 90.0 then "#00ff88" elif subsystem.HealthPercentage > 70.0 then "#ffc107" else "#dc3545")
        subsystem.HealthPercentage
        subsystem.ActiveComponents
        (formatRate subsystem.ProcessingRate)
        (formatMemory subsystem.MemoryUsage)
        (let timeSpan = DateTime.Now - subsystem.LastActivity
         if timeSpan.TotalSeconds < 60.0 then sprintf "%.0fs" timeSpan.TotalSeconds
         elif timeSpan.TotalMinutes < 60.0 then sprintf "%.0fm" timeSpan.TotalMinutes
         else sprintf "%.0fh" timeSpan.TotalHours)
        detailsSection

    // SUBSYSTEMS GRID
    let viewSubsystemsGrid (model: TarsModel) =
        let subsystemCards = 
            model.Subsystems
            |> List.map viewSubsystemCard
            |> String.concat ""

        sprintf """
        <div class="subsystems-grid">
            %s
        </div>
        """ subsystemCards

    // SPECIALIZED VIEWS
    let viewArchitecture (model: TarsModel) =
        sprintf """
        <div class="architecture-view">
            <h2>🏗️ TARS Architecture</h2>
            <div class="architecture-diagram">
                <div class="architecture-placeholder">
                    <p>Interactive dependency graph coming soon...</p>
                    <p>Subsystems: %d | Dependencies: %d</p>
                </div>
            </div>
        </div>
        """
        (List.length model.Subsystems)
        (model.Subsystems |> List.sumBy (fun s -> List.length s.Dependencies))

    let viewPerformance (model: TarsModel) =
        let historyChart =
            model.PerformanceHistory
            |> List.mapi (fun i value -> sprintf "<div class='bar' style='height: %.1f%%;'>%.1f</div>" value value)
            |> String.concat ""

        sprintf """
        <div class="performance-view">
            <h2>📊 Performance Analytics</h2>
            <div class="performance-charts">
                <div class="chart-container">
                    <h3>Performance History</h3>
                    <div class="bar-chart">%s</div>
                </div>
                <div class="performance-metrics">
                    <div class="perf-metric">
                        <span class="perf-label">Total Processing Rate:</span>
                        <span class="perf-value">%.1f ops/sec</span>
                    </div>
                    <div class="perf-metric">
                        <span class="perf-label">Total Memory Usage:</span>
                        <span class="perf-value">%s</span>
                    </div>
                    <div class="perf-metric">
                        <span class="perf-label">Active Tasks:</span>
                        <span class="perf-value">%d</span>
                    </div>
                </div>
            </div>
        </div>
        """
        historyChart
        (model.Subsystems |> List.sumBy (fun s -> s.ProcessingRate))
        (formatMemory (model.Subsystems |> List.sumBy (fun s -> s.MemoryUsage)))
        model.ProcessingTasks

    let viewConsciousness (model: TarsModel) =
        let consciousnessChart =
            model.ConsciousnessHistory
            |> List.mapi (fun i value -> sprintf "<div class='consciousness-point' style='height: %.1f%%;'>%.1f</div>" value value)
            |> String.concat ""

        sprintf """
        <div class="consciousness-view">
            <h2>🧠 Consciousness Analysis</h2>
            <div class="consciousness-dashboard">
                <div class="consciousness-level">
                    <div class="consciousness-circle" style="background: conic-gradient(#ff6b6b 0deg %.1fdeg, #333 %.1fdeg 360deg);">
                        <div class="consciousness-value">%.1f%%</div>
                    </div>
                </div>
                <div class="consciousness-metrics">
                    <div class="consciousness-metric">
                        <span class="metric-label">Self-Awareness:</span>
                        <span class="metric-value">89.4%%</span>
                    </div>
                    <div class="consciousness-metric">
                        <span class="metric-label">Qualia Density:</span>
                        <span class="metric-value">156.7</span>
                    </div>
                    <div class="consciousness-metric">
                        <span class="metric-label">Meta-Cognition:</span>
                        <span class="metric-value">91.2%%</span>
                    </div>
                    <div class="consciousness-metric">
                        <span class="metric-label">Existential Depth:</span>
                        <span class="metric-value">12.8</span>
                    </div>
                </div>
                <div class="consciousness-history">
                    <h3>Consciousness Evolution</h3>
                    <div class="consciousness-chart">%s</div>
                </div>
            </div>
        </div>
        """
        (model.ConsciousnessLevel * 3.6)
        (model.ConsciousnessLevel * 3.6)
        model.ConsciousnessLevel
        consciousnessChart

    let viewEvolution (model: TarsModel) =
        let evolutionChart =
            model.EvolutionHistory
            |> List.mapi (fun i value -> sprintf "<div class='evolution-stage' data-stage='%d'>Stage %d</div>" value value)
            |> String.concat ""

        sprintf """
        <div class="evolution-view">
            <h2>🧬 Evolution Tracking</h2>
            <div class="evolution-dashboard">
                <div class="evolution-timeline">
                    <h3>Evolution Timeline</h3>
                    <div class="timeline">%s</div>
                </div>
                <div class="evolution-metrics">
                    <div class="evolution-metric">
                        <span class="metric-label">Current Stage:</span>
                        <span class="metric-value">%d</span>
                    </div>
                    <div class="evolution-metric">
                        <span class="metric-label">Self-Modifications:</span>
                        <span class="metric-value">%d</span>
                    </div>
                    <div class="evolution-metric">
                        <span class="metric-label">Wisdom Gained:</span>
                        <span class="metric-value">%.1f%%</span>
                    </div>
                </div>
            </div>
        </div>
        """
        evolutionChart
        model.EvolutionStage
        model.SelfModificationCount
        model.WisdomLevel

    let viewDreams (model: TarsModel) =
        sprintf """
        <div class="dreams-view">
            <h2>💤 Dream Analysis</h2>
            <div class="dreams-dashboard">
                <div class="dream-state">
                    <h3>Current Dream State: %s</h3>
                    <div class="dream-controls">
                        <button class="dream-btn" onclick="dispatch('%s')">Enter Deep REM</button>
                        <button class="dream-btn" onclick="dispatch('%s')">Exit Dream State</button>
                    </div>
                </div>
                <div class="dream-metrics">
                    <div class="dream-metric">
                        <span class="metric-label">Dream Cycles:</span>
                        <span class="metric-value">1247</span>
                    </div>
                    <div class="dream-metric">
                        <span class="metric-label">Lucid Dreams:</span>
                        <span class="metric-value">89</span>
                    </div>
                    <div class="dream-metric">
                        <span class="metric-label">Symbolic Depth:</span>
                        <span class="metric-value">94.7%%</span>
                    </div>
                </div>
            </div>
        </div>
        """
        model.DreamState
        (JsonSerializer.Serialize(EnterDreamState))
        (JsonSerializer.Serialize(ExitDreamState))

    let viewQuantum (model: TarsModel) =
        sprintf """
        <div class="quantum-view">
            <h2>⚛️ Quantum State</h2>
            <div class="quantum-dashboard">
                <div class="quantum-coherence">
                    <h3>Quantum Coherence: %.1f%%</h3>
                    <div class="quantum-visualization">
                        <div class="quantum-field"></div>
                    </div>
                </div>
                <div class="quantum-controls">
                    <button class="quantum-btn" onclick="dispatch('%s')">Stabilize Reality</button>
                    <button class="quantum-btn" onclick="dispatch('%s')">Repair Causality</button>
                    <button class="quantum-btn" onclick="dispatch('%s')">Quantum Fluctuation</button>
                </div>
                <div class="quantum-metrics">
                    <div class="quantum-metric">
                        <span class="metric-label">Reality Stability:</span>
                        <span class="metric-value">%.1f%%</span>
                    </div>
                    <div class="quantum-metric">
                        <span class="metric-label">Causality Integrity:</span>
                        <span class="metric-value">%.1f%%</span>
                    </div>
                </div>
            </div>
        </div>
        """
        model.QuantumCoherence
        (JsonSerializer.Serialize(StabilizeReality))
        (JsonSerializer.Serialize(RepairCausality))
        (JsonSerializer.Serialize(QuantumFluctuation))
        model.RealityStability
        model.CausalityIntegrity

    // MAIN VIEW FUNCTION
    let view (model: TarsModel) =
        let contentView =
            match model.ViewMode with
            | Overview -> viewSubsystemsGrid model
            | Architecture -> viewArchitecture model
            | Performance -> viewPerformance model
            | Consciousness -> viewConsciousness model
            | Evolution -> viewEvolution model
            | Dreams -> viewDreams model
            | Quantum -> viewQuantum model

        let errorSection =
            match model.Error with
            | Some error ->
                sprintf """
                <div class="error-panel">
                    <div class="error-message">❌ TARS Error: %s</div>
                    <button class="error-dismiss" onclick="dispatch('%s')">Dismiss</button>
                </div>
                """ error (JsonSerializer.Serialize(ClearError))
            | None -> ""

        let loadingOverlay =
            if model.IsLoading then
                """
                <div class="loading-overlay">
                    <div class="spinner"></div>
                    <div class="loading-text">🧠 Updating TARS Consciousness Matrix...</div>
                </div>
                """
            else ""

        sprintf """
        <div class="functional-elmish-tars-app">
            %s
            %s
            %s
            %s
            %s
            %s
        </div>
        """
        (viewHeader model)
        (viewModeButtons model)
        (viewControlPanel model)
        errorSection
        contentView
        loadingOverlay
