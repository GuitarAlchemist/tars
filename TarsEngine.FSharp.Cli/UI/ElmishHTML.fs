namespace TarsEngine.FSharp.Cli.UI

/// HTML generation for Elmish applications
module ElmishHTML =

    /// Generate subsystem detail HTML
    let generateSubsystemDetailHTML (subsystem: obj) =
        """
        <div class="subsystem-detail-view">
            <div class="detail-header">
                <div class="detail-title-section">
                    <h2>🔧 ${subsystem.Name}</h2>
                    <span class="detail-status" style="color: ${getStatusColor(subsystem.Status)}">
                        ${getStatusIcon(subsystem.Status)} ${subsystem.Status}
                    </span>
                </div>
                <button class="close-detail-btn" onclick="dispatch('ClearSubsystemSelection')">
                    ✕ Close
                </button>
            </div>

            <div class="detail-metrics-grid">
                <div class="metric-card primary">
                    <div class="metric-icon">💚</div>
                    <div class="metric-content">
                        <div class="metric-value large">${subsystem.HealthPercentage.toFixed(1)}%</div>
                        <div class="metric-label">System Health</div>
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-icon">⚙️</div>
                    <div class="metric-content">
                        <div class="metric-value">${subsystem.ActiveComponents}</div>
                        <div class="metric-label">Active Components</div>
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-icon">⚡</div>
                    <div class="metric-content">
                        <div class="metric-value">${subsystem.ProcessingRate.toFixed(1)}/s</div>
                        <div class="metric-label">Processing Rate</div>
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-icon">💾</div>
                    <div class="metric-content">
                        <div class="metric-value">${(subsystem.MemoryUsage / 1024 / 1024 / 1024).toFixed(1)} GB</div>
                        <div class="metric-label">Memory Usage</div>
                    </div>
                </div>
            </div>

            <div class="detail-section">
                <h3>🔗 Dependencies</h3>
                <div class="dependencies-grid">
                    ${subsystem.Dependencies.map(dep => `
                        <div class="dependency-item">
                            <span class="dependency-icon">🔗</span>
                            <span class="dependency-name">${dep}</span>
                        </div>
                    `).join('')}
                </div>
            </div>

            <div class="detail-section">
                <h3>📊 Advanced Metrics</h3>
                <div class="advanced-metrics">
                    ${Object.entries(subsystem.Metrics || {}).map(([key, value]) => `
                        <div class="advanced-metric-row">
                            <span class="metric-name">${key}</span>
                            <span class="metric-value">${formatMetricValue(value)}</span>
                        </div>
                    `).join('')}
                </div>
            </div>
        </div>
        """

    /// Generate overview HTML
    let generateOverviewHTML () =
        """
        <div class="tars-overview">
            <div class="subsystems-grid">
                ${currentModel.AllSubsystems.map(subsystem => `
                    <div class="subsystem-card clickable ${subsystem.Name.toLowerCase()}"
                         onclick="dispatch('SelectSubsystem', '${subsystem.Name}')">
                        <div class="subsystem-header">
                            <h3>${subsystem.Name}</h3>
                            <span class="subsystem-status" style="color: ${getStatusColor(subsystem.Status)}">
                                ${getStatusIcon(subsystem.Status)} ${subsystem.Status}
                            </span>
                        </div>
                        <div class="subsystem-metrics">
                            <div class="metric-row">
                                <span class="metric-label">Health:</span>
                                <span class="metric-value">${subsystem.HealthPercentage.toFixed(1)}%</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-label">Components:</span>
                                <span class="metric-value">${subsystem.ActiveComponents}</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-label">Rate:</span>
                                <span class="metric-value">${subsystem.ProcessingRate.toFixed(1)}/sec</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-label">Memory:</span>
                                <span class="metric-value">${(subsystem.MemoryUsage / 1024 / 1024).toFixed(1)} MB</span>
                            </div>
                        </div>
                    </div>
                `).join('')}
            </div>
        </div>
        """

    /// Generate breadcrumb navigation HTML
    let generateBreadcrumbHTML (selectedSubsystem: string option) =
        match selectedSubsystem with
        | Some subsystem ->
            $"""
            <div class="breadcrumb-container">
                <div class="breadcrumb-status">
                    <span class="breadcrumb-status-text">🔍 Viewing {subsystem} Details</span>
                </div>
                <div class="breadcrumb-path">
                    <span class="breadcrumb-item home" onclick="dispatch('ClearSubsystemSelection')">🧠 TARS Diagnostics</span>
                    <span class="breadcrumb-separator"> > </span>
                    <span class="breadcrumb-item view" onclick="dispatch('ClearSubsystemSelection')">🔍 Subsystems Overview</span>
                    <span class="breadcrumb-separator"> > </span>
                    <span class="breadcrumb-item subsystem current">🔧 {subsystem}</span>
                    <span class="breadcrumb-close" onclick="dispatch('ClearSubsystemSelection')"> ✕</span>
                </div>
            </div>
            """
        | None ->
            """
            <div class="breadcrumb-container">
                <div class="breadcrumb-status">
                    <span class="breadcrumb-status-text">🏠 Browsing Subsystems Overview</span>
                </div>
                <div class="breadcrumb-path">
                    <span class="breadcrumb-item home current">🧠 TARS Diagnostics</span>
                    <span class="breadcrumb-separator"> > </span>
                    <span class="breadcrumb-item view current">🔍 Subsystems Overview</span>
                </div>
            </div>
            """

    /// Generate sidebar HTML
    let generateSidebarHTML (autoRefresh: bool) (showDetails: bool) =
        $"""
        <div class="tars-sidebar">
            <div class="sidebar-header">
                <h2>🧠 TARS</h2>
                <div class="sidebar-subtitle">Diagnostics</div>
            </div>
            
            <div class="nav-section">
                <h3>🎛️ Controls</h3>
                <div class="nav-item control-item" onclick="dispatch('ToggleAutoRefresh')">
                    <span class="control-label">🔄 Auto Refresh</span>
                    <span class="{if autoRefresh then "status-on" else "status-off"}">{if autoRefresh then "ON" else "OFF"}</span>
                </div>
                <div class="nav-item control-item" onclick="dispatch('ToggleDetails')">
                    <span class="control-label">📋 Show Details</span>
                    <span class="{if showDetails then "status-on" else "status-off"}">{if showDetails then "ON" else "OFF"}</span>
                </div>
            </div>
            
            <div class="nav-section">
                <h3>📊 Views</h3>
                <div class="nav-item" onclick="dispatch('ChangeViewMode', 'Overview')">
                    <span class="nav-icon">🏠</span>
                    <span class="nav-label">Overview</span>
                </div>
                <div class="nav-item" onclick="dispatch('ChangeViewMode', 'Architecture')">
                    <span class="nav-icon">🏗️</span>
                    <span class="nav-label">Architecture</span>
                </div>
                <div class="nav-item" onclick="dispatch('ChangeViewMode', 'Performance')">
                    <span class="nav-icon">⚡</span>
                    <span class="nav-label">Performance</span>
                </div>
                <div class="nav-item" onclick="dispatch('ChangeViewMode', 'Evolution')">
                    <span class="nav-icon">🧬</span>
                    <span class="nav-label">Evolution</span>
                </div>
            </div>
            
            <div class="nav-section">
                <h3>🛠️ Actions</h3>
                <div class="nav-item action-item" onclick="dispatch('RefreshAll')">
                    <span class="nav-icon">🔄</span>
                    <span class="nav-label">Refresh All</span>
                </div>
                <div class="nav-item action-item" onclick="dispatch('Evolve')">
                    <span class="nav-icon">🧬</span>
                    <span class="nav-label">Evolve</span>
                </div>
                <div class="nav-item action-item" onclick="toggleDarkMode()">
                    <span class="nav-icon">🌙</span>
                    <span class="nav-label">Dark Mode</span>
                </div>
            </div>
        </div>
        """

    /// Generate main layout HTML
    let generateMainLayoutHTML (sidebarHTML: string) (breadcrumbHTML: string) (contentHTML: string) =
        $"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>TARS Diagnostics</title>
        </head>
        <body class="dark-mode">
            <div id="elmish-tars-root">
                <div class="tars-container">
                    {sidebarHTML}
                    <div class="tars-main">
                        {breadcrumbHTML}
                        <div class="main-content">
                            {contentHTML}
                        </div>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
