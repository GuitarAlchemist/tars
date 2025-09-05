namespace TarsEngine.FSharp.Cli.UI

/// CSS generation for Elmish applications
module ElmishCSS =

    /// Generate base CSS styles
    let generateBaseCSS () =
        """
        <style>
        /* TARS DIAGNOSTICS - DARK MODE STYLING */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
            color: #e0e0e0;
            overflow-x: hidden;
            min-height: 100vh;
        }

        .dark-mode {
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
            color: #e0e0e0;
        }

        .tars-container {
            display: flex;
            min-height: 100vh;
            position: relative;
        }

        /* SIDEBAR STYLING */
        .tars-sidebar {
            width: 280px;
            background: rgba(20, 20, 40, 0.95);
            backdrop-filter: blur(10px);
            border-right: 1px solid rgba(0, 255, 136, 0.3);
            padding: 20px;
            position: fixed;
            height: 100vh;
            overflow-y: auto;
            z-index: 1000;
        }

        .sidebar-header {
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 1px solid rgba(0, 255, 136, 0.3);
        }

        .sidebar-header h2 {
            font-size: 2.2em;
            color: #00ff88;
            text-shadow: 0 0 20px rgba(0, 255, 136, 0.5);
            margin-bottom: 5px;
        }

        .sidebar-subtitle {
            color: #888;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 2px;
        }

        /* NAVIGATION STYLING */
        .nav-section {
            margin-bottom: 25px;
        }

        .nav-section h3 {
            color: #00ff88;
            font-size: 1em;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .nav-item {
            display: flex;
            align-items: center;
            padding: 12px 15px;
            margin: 5px 0;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            background: rgba(255, 255, 255, 0.05);
        }

        .nav-item:hover {
            background: rgba(0, 255, 136, 0.1);
            transform: translateX(5px);
        }

        .nav-icon {
            margin-right: 10px;
            font-size: 1.2em;
        }

        .nav-label {
            flex: 1;
        }

        .control-item {
            justify-content: space-between;
        }

        .status-on {
            color: #00ff88;
            font-weight: bold;
        }

        .status-off {
            color: #ff6b6b;
            font-weight: bold;
        }

        /* MAIN CONTENT AREA */
        .tars-main {
            flex: 1;
            margin-left: 280px;
            padding: 20px;
            min-height: 100vh;
        }

        /* BREADCRUMB STYLING */
        .breadcrumb-container {
            background: rgba(20, 20, 40, 0.8);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(0, 255, 136, 0.3);
            border-radius: 12px;
            padding: 15px 20px;
            margin-bottom: 20px;
        }

        .breadcrumb-status-text {
            color: #00ff88;
            font-weight: bold;
            font-size: 1.1em;
        }

        .breadcrumb-path {
            margin-top: 8px;
            font-size: 0.9em;
        }

        .breadcrumb-item {
            color: #888;
            cursor: pointer;
            transition: color 0.3s ease;
        }

        .breadcrumb-item:hover {
            color: #00ff88;
        }

        .breadcrumb-item.current {
            color: #e0e0e0;
            font-weight: bold;
        }

        .breadcrumb-separator {
            color: #555;
            margin: 0 5px;
        }

        .breadcrumb-close {
            color: #ff6b6b;
            cursor: pointer;
            margin-left: 10px;
            font-weight: bold;
        }

        .breadcrumb-close:hover {
            color: #ff4444;
        }
        </style>
        """

    /// Generate grid and card CSS
    let generateGridCSS () =
        """
        <style>
        /* SUBSYSTEMS GRID */
        .subsystems-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            padding: 20px 0;
        }

        .subsystem-card {
            background: rgba(20, 20, 40, 0.8);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(0, 255, 136, 0.3);
            border-radius: 12px;
            padding: 20px;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .subsystem-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, #00ff88, #0088ff);
            opacity: 0;
            transition: opacity 0.3s ease;
        }

        .subsystem-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0, 255, 136, 0.2);
            border-color: rgba(0, 255, 136, 0.6);
        }

        .subsystem-card:hover::before {
            opacity: 1;
        }

        .subsystem-card.clickable {
            cursor: pointer;
        }

        .subsystem-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }

        .subsystem-header h3 {
            color: #e0e0e0;
            font-size: 1.3em;
            margin: 0;
        }

        .subsystem-status {
            font-size: 0.9em;
            font-weight: bold;
        }

        .subsystem-metrics {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }

        .metric-row {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .metric-label {
            color: #888;
            font-size: 0.9em;
        }

        .metric-value {
            color: #00ff88;
            font-weight: bold;
        }

        /* DETAIL VIEW STYLING */
        .subsystem-detail-view {
            background: rgba(20, 20, 40, 0.9);
            backdrop-filter: blur(15px);
            border: 1px solid rgba(0, 255, 136, 0.4);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
        }

        .detail-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 25px;
            padding-bottom: 15px;
            border-bottom: 1px solid rgba(0, 255, 136, 0.3);
        }

        .detail-title-section h2 {
            color: #e0e0e0;
            font-size: 1.8em;
            margin: 0 0 5px 0;
        }

        .detail-status {
            font-size: 1em;
            font-weight: bold;
        }

        .close-detail-btn {
            background: rgba(255, 107, 107, 0.2);
            border: 1px solid #ff6b6b;
            color: #ff6b6b;
            padding: 8px 15px;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .close-detail-btn:hover {
            background: rgba(255, 107, 107, 0.3);
            transform: scale(1.05);
        }

        /* METRICS GRID */
        .detail-metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 25px;
        }

        .metric-card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 15px;
            display: flex;
            align-items: center;
            transition: all 0.3s ease;
        }

        .metric-card:hover {
            background: rgba(255, 255, 255, 0.08);
            transform: scale(1.02);
        }

        .metric-card.primary {
            border-color: rgba(0, 255, 136, 0.5);
            background: rgba(0, 255, 136, 0.1);
        }

        .metric-icon {
            font-size: 1.5em;
            margin-right: 12px;
        }

        .metric-content {
            flex: 1;
        }

        .metric-value {
            font-size: 1.4em;
            font-weight: bold;
            color: #00ff88;
            margin-bottom: 2px;
        }

        .metric-value.large {
            font-size: 1.8em;
        }

        .metric-label {
            color: #888;
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        </style>
        """
