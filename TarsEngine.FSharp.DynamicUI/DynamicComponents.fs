namespace TarsEngine.FSharp.DynamicUI

open TarsEngine.FSharp.DynamicUI.Types

module DynamicComponents =

    /// Create a dynamic agent node component as HTML string
    let createAgentNodeHtml (agent: TarsAgent) =
        let color = if agent.Status = Running then "#00ff88" else "#333"
        let shadow = if agent.Status = Running then "0 0 20px #00ff88" else "none"
        let name = agent.Name.Substring(0, min 3 agent.Name.Length)

        sprintf """
        <div style="
            position: absolute;
            width: 60px;
            height: 60px;
            border-radius: 50%%;
            background: %s;
            border: 2px solid #00ff88;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 10px;
            font-weight: bold;
            color: #000;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: %s;
        " title="%s - %A - %d tasks">
            %s
        </div>
        """ color shadow agent.Name agent.Status agent.TaskCount name

    /// Create a performance metric display as HTML
    let createPerformanceMetricHtml (label: string) (value: string) (color: string) =
        sprintf """
        <div style="
            margin: 5px 0;
            padding: 5px 10px;
            background: rgba(0, 0, 0, 0.7);
            border-radius: 5px;
            border: 1px solid %s;
            font-size: 12px;
        ">
            <span style="color: %s; font-weight: bold;">%s: </span>
            <span>%s</span>
        </div>
        """ color color label value

    /// Create a generation activity indicator as HTML
    let createGenerationActivityHtml (agentName: string) (activity: string) =
        sprintf """
        <div style="
            margin: 3px 0;
            padding: 5px 8px;
            background: rgba(0, 255, 136, 0.1);
            border-left: 3px solid #00ff88;
            border-radius: 3px;
            font-size: 11px;
            animation: fadeIn 0.5s ease-in;
        ">
            <strong>%s: </strong>%s
        </div>
        """ agentName activity
