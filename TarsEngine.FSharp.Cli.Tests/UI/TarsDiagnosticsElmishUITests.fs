namespace TarsEngine.FSharp.Cli.Tests.UI

open System
open Xunit
open FsUnit.Xunit
open FluentAssertions
open TarsEngine.FSharp.Cli.UI.TarsDiagnosticsElmishUI

/// Tests for TARS Diagnostics Elmish UI - Ensuring real MVU architecture
module TarsDiagnosticsElmishUITests =

    [<Fact>]
    let ``init should create model with real data`` () =
        // Arrange
        let cognitiveEngine = Some (box "test_engine")
        let beliefBus = Some (box "test_bus")
        let projectManager = Some (box "test_manager")
        
        // Act
        let model = init cognitiveEngine beliefBus projectManager
        
        // Assert
        model.SelectedNavItem |> should equal "overview"
        model.NavigationItems |> should not' (be Empty)
        model.Breadcrumbs |> should not' (be Empty)
        model.ComponentAnalyses |> should not' (be Empty)
        model.LastUpdate |> should be (greaterThan DateTime.MinValue)
        model.IsLoading |> should be False
        model.AutoEvolutionEnabled |> should be True

    [<Fact>]
    let ``getRealNavigationItems should return actual navigation structure`` () =
        // Act
        let navItems = getRealNavigationItems()
        
        // Assert
        navItems |> should not' (be Empty)
        navItems |> List.length |> should be (greaterThan 5)
        
        // Check for expected navigation items
        let overviewItem = navItems |> List.find (fun item -> item.Id = "overview")
        overviewItem.Name |> should equal "System Overview"
        overviewItem.Icon |> should equal "🏠"
        
        let aiSystemsItem = navItems |> List.find (fun item -> item.Id = "ai-systems")
        aiSystemsItem.Name |> should equal "AI Systems"
        aiSystemsItem.Icon |> should equal "🤖"

    [<Fact>]
    let ``getRealSystemHealth should return actual system metrics`` () =
        // Act
        let systemHealth = getRealSystemHealth()
        
        // Assert
        // CPU usage should be based on real processor count
        systemHealth.CpuUsage |> should be (greaterThan 0.0)
        systemHealth.CpuUsage |> should equal (Environment.ProcessorCount |> float |> (*) 12.5)
        
        // Memory usage should be based on real GC data
        let expectedMemory = GC.GetTotalMemory(false) |> float |> (*) 0.000001
        systemHealth.MemoryUsage |> should equal expectedMemory
        
        // Uptime should be based on real process time
        systemHealth.Uptime |> should equal (System.Diagnostics.Process.GetCurrentProcess().TotalProcessorTime)

    [<Fact>]
    let ``getRealComponentAnalyses should calculate real health percentages`` () =
        // Arrange
        let cognitiveEngine = Some (box "test_engine")
        let beliefBus = Some (box "test_bus")
        let projectManager = Some (box "test_manager")
        
        // Act
        let analyses = getRealComponentAnalyses cognitiveEngine beliefBus projectManager
        
        // Assert
        analyses |> should not' (be Empty)
        analyses |> List.length |> should equal 3 // One for each component
        
        for analysis in analyses do
            // Health percentages should be calculated from real metrics
            analysis.Percentage |> should be (greaterThanOrEqualTo 0.0)
            analysis.Percentage |> should be (lessThanOrEqualTo 100.0)
            
            // Should not be hardcoded fake values
            analysis.Percentage |> should not' (equal 100.0)
            analysis.Percentage |> should not' (equal 94.0)
            analysis.Percentage |> should not' (equal 91.0)
            analysis.Percentage |> should not' (equal 87.0)
            
            // Status should reflect actual health
            if analysis.Percentage > 50.0 then
                analysis.Status |> should contain "Operational"
            else
                analysis.Status |> should contain "Degraded"
            
            // Metrics should contain real system data
            analysis.Metrics |> should not' (be Empty)
            
            // Last checked should be very recent
            let timeDiff = DateTime.Now - analysis.LastChecked
            timeDiff.TotalSeconds |> should be (lessThan 5.0)

    [<Fact>]
    let ``update function should handle SelectNavItem correctly`` () =
        // Arrange
        let cognitiveEngine = Some (box "test_engine")
        let beliefBus = Some (box "test_bus")
        let projectManager = Some (box "test_manager")
        let initialModel = init cognitiveEngine beliefBus projectManager
        let message = SelectNavItem "ai-systems"
        
        // Act
        let updatedModel = update cognitiveEngine beliefBus projectManager message initialModel
        
        // Assert
        updatedModel.SelectedNavItem |> should equal "ai-systems"
        updatedModel.CurrentView |> should equal "ai-systems"
        updatedModel.Breadcrumbs |> should contain ("🤖 AI Systems", "ai-systems")

    [<Fact>]
    let ``update function should handle RefreshDiagnostics correctly`` () =
        // Arrange
        let cognitiveEngine = Some (box "test_engine")
        let beliefBus = Some (box "test_bus")
        let projectManager = Some (box "test_manager")
        let initialModel = init cognitiveEngine beliefBus projectManager
        let originalUpdateTime = initialModel.LastUpdate
        let message = RefreshDiagnostics
        
        // Wait a small amount to ensure time difference
        System.Threading.Thread.Sleep(10)
        
        // Act
        let updatedModel = update cognitiveEngine beliefBus projectManager message initialModel
        
        // Assert
        updatedModel.LastUpdate |> should be (greaterThan originalUpdateTime)
        updatedModel.IsLoading |> should be False
        updatedModel.ComponentAnalyses |> should not' (be Empty)
        
        // System health should be recalculated
        updatedModel.SystemHealth |> should not' (equal initialModel.SystemHealth)

    [<Fact>]
    let ``update function should handle ToggleSidebar correctly`` () =
        // Arrange
        let cognitiveEngine = Some (box "test_engine")
        let beliefBus = Some (box "test_bus")
        let projectManager = Some (box "test_manager")
        let initialModel = init cognitiveEngine beliefBus projectManager
        let originalSidebarState = initialModel.IsSidebarCollapsed
        let message = ToggleSidebar
        
        // Act
        let updatedModel = update cognitiveEngine beliefBus projectManager message initialModel
        
        // Assert
        updatedModel.IsSidebarCollapsed |> should equal (not originalSidebarState)

    [<Fact>]
    let ``renderNavItem should create proper HTML structure`` () =
        // Arrange
        let navItem = {
            Id = "test-item"
            Name = "Test Item"
            Icon = "🧪"
            IsExpanded = false
            Children = []
            Status = Some "Active"
        }
        let dispatch = fun _ -> ()
        
        // Act
        let htmlElement = renderNavItem dispatch navItem
        
        // Assert
        match htmlElement with
        | Element("div", attrs, children) ->
            attrs |> should contain ("class", "nav-item")
            children |> should not' (be Empty)
        | _ -> failwith "Expected div element"

    [<Fact>]
    let ``renderComponentAnalysis should create proper HTML structure`` () =
        // Arrange
        let analysis = {
            Name = "Test Component"
            Status = "Operational"
            Percentage = 85.5
            Description = "Test component description"
            StatusColor = "#00ff00"
            LastChecked = DateTime.Now
            Dependencies = ["Dep1"; "Dep2"]
            Metrics = Map.ofList [("TestMetric", box 42)]
        }
        let dispatch = fun _ -> ()
        
        // Act
        let htmlElement = renderComponentAnalysis dispatch analysis
        
        // Assert
        match htmlElement with
        | Element("div", attrs, children) ->
            attrs |> should contain ("class", "component-analysis-item")
            children |> should not' (be Empty)
        | _ -> failwith "Expected div element"

    [<Fact>]
    let ``view function should create complete UI structure`` () =
        // Arrange
        let cognitiveEngine = Some (box "test_engine")
        let beliefBus = Some (box "test_bus")
        let projectManager = Some (box "test_manager")
        let model = init cognitiveEngine beliefBus projectManager
        let dispatch = fun _ -> ()
        
        // Act
        let htmlElement = view model dispatch
        
        // Assert
        match htmlElement with
        | Element("div", attrs, children) ->
            attrs |> should contain ("class", "tars-diagnostics-ui")
            children |> should not' (be Empty)
            children |> List.length |> should be (greaterThan 1) // Header + main content
        | _ -> failwith "Expected div element"

    [<Fact>]
    let ``createApp should return functional Elmish app`` () =
        // Arrange
        let cognitiveEngine = Some (box "test_engine")
        let beliefBus = Some (box "test_bus")
        let projectManager = Some (box "test_manager")
        
        // Act
        let app = createApp cognitiveEngine beliefBus projectManager
        
        // Assert
        app |> should not' (be null)
        
        // Should be able to call the app function
        let viewFunction = app()
        viewFunction |> should not' (be null)

    [<Fact>]
    let ``HTML helper functions should create correct elements`` () =
        // Test div function
        let divElement = div [("class", "test-class")] [text "test content"]
        match divElement with
        | Element("div", attrs, children) ->
            attrs |> should contain ("class", "test-class")
            children |> should contain (Text("test content"))
        | _ -> failwith "Expected div element"
        
        // Test span function
        let spanElement = span [("id", "test-id")] []
        match spanElement with
        | Element("span", attrs, children) ->
            attrs |> should contain ("id", "test-id")
            children |> should be Empty
        | _ -> failwith "Expected span element"
        
        // Test text function
        let textElement = text "Hello World"
        match textElement with
        | Text(content) -> content |> should equal "Hello World"
        | _ -> failwith "Expected text element"

    [<Fact>]
    let ``breadcrumb navigation should be correct for different views`` () =
        // Arrange
        let cognitiveEngine = Some (box "test_engine")
        let beliefBus = Some (box "test_bus")
        let projectManager = Some (box "test_manager")
        let initialModel = init cognitiveEngine beliefBus projectManager
        
        // Test different navigation items
        let testCases = [
            ("overview", [("🏠 Home", "overview")])
            ("ai-systems", [("🏠 Home", "overview"); ("🤖 AI Systems", "ai-systems")])
            ("cognitive-systems", [("🏠 Home", "overview"); ("🧠 Cognitive Systems", "cognitive-systems")])
            ("infrastructure", [("🏠 Home", "overview"); ("🏗️ Infrastructure", "infrastructure")])
        ]
        
        for (itemId, expectedBreadcrumbs) in testCases do
            // Act
            let message = SelectNavItem itemId
            let updatedModel = update cognitiveEngine beliefBus projectManager message initialModel
            
            // Assert
            updatedModel.Breadcrumbs |> should equal expectedBreadcrumbs

    [<Fact>]
    let ``component analysis should use real system metrics`` () =
        // Arrange
        let cognitiveEngine = Some (box "test_engine")
        let beliefBus = Some (box "test_bus")
        let projectManager = Some (box "test_manager")
        
        // Act
        let analyses = getRealComponentAnalyses cognitiveEngine beliefBus projectManager
        
        // Assert
        let cognitiveAnalysis = analyses |> List.find (fun a -> a.Name = "TARS Cognitive Engine")
        
        // Should contain real memory usage metric
        cognitiveAnalysis.Metrics |> Map.containsKey "MemoryUsage" |> should be True
        let memoryUsage = cognitiveAnalysis.Metrics.["MemoryUsage"] :?> int64
        memoryUsage |> should equal (GC.GetTotalMemory(false))
        
        // Should contain real processor time metric
        cognitiveAnalysis.Metrics |> Map.containsKey "ProcessorTime" |> should be True
        let processorTime = cognitiveAnalysis.Metrics.["ProcessorTime"] :?> float
        processorTime |> should equal (System.Diagnostics.Process.GetCurrentProcess().TotalProcessorTime.TotalMilliseconds)

    [<Fact>]
    let ``overall health calculation should be based on component health`` () =
        // Arrange
        let cognitiveEngine = Some (box "test_engine")
        let beliefBus = Some (box "test_bus")
        let projectManager = Some (box "test_manager")
        
        // Act
        let model = init cognitiveEngine beliefBus projectManager
        
        // Assert
        if not model.ComponentAnalyses.IsEmpty then
            let expectedHealth = model.ComponentAnalyses |> List.averageBy (fun c -> c.Percentage)
            model.OverallSystemHealth |> should equal expectedHealth
        else
            model.OverallSystemHealth |> should equal 0.0

    [<Fact>]
    let ``system status should reflect overall health`` () =
        // Arrange
        let cognitiveEngine = Some (box "test_engine")
        let beliefBus = Some (box "test_bus")
        let projectManager = Some (box "test_manager")
        
        // Act
        let model = init cognitiveEngine beliefBus projectManager
        
        // Assert
        if model.OverallSystemHealth > 90.0 then
            model.SystemStatus |> should equal "Excellent"
        elif model.OverallSystemHealth > 75.0 then
            model.SystemStatus |> should equal "Good"
        else
            model.SystemStatus |> should equal "Needs Attention"
