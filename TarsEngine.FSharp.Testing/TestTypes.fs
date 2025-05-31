namespace TarsEngine.FSharp.Testing

open System
open System.Collections.Generic

/// Core types for TARS UI and integration testing framework
module TestTypes =
    
    /// Test execution status
    type TestStatus =
        | NotStarted
        | Running
        | Passed
        | Failed of string
        | Skipped of string
        | Blocked of string
    
    /// Test severity levels
    type TestSeverity =
        | Critical
        | High
        | Medium
        | Low
        | Informational
    
    /// Test categories
    type TestCategory =
        | Unit
        | Integration
        | UI
        | API
        | Performance
        | Security
        | Accessibility
        | Smoke
        | Regression
        | EndToEnd
    
    /// Browser types for UI testing
    type BrowserType =
        | Chrome
        | Firefox
        | Edge
        | Safari
        | ChromeHeadless
        | FirefoxHeadless
    
    /// Device types for responsive testing
    type DeviceType =
        | Desktop
        | Tablet
        | Mobile
        | Custom of width: int * height: int
    
    /// UI element locator strategies
    type LocatorStrategy =
        | Id of string
        | ClassName of string
        | TagName of string
        | XPath of string
        | CssSelector of string
        | LinkText of string
        | PartialLinkText of string
        | Name of string
    
    /// UI interaction types
    type UIInteraction =
        | Click
        | DoubleClick
        | RightClick
        | Hover
        | Type of string
        | Clear
        | Submit
        | ScrollTo
        | DragAndDrop of target: LocatorStrategy
    
    /// Test assertion types
    type TestAssertion =
        | ElementExists of LocatorStrategy
        | ElementNotExists of LocatorStrategy
        | ElementVisible of LocatorStrategy
        | ElementHidden of LocatorStrategy
        | ElementEnabled of LocatorStrategy
        | ElementDisabled of LocatorStrategy
        | ElementText of LocatorStrategy * expectedText: string
        | ElementAttribute of LocatorStrategy * attribute: string * expectedValue: string
        | PageTitle of expectedTitle: string
        | PageUrl of expectedUrl: string
        | ResponseStatus of expectedStatus: int
        | ResponseTime of maxMilliseconds: int
        | Custom of description: string * assertion: unit -> bool
    
    /// Test step definition
    type TestStep = {
        Id: Guid
        Name: string
        Description: string
        Action: TestAction
        Assertions: TestAssertion list
        Timeout: TimeSpan
        RetryCount: int
        Screenshot: bool
    }
    
    and TestAction =
        | Navigate of url: string
        | Interact of LocatorStrategy * UIInteraction
        | Wait of TimeSpan
        | WaitForElement of LocatorStrategy * TimeSpan
        | ExecuteScript of script: string
        | SwitchToFrame of LocatorStrategy
        | SwitchToWindow of windowHandle: string
        | TakeScreenshot of filename: string
        | APICall of method: string * url: string * body: string option
        | Custom of description: string * action: unit -> unit
    
    /// Test case definition
    type TestCase = {
        Id: Guid
        Name: string
        Description: string
        Category: TestCategory
        Severity: TestSeverity
        Tags: string list
        Prerequisites: string list
        Steps: TestStep list
        ExpectedResult: string
        Browser: BrowserType option
        Device: DeviceType option
        TestData: Map<string, obj>
        Timeout: TimeSpan
        RetryOnFailure: bool
    }
    
    /// Test execution result
    type TestResult = {
        TestCase: TestCase
        Status: TestStatus
        StartTime: DateTime
        EndTime: DateTime option
        Duration: TimeSpan
        ErrorMessage: string option
        Screenshots: string list
        Logs: string list
        PerformanceMetrics: Map<string, float>
        StepResults: TestStepResult list
    }
    
    and TestStepResult = {
        Step: TestStep
        Status: TestStatus
        StartTime: DateTime
        EndTime: DateTime option
        Duration: TimeSpan
        ErrorMessage: string option
        Screenshot: string option
        ActualValue: string option
        ExpectedValue: string option
    }
    
    /// Test suite definition
    type TestSuite = {
        Id: Guid
        Name: string
        Description: string
        TestCases: TestCase list
        SetupSteps: TestStep list
        TeardownSteps: TestStep list
        ParallelExecution: bool
        MaxParallelTests: int
        Environment: string
        Configuration: Map<string, string>
    }
    
    /// Test execution report
    type TestReport = {
        Suite: TestSuite
        Results: TestResult list
        StartTime: DateTime
        EndTime: DateTime
        TotalDuration: TimeSpan
        Summary: TestSummary
        Environment: TestEnvironment
        Screenshots: string list
        Logs: string list
    }
    
    and TestSummary = {
        TotalTests: int
        PassedTests: int
        FailedTests: int
        SkippedTests: int
        BlockedTests: int
        PassRate: float
        FailRate: float
        AverageExecutionTime: TimeSpan
        CriticalFailures: int
        HighSeverityFailures: int
    }
    
    and TestEnvironment = {
        OperatingSystem: string
        Browser: string
        BrowserVersion: string
        ScreenResolution: string
        ApplicationVersion: string
        DatabaseVersion: string option
        TestFrameworkVersion: string
        Timestamp: DateTime
    }
    
    /// Performance test metrics
    type PerformanceMetrics = {
        ResponseTime: float
        Throughput: float
        ErrorRate: float
        CPUUsage: float
        MemoryUsage: float
        NetworkLatency: float
        DatabaseResponseTime: float option
        ConcurrentUsers: int
        TransactionsPerSecond: float
    }
    
    /// Accessibility test results
    type AccessibilityResult = {
        Rule: string
        Impact: string
        Description: string
        Help: string
        HelpUrl: string
        Nodes: AccessibilityNode list
    }
    
    and AccessibilityNode = {
        Target: string list
        Html: string
        Impact: string
        Any: AccessibilityCheck list
        All: AccessibilityCheck list
        None: AccessibilityCheck list
    }
    
    and AccessibilityCheck = {
        Id: string
        Impact: string
        Message: string
        Data: Map<string, obj>
    }
    
    /// Test configuration
    type TestConfiguration = {
        BaseUrl: string
        Timeout: TimeSpan
        ImplicitWait: TimeSpan
        ExplicitWait: TimeSpan
        ScreenshotOnFailure: bool
        ScreenshotDirectory: string
        LogLevel: string
        BrowserOptions: Map<string, obj>
        RetryCount: int
        ParallelExecution: bool
        MaxParallelTests: int
        ReportFormat: string list
        ReportDirectory: string
    }
    
    /// QA Agent persona for testing
    type QAAgentPersona = {
        Name: string
        TestingExpertise: TestCategory list
        AutomationSkills: string list
        QualityStandards: string list
        TestingPhilosophy: string
        PreferredTools: string list
        ExperienceLevel: string
        Specializations: string list
    }
    
    /// Test execution context
    type TestExecutionContext = {
        Configuration: TestConfiguration
        Environment: TestEnvironment
        QAAgent: QAAgentPersona
        StartTime: DateTime
        SessionId: Guid
        TestDataDirectory: string
        OutputDirectory: string
        Variables: Map<string, obj>
    }
