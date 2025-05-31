namespace TarsEngine.FSharp.Testing

open System
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open OpenQA.Selenium
open OpenQA.Selenium.Chrome
open OpenQA.Selenium.Firefox
open OpenQA.Selenium.Edge
open OpenQA.Selenium.Support.UI
open TestTypes

/// Comprehensive UI testing framework for TARS
module UITestFramework =
    
    /// UI test executor with Selenium WebDriver
    type UITestExecutor(logger: ILogger<UITestExecutor>, config: TestConfiguration) =
        
        let mutable driver: IWebDriver option = None
        let mutable wait: WebDriverWait option = None
        
        /// Initialize WebDriver based on browser type
        member private this.InitializeDriver(browserType: BrowserType) =
            let newDriver = 
                match browserType with
                | Chrome ->
                    let options = ChromeOptions()
                    options.AddArgument("--disable-web-security")
                    options.AddArgument("--disable-features=VizDisplayCompositor")
                    new ChromeDriver(options) :> IWebDriver
                | ChromeHeadless ->
                    let options = ChromeOptions()
                    options.AddArgument("--headless")
                    options.AddArgument("--disable-web-security")
                    options.AddArgument("--disable-gpu")
                    options.AddArgument("--no-sandbox")
                    new ChromeDriver(options) :> IWebDriver
                | Firefox ->
                    let options = FirefoxOptions()
                    new FirefoxDriver(options) :> IWebDriver
                | FirefoxHeadless ->
                    let options = FirefoxOptions()
                    options.AddArgument("--headless")
                    new FirefoxDriver(options) :> IWebDriver
                | Edge ->
                    let options = EdgeOptions()
                    new EdgeDriver(options) :> IWebDriver
                | Safari ->
                    failwith "Safari driver not supported on Windows"
            
            newDriver.Manage().Timeouts().ImplicitWait <- config.ImplicitWait
            newDriver.Manage().Window.Maximize()
            
            driver <- Some newDriver
            wait <- Some (WebDriverWait(newDriver, config.ExplicitWait))
            
            logger.LogInformation("WebDriver initialized: {BrowserType}", browserType)
            newDriver
        
        /// Get WebDriver element using locator strategy
        member private this.FindElement(locator: LocatorStrategy) =
            match driver with
            | Some d ->
                let by = 
                    match locator with
                    | Id id -> By.Id(id)
                    | ClassName className -> By.ClassName(className)
                    | TagName tagName -> By.TagName(tagName)
                    | XPath xpath -> By.XPath(xpath)
                    | CssSelector css -> By.CssSelector(css)
                    | LinkText text -> By.LinkText(text)
                    | PartialLinkText text -> By.PartialLinkText(text)
                    | Name name -> By.Name(name)
                
                try
                    Some (d.FindElement(by))
                with
                | :? NoSuchElementException -> None
            | None -> 
                failwith "WebDriver not initialized"
        
        /// Wait for element to be present
        member private this.WaitForElement(locator: LocatorStrategy, timeout: TimeSpan) =
            match wait with
            | Some w ->
                try
                    let by = 
                        match locator with
                        | Id id -> By.Id(id)
                        | ClassName className -> By.ClassName(className)
                        | TagName tagName -> By.TagName(tagName)
                        | XPath xpath -> By.XPath(xpath)
                        | CssSelector css -> By.CssSelector(css)
                        | LinkText text -> By.LinkText(text)
                        | PartialLinkText text -> By.PartialLinkText(text)
                        | Name name -> By.Name(name)
                    
                    let customWait = WebDriverWait(driver.Value, timeout)
                    Some (customWait.Until(SeleniumExtras.WaitHelpers.ExpectedConditions.ElementExists(by)))
                with
                | :? WebDriverTimeoutException -> None
            | None ->
                failwith "WebDriver not initialized"
        
        /// Take screenshot
        member private this.TakeScreenshot(filename: string) =
            match driver with
            | Some d ->
                try
                    let screenshot = (d :?> ITakesScreenshot).GetScreenshot()
                    let fullPath = Path.Combine(config.ScreenshotDirectory, filename)
                    Directory.CreateDirectory(Path.GetDirectoryName(fullPath)) |> ignore
                    screenshot.SaveAsFile(fullPath)
                    logger.LogInformation("Screenshot saved: {Filename}", fullPath)
                    Some fullPath
                with
                | ex ->
                    logger.LogError(ex, "Failed to take screenshot: {Filename}", filename)
                    None
            | None -> None
        
        /// Execute UI interaction
        member private this.ExecuteInteraction(element: IWebElement, interaction: UIInteraction) =
            try
                match interaction with
                | Click -> element.Click()
                | DoubleClick -> 
                    let actions = OpenQA.Selenium.Interactions.Actions(driver.Value)
                    actions.DoubleClick(element).Perform()
                | RightClick ->
                    let actions = OpenQA.Selenium.Interactions.Actions(driver.Value)
                    actions.ContextClick(element).Perform()
                | Hover ->
                    let actions = OpenQA.Selenium.Interactions.Actions(driver.Value)
                    actions.MoveToElement(element).Perform()
                | Type text ->
                    element.Clear()
                    element.SendKeys(text)
                | Clear -> element.Clear()
                | Submit -> element.Submit()
                | ScrollTo ->
                    let js = driver.Value :?> IJavaScriptExecutor
                    js.ExecuteScript("arguments[0].scrollIntoView(true);", element) |> ignore
                | DragAndDrop target ->
                    match this.FindElement(target) with
                    | Some targetElement ->
                        let actions = OpenQA.Selenium.Interactions.Actions(driver.Value)
                        actions.DragAndDrop(element, targetElement).Perform()
                    | None -> failwith $"Target element not found for drag and drop: {target}"
                
                true
            with
            | ex ->
                logger.LogError(ex, "Failed to execute interaction: {Interaction}", interaction)
                false
        
        /// Validate assertion
        member private this.ValidateAssertion(assertion: TestAssertion) =
            try
                match assertion with
                | ElementExists locator ->
                    match this.FindElement(locator) with
                    | Some _ -> true, None
                    | None -> false, Some $"Element not found: {locator}"
                
                | ElementNotExists locator ->
                    match this.FindElement(locator) with
                    | Some _ -> false, Some $"Element should not exist: {locator}"
                    | None -> true, None
                
                | ElementVisible locator ->
                    match this.FindElement(locator) with
                    | Some element -> element.Displayed, if element.Displayed then None else Some $"Element not visible: {locator}"
                    | None -> false, Some $"Element not found: {locator}"
                
                | ElementHidden locator ->
                    match this.FindElement(locator) with
                    | Some element -> not element.Displayed, if not element.Displayed then None else Some $"Element should be hidden: {locator}"
                    | None -> true, None
                
                | ElementEnabled locator ->
                    match this.FindElement(locator) with
                    | Some element -> element.Enabled, if element.Enabled then None else Some $"Element not enabled: {locator}"
                    | None -> false, Some $"Element not found: {locator}"
                
                | ElementDisabled locator ->
                    match this.FindElement(locator) with
                    | Some element -> not element.Enabled, if not element.Enabled then None else Some $"Element should be disabled: {locator}"
                    | None -> false, Some $"Element not found: {locator}"
                
                | ElementText (locator, expectedText) ->
                    match this.FindElement(locator) with
                    | Some element -> 
                        let actualText = element.Text
                        let matches = actualText = expectedText
                        matches, if matches then None else Some $"Text mismatch. Expected: '{expectedText}', Actual: '{actualText}'"
                    | None -> false, Some $"Element not found: {locator}"
                
                | ElementAttribute (locator, attribute, expectedValue) ->
                    match this.FindElement(locator) with
                    | Some element ->
                        let actualValue = element.GetAttribute(attribute)
                        let matches = actualValue = expectedValue
                        matches, if matches then None else Some $"Attribute '{attribute}' mismatch. Expected: '{expectedValue}', Actual: '{actualValue}'"
                    | None -> false, Some $"Element not found: {locator}"
                
                | PageTitle expectedTitle ->
                    match driver with
                    | Some d ->
                        let actualTitle = d.Title
                        let matches = actualTitle = expectedTitle
                        matches, if matches then None else Some $"Page title mismatch. Expected: '{expectedTitle}', Actual: '{actualTitle}'"
                    | None -> false, Some "WebDriver not initialized"
                
                | PageUrl expectedUrl ->
                    match driver with
                    | Some d ->
                        let actualUrl = d.Url
                        let matches = actualUrl = expectedUrl
                        matches, if matches then None else Some $"Page URL mismatch. Expected: '{expectedUrl}', Actual: '{actualUrl}'"
                    | None -> false, Some "WebDriver not initialized"
                
                | ResponseStatus expectedStatus ->
                    // This would require additional HTTP monitoring
                    true, None
                
                | ResponseTime maxMilliseconds ->
                    // This would require performance monitoring
                    true, None
                
                | Custom (description, assertionFunc) ->
                    try
                        let result = assertionFunc()
                        result, if result then None else Some $"Custom assertion failed: {description}"
                    with
                    | ex -> false, Some $"Custom assertion error: {ex.Message}"
            
            with
            | ex ->
                logger.LogError(ex, "Error validating assertion: {Assertion}", assertion)
                false, Some ex.Message
        
        /// Execute test action
        member this.ExecuteAction(action: TestAction) =
            async {
                try
                    match action with
                    | Navigate url ->
                        match driver with
                        | Some d -> 
                            d.Navigate().GoToUrl(url)
                            logger.LogInformation("Navigated to: {Url}", url)
                            return true, None
                        | None -> return false, Some "WebDriver not initialized"
                    
                    | Interact (locator, interaction) ->
                        match this.FindElement(locator) with
                        | Some element ->
                            let success = this.ExecuteInteraction(element, interaction)
                            return success, if success then None else Some $"Failed to execute interaction: {interaction}"
                        | None ->
                            return false, Some $"Element not found for interaction: {locator}"
                    
                    | Wait duration ->
                        do! Async.Sleep(int duration.TotalMilliseconds)
                        return true, None
                    
                    | WaitForElement (locator, timeout) ->
                        match this.WaitForElement(locator, timeout) with
                        | Some _ -> return true, None
                        | None -> return false, Some $"Element not found within timeout: {locator}"
                    
                    | ExecuteScript script ->
                        match driver with
                        | Some d ->
                            let js = d :?> IJavaScriptExecutor
                            js.ExecuteScript(script) |> ignore
                            return true, None
                        | None -> return false, Some "WebDriver not initialized"
                    
                    | SwitchToFrame locator ->
                        match this.FindElement(locator) with
                        | Some element ->
                            driver.Value.SwitchTo().Frame(element) |> ignore
                            return true, None
                        | None -> return false, Some $"Frame not found: {locator}"
                    
                    | SwitchToWindow windowHandle ->
                        driver.Value.SwitchTo().Window(windowHandle) |> ignore
                        return true, None
                    
                    | TakeScreenshot filename ->
                        match this.TakeScreenshot(filename) with
                        | Some path -> return true, None
                        | None -> return false, Some "Failed to take screenshot"
                    
                    | APICall (method, url, body) ->
                        // This would integrate with API testing framework
                        return true, None
                    
                    | Custom (description, actionFunc) ->
                        try
                            actionFunc()
                            return true, None
                        with
                        | ex -> return false, Some $"Custom action failed: {ex.Message}"
                
                with
                | ex ->
                    logger.LogError(ex, "Error executing action: {Action}", action)
                    return false, Some ex.Message
            }
        
        /// Execute test step
        member this.ExecuteStep(step: TestStep) =
            async {
                let startTime = DateTime.UtcNow
                logger.LogInformation("Executing step: {StepName}", step.Name)
                
                try
                    // Execute action
                    let! actionSuccess, actionError = this.ExecuteAction(step.Action)
                    
                    if not actionSuccess then
                        let endTime = DateTime.UtcNow
                        return {
                            Step = step
                            Status = Failed (actionError |> Option.defaultValue "Action failed")
                            StartTime = startTime
                            EndTime = Some endTime
                            Duration = endTime - startTime
                            ErrorMessage = actionError
                            Screenshot = None
                            ActualValue = None
                            ExpectedValue = None
                        }
                    
                    // Validate assertions
                    let mutable allAssertionsPassed = true
                    let mutable assertionErrors = []
                    
                    for assertion in step.Assertions do
                        let passed, error = this.ValidateAssertion(assertion)
                        if not passed then
                            allAssertionsPassed <- false
                            assertionErrors <- (error |> Option.defaultValue "Assertion failed") :: assertionErrors
                    
                    // Take screenshot if configured or on failure
                    let screenshot = 
                        if step.Screenshot || (not allAssertionsPassed && config.ScreenshotOnFailure) then
                            let filename = $"{step.Id}_{DateTime.UtcNow:yyyyMMdd_HHmmss}.png"
                            this.TakeScreenshot(filename)
                        else None
                    
                    let endTime = DateTime.UtcNow
                    let status = if allAssertionsPassed then Passed else Failed (String.concat "; " assertionErrors)
                    
                    return {
                        Step = step
                        Status = status
                        StartTime = startTime
                        EndTime = Some endTime
                        Duration = endTime - startTime
                        ErrorMessage = if allAssertionsPassed then None else Some (String.concat "; " assertionErrors)
                        Screenshot = screenshot
                        ActualValue = None
                        ExpectedValue = None
                    }
                
                with
                | ex ->
                    logger.LogError(ex, "Error executing step: {StepName}", step.Name)
                    let endTime = DateTime.UtcNow
                    return {
                        Step = step
                        Status = Failed ex.Message
                        StartTime = startTime
                        EndTime = Some endTime
                        Duration = endTime - startTime
                        ErrorMessage = Some ex.Message
                        Screenshot = None
                        ActualValue = None
                        ExpectedValue = None
                    }
            }
        
        /// Execute complete test case
        member this.ExecuteTestCase(testCase: TestCase) =
            async {
                let startTime = DateTime.UtcNow
                logger.LogInformation("Executing test case: {TestCaseName}", testCase.Name)
                
                try
                    // Initialize driver if browser specified
                    match testCase.Browser with
                    | Some browserType -> this.InitializeDriver(browserType) |> ignore
                    | None -> ()
                    
                    // Execute all steps
                    let stepResults = ResizeArray<TestStepResult>()
                    let mutable testPassed = true
                    
                    for step in testCase.Steps do
                        let! stepResult = this.ExecuteStep(step)
                        stepResults.Add(stepResult)
                        
                        match stepResult.Status with
                        | Failed _ -> testPassed <- false
                        | _ -> ()
                    
                    let endTime = DateTime.UtcNow
                    let status = if testPassed then Passed else Failed "One or more steps failed"
                    
                    return {
                        TestCase = testCase
                        Status = status
                        StartTime = startTime
                        EndTime = Some endTime
                        Duration = endTime - startTime
                        ErrorMessage = None
                        Screenshots = stepResults |> Seq.choose (fun sr -> sr.Screenshot) |> Seq.toList
                        Logs = []
                        PerformanceMetrics = Map.empty
                        StepResults = stepResults |> Seq.toList
                    }
                
                with
                | ex ->
                    logger.LogError(ex, "Error executing test case: {TestCaseName}", testCase.Name)
                    let endTime = DateTime.UtcNow
                    return {
                        TestCase = testCase
                        Status = Failed ex.Message
                        StartTime = startTime
                        EndTime = Some endTime
                        Duration = endTime - startTime
                        ErrorMessage = Some ex.Message
                        Screenshots = []
                        Logs = []
                        PerformanceMetrics = Map.empty
                        StepResults = []
                    }
            }
        
        /// Cleanup resources
        member this.Dispose() =
            match driver with
            | Some d ->
                try
                    d.Quit()
                    d.Dispose()
                    logger.LogInformation("WebDriver disposed")
                with
                | ex -> logger.LogError(ex, "Error disposing WebDriver")
            | None -> ()
            
            driver <- None
            wait <- None
        
        interface IDisposable with
            member this.Dispose() = this.Dispose()
