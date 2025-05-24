namespace TarsEngine.FSharp.Core.CodeGen.Testing.Models

open System

/// <summary>
/// Represents a parameter for a test case.
/// </summary>
type TestCaseParameter = {
    /// <summary>
    /// The name of the parameter.
    /// </summary>
    Name: string
    
    /// <summary>
    /// The type of the parameter.
    /// </summary>
    Type: string
    
    /// <summary>
    /// The value of the parameter.
    /// </summary>
    Value: string
    
    /// <summary>
    /// The setup code for the parameter.
    /// </summary>
    SetupCode: string option
}

/// <summary>
/// Represents an assertion for a test case.
/// </summary>
type TestCaseAssertion = {
    /// <summary>
    /// The type of the assertion.
    /// </summary>
    Type: string
    
    /// <summary>
    /// The expected value for the assertion.
    /// </summary>
    ExpectedValue: string option
    
    /// <summary>
    /// The actual value for the assertion.
    /// </summary>
    ActualValue: string
    
    /// <summary>
    /// The message for the assertion.
    /// </summary>
    Message: string option
    
    /// <summary>
    /// The code for the assertion.
    /// </summary>
    Code: string
}

/// <summary>
/// Represents the result of a test case.
/// </summary>
type TestCaseResult = {
    /// <summary>
    /// The type of the result.
    /// </summary>
    Type: string
    
    /// <summary>
    /// The value of the result.
    /// </summary>
    Value: string option
    
    /// <summary>
    /// The variable name for the result.
    /// </summary>
    VariableName: string option
}

/// <summary>
/// Represents a test case.
/// </summary>
type TestCase = {
    /// <summary>
    /// The name of the test case.
    /// </summary>
    Name: string
    
    /// <summary>
    /// The description of the test case.
    /// </summary>
    Description: string
    
    /// <summary>
    /// The method being tested.
    /// </summary>
    MethodName: string
    
    /// <summary>
    /// The class being tested.
    /// </summary>
    ClassName: string
    
    /// <summary>
    /// The namespace of the class being tested.
    /// </summary>
    Namespace: string
    
    /// <summary>
    /// The parameters for the test case.
    /// </summary>
    Parameters: TestCaseParameter list
    
    /// <summary>
    /// The setup code for the test case.
    /// </summary>
    SetupCode: string option
    
    /// <summary>
    /// The code to execute for the test case.
    /// </summary>
    ExecutionCode: string
    
    /// <summary>
    /// The result of the test case.
    /// </summary>
    Result: TestCaseResult option
    
    /// <summary>
    /// The assertions for the test case.
    /// </summary>
    Assertions: TestCaseAssertion list
    
    /// <summary>
    /// The cleanup code for the test case.
    /// </summary>
    CleanupCode: string option
    
    /// <summary>
    /// The test framework for the test case.
    /// </summary>
    TestFramework: string
    
    /// <summary>
    /// The language for the test case.
    /// </summary>
    Language: string
    
    /// <summary>
    /// Additional information for the test case.
    /// </summary>
    AdditionalInfo: Map<string, string>
}

/// <summary>
/// Represents a test suite.
/// </summary>
type TestSuite = {
    /// <summary>
    /// The name of the test suite.
    /// </summary>
    Name: string
    
    /// <summary>
    /// The description of the test suite.
    /// </summary>
    Description: string
    
    /// <summary>
    /// The class being tested.
    /// </summary>
    ClassName: string
    
    /// <summary>
    /// The namespace of the class being tested.
    /// </summary>
    Namespace: string
    
    /// <summary>
    /// The test cases in the suite.
    /// </summary>
    TestCases: TestCase list
    
    /// <summary>
    /// The setup code for the test suite.
    /// </summary>
    SetupCode: string option
    
    /// <summary>
    /// The cleanup code for the test suite.
    /// </summary>
    CleanupCode: string option
    
    /// <summary>
    /// The test framework for the test suite.
    /// </summary>
    TestFramework: string
    
    /// <summary>
    /// The language for the test suite.
    /// </summary>
    Language: string
    
    /// <summary>
    /// Additional information for the test suite.
    /// </summary>
    AdditionalInfo: Map<string, string>
}
