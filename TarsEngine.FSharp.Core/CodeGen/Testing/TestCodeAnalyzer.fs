namespace TarsEngine.FSharp.Core.CodeGen.Testing

open System
open System.Collections.Generic
open System.Text.RegularExpressions
open Microsoft.Extensions.Logging

/// <summary>
/// Represents a method extracted from code.
/// </summary>
type ExtractedMethod = {
    /// <summary>
    /// The name of the method.
    /// </summary>
    Name: string
    
    /// <summary>
    /// The return type of the method.
    /// </summary>
    ReturnType: string
    
    /// <summary>
    /// The parameters of the method.
    /// </summary>
    Parameters: (string * string) list
    
    /// <summary>
    /// The body of the method.
    /// </summary>
    Body: string
    
    /// <summary>
    /// The modifiers of the method.
    /// </summary>
    Modifiers: string list
    
    /// <summary>
    /// The class name containing the method.
    /// </summary>
    ClassName: string
    
    /// <summary>
    /// The namespace containing the class.
    /// </summary>
    Namespace: string
}

/// <summary>
/// Represents a class extracted from code.
/// </summary>
type ExtractedClass = {
    /// <summary>
    /// The name of the class.
    /// </summary>
    Name: string
    
    /// <summary>
    /// The namespace containing the class.
    /// </summary>
    Namespace: string
    
    /// <summary>
    /// The methods in the class.
    /// </summary>
    Methods: ExtractedMethod list
    
    /// <summary>
    /// The properties in the class.
    /// </summary>
    Properties: (string * string) list
    
    /// <summary>
    /// The modifiers of the class.
    /// </summary>
    Modifiers: string list
    
    /// <summary>
    /// The base class of the class.
    /// </summary>
    BaseClass: string option
    
    /// <summary>
    /// The interfaces implemented by the class.
    /// </summary>
    Interfaces: string list
}

/// <summary>
/// Analyzer for code to generate tests.
/// </summary>
type TestCodeAnalyzer(logger: ILogger<TestCodeAnalyzer>) =
    
    /// <summary>
    /// Extracts methods from C# code.
    /// </summary>
    /// <param name="code">The code to extract methods from.</param>
    /// <returns>The list of extracted methods.</returns>
    member _.ExtractMethodsFromCSharp(code: string) =
        try
            logger.LogInformation("Extracting methods from C# code")
            
            // Define regex pattern for methods
            let methodPattern = @"(public|private|protected|internal)?\s*(static|virtual|abstract|override|sealed)?\s*([a-zA-Z0-9_<>\.]+)\s+([a-zA-Z0-9_]+)\s*\(([^)]*)\)\s*\{([^}]*)\}"
            
            // Find all methods
            let methodMatches = Regex.Matches(code, methodPattern, RegexOptions.Singleline)
            
            // Extract class and namespace information
            let classPattern = @"(public|private|protected|internal)?\s*(static|abstract|sealed)?\s*class\s+([a-zA-Z0-9_]+)(?:<[^>]+>)?\s*(?::\s*[^{]+)?\s*\{"
            let namespacePattern = @"namespace\s+([a-zA-Z0-9_\.]+)\s*\{"
            
            let classMatch = Regex.Match(code, classPattern)
            let namespaceMatch = Regex.Match(code, namespacePattern)
            
            let className = if classMatch.Success then classMatch.Groups.[3].Value else "UnknownClass"
            let namespaceName = if namespaceMatch.Success then namespaceMatch.Groups.[1].Value else "UnknownNamespace"
            
            // Extract methods
            methodMatches
            |> Seq.cast<Match>
            |> Seq.map (fun m ->
                let modifiers = ResizeArray<string>()
                if m.Groups.[1].Success && not (String.IsNullOrWhiteSpace(m.Groups.[1].Value)) then
                    modifiers.Add(m.Groups.[1].Value)
                if m.Groups.[2].Success && not (String.IsNullOrWhiteSpace(m.Groups.[2].Value)) then
                    modifiers.Add(m.Groups.[2].Value)
                
                let returnType = m.Groups.[3].Value
                let methodName = m.Groups.[4].Value
                let parameterString = m.Groups.[5].Value
                let methodBody = m.Groups.[6].Value
                
                // Parse parameters
                let parameters = 
                    if String.IsNullOrWhiteSpace(parameterString) then
                        []
                    else
                        parameterString.Split(',')
                        |> Array.map (fun p -> 
                            let parts = p.Trim().Split(' ')
                            if parts.Length >= 2 then
                                (parts.[parts.Length - 1], parts.[0])
                            else
                                ("", p.Trim())
                        )
                        |> Array.toList
                
                {
                    Name = methodName
                    ReturnType = returnType
                    Parameters = parameters
                    Body = methodBody
                    Modifiers = modifiers |> Seq.toList
                    ClassName = className
                    Namespace = namespaceName
                }
            )
            |> Seq.toList
        with
        | ex ->
            logger.LogError(ex, "Error extracting methods from C# code")
            []
    
    /// <summary>
    /// Extracts methods from F# code.
    /// </summary>
    /// <param name="code">The code to extract methods from.</param>
    /// <returns>The list of extracted methods.</returns>
    member _.ExtractMethodsFromFSharp(code: string) =
        try
            logger.LogInformation("Extracting methods from F# code")
            
            // Define regex patterns for functions and members
            let functionPattern = @"let\s+(?:rec\s+)?(?:private\s+)?([a-zA-Z0-9_]+)(?:<[^>]+>)?\s*([a-zA-Z0-9_]+(?:\s+[a-zA-Z0-9_]+)*)\s*(?::\s*([^=]+))?\s*=\s*([^l]*)"
            let memberPattern = @"member\s+(?:private\s+)?(?:this|self|_)\.([a-zA-Z0-9_]+)(?:<[^>]+>)?\s*([a-zA-Z0-9_]+(?:\s+[a-zA-Z0-9_]+)*)\s*(?::\s*([^=]+))?\s*=\s*([^m]*)"
            
            // Extract module and namespace information
            let modulePattern = @"module\s+(?:rec\s+)?([a-zA-Z0-9_\.]+)"
            let namespacePattern = @"namespace\s+([a-zA-Z0-9_\.]+)"
            let typePattern = @"type\s+(?:private\s+)?([a-zA-Z0-9_]+)(?:<[^>]+>)?\s*(?:=\s*[^{]+)?\s*(?:\{|=|\()"
            
            let moduleMatch = Regex.Match(code, modulePattern)
            let namespaceMatch = Regex.Match(code, namespacePattern)
            let typeMatch = Regex.Match(code, typePattern)
            
            let moduleName = if moduleMatch.Success then moduleMatch.Groups.[1].Value else "UnknownModule"
            let namespaceName = if namespaceMatch.Success then namespaceMatch.Groups.[1].Value else "UnknownNamespace"
            let typeName = if typeMatch.Success then typeMatch.Groups.[1].Value else "UnknownType"
            
            // Extract functions
            let functionMatches = Regex.Matches(code, functionPattern, RegexOptions.Singleline)
            let memberMatches = Regex.Matches(code, memberPattern, RegexOptions.Singleline)
            
            // Process functions
            let functions = 
                functionMatches
                |> Seq.cast<Match>
                |> Seq.map (fun m ->
                    let functionName = m.Groups.[1].Value
                    let parameterString = m.Groups.[2].Value
                    let returnType = if m.Groups.[3].Success then m.Groups.[3].Value else "unknown"
                    let functionBody = m.Groups.[4].Value
                    
                    // Parse parameters
                    let parameters = 
                        if String.IsNullOrWhiteSpace(parameterString) then
                            []
                        else
                            parameterString.Split(' ')
                            |> Array.map (fun p -> (p.Trim(), "unknown"))
                            |> Array.toList
                    
                    {
                        Name = functionName
                        ReturnType = returnType
                        Parameters = parameters
                        Body = functionBody
                        Modifiers = ["let"]
                        ClassName = moduleName
                        Namespace = namespaceName
                    }
                )
                |> Seq.toList
            
            // Process members
            let members = 
                memberMatches
                |> Seq.cast<Match>
                |> Seq.map (fun m ->
                    let memberName = m.Groups.[1].Value
                    let parameterString = m.Groups.[2].Value
                    let returnType = if m.Groups.[3].Success then m.Groups.[3].Value else "unknown"
                    let memberBody = m.Groups.[4].Value
                    
                    // Parse parameters
                    let parameters = 
                        if String.IsNullOrWhiteSpace(parameterString) then
                            []
                        else
                            parameterString.Split(' ')
                            |> Array.map (fun p -> (p.Trim(), "unknown"))
                            |> Array.toList
                    
                    {
                        Name = memberName
                        ReturnType = returnType
                        Parameters = parameters
                        Body = memberBody
                        Modifiers = ["member"]
                        ClassName = typeName
                        Namespace = namespaceName
                    }
                )
                |> Seq.toList
            
            // Combine functions and members
            functions @ members
        with
        | ex ->
            logger.LogError(ex, "Error extracting methods from F# code")
            []
    
    /// <summary>
    /// Extracts classes from C# code.
    /// </summary>
    /// <param name="code">The code to extract classes from.</param>
    /// <returns>The list of extracted classes.</returns>
    member this.ExtractClassesFromCSharp(code: string) =
        try
            logger.LogInformation("Extracting classes from C# code")
            
            // Define regex pattern for classes
            let classPattern = @"(public|private|protected|internal)?\s*(static|abstract|sealed)?\s*class\s+([a-zA-Z0-9_]+)(?:<[^>]+>)?\s*(?::\s*([^{]+))?\s*\{([^}]*)\}"
            
            // Extract namespace information
            let namespacePattern = @"namespace\s+([a-zA-Z0-9_\.]+)\s*\{"
            let namespaceMatch = Regex.Match(code, namespacePattern)
            let namespaceName = if namespaceMatch.Success then namespaceMatch.Groups.[1].Value else "UnknownNamespace"
            
            // Find all classes
            let classMatches = Regex.Matches(code, classPattern, RegexOptions.Singleline)
            
            // Extract classes
            classMatches
            |> Seq.cast<Match>
            |> Seq.map (fun m ->
                let modifiers = ResizeArray<string>()
                if m.Groups.[1].Success && not (String.IsNullOrWhiteSpace(m.Groups.[1].Value)) then
                    modifiers.Add(m.Groups.[1].Value)
                if m.Groups.[2].Success && not (String.IsNullOrWhiteSpace(m.Groups.[2].Value)) then
                    modifiers.Add(m.Groups.[2].Value)
                
                let className = m.Groups.[3].Value
                let inheritance = m.Groups.[4].Value
                let classBody = m.Groups.[5].Value
                
                // Parse inheritance
                let baseClass, interfaces = 
                    if String.IsNullOrWhiteSpace(inheritance) then
                        None, []
                    else
                        let parts = inheritance.Split(',')
                        let baseClassPart = parts.[0].Trim()
                        let interfaceParts = parts |> Array.skip 1 |> Array.map (fun p -> p.Trim())
                        
                        if baseClassPart.StartsWith("I") then
                            None, baseClassPart :: (interfaceParts |> Array.toList)
                        else
                            Some baseClassPart, interfaceParts |> Array.toList
                
                // Extract methods
                let methods = this.ExtractMethodsFromCSharp(classBody)
                
                // Extract properties
                let propertyPattern = @"(public|private|protected|internal)?\s*(static|virtual|abstract|override|sealed)?\s*([a-zA-Z0-9_<>\.]+)\s+([a-zA-Z0-9_]+)\s*\{\s*(?:get;)?\s*(?:set;)?\s*\}"
                let propertyMatches = Regex.Matches(classBody, propertyPattern)
                
                let properties = 
                    propertyMatches
                    |> Seq.cast<Match>
                    |> Seq.map (fun m ->
                        let propertyType = m.Groups.[3].Value
                        let propertyName = m.Groups.[4].Value
                        (propertyName, propertyType)
                    )
                    |> Seq.toList
                
                {
                    Name = className
                    Namespace = namespaceName
                    Methods = methods
                    Properties = properties
                    Modifiers = modifiers |> Seq.toList
                    BaseClass = baseClass
                    Interfaces = interfaces
                }
            )
            |> Seq.toList
        with
        | ex ->
            logger.LogError(ex, "Error extracting classes from C# code")
            []
    
    /// <summary>
    /// Extracts classes from F# code.
    /// </summary>
    /// <param name="code">The code to extract classes from.</param>
    /// <returns>The list of extracted classes.</returns>
    member this.ExtractClassesFromFSharp(code: string) =
        try
            logger.LogInformation("Extracting classes from F# code")
            
            // Define regex pattern for types
            let typePattern = @"type\s+(?:private\s+)?([a-zA-Z0-9_]+)(?:<[^>]+>)?\s*(?:=\s*[^{]+)?\s*(?:\{([^}]*)\}|\([^)]*\)|=\s*([^t]*))"
            
            // Extract namespace and module information
            let namespacePattern = @"namespace\s+([a-zA-Z0-9_\.]+)"
            let modulePattern = @"module\s+(?:rec\s+)?([a-zA-Z0-9_\.]+)"
            
            let namespaceMatch = Regex.Match(code, namespacePattern)
            let moduleMatch = Regex.Match(code, modulePattern)
            
            let namespaceName = 
                if namespaceMatch.Success then 
                    namespaceMatch.Groups.[1].Value 
                elif moduleMatch.Success then 
                    moduleMatch.Groups.[1].Value 
                else 
                    "UnknownNamespace"
            
            // Find all types
            let typeMatches = Regex.Matches(code, typePattern, RegexOptions.Singleline)
            
            // Extract types
            typeMatches
            |> Seq.cast<Match>
            |> Seq.map (fun m ->
                let typeName = m.Groups.[1].Value
                let typeBody = 
                    if m.Groups.[2].Success then 
                        m.Groups.[2].Value 
                    elif m.Groups.[3].Success then 
                        m.Groups.[3].Value 
                    else 
                        ""
                
                // Extract methods
                let methods = this.ExtractMethodsFromFSharp(typeBody)
                
                // Extract properties
                let propertyPattern = @"member\s+(?:private\s+)?(?:this|self|_)\.([a-zA-Z0-9_]+)\s*(?::\s*([^=]+))?\s*with\s+(?:get|set)"
                let propertyMatches = Regex.Matches(typeBody, propertyPattern)
                
                let properties = 
                    propertyMatches
                    |> Seq.cast<Match>
                    |> Seq.map (fun m ->
                        let propertyName = m.Groups.[1].Value
                        let propertyType = if m.Groups.[2].Success then m.Groups.[2].Value else "unknown"
                        (propertyName, propertyType)
                    )
                    |> Seq.toList
                
                // Extract inheritance
                let inheritancePattern = @"inherit\s+([a-zA-Z0-9_<>\.]+)(?:\([^)]*\))?"
                let interfacePattern = @"interface\s+([a-zA-Z0-9_<>\.]+)"
                
                let inheritanceMatch = Regex.Match(typeBody, inheritancePattern)
                let interfaceMatches = Regex.Matches(typeBody, interfacePattern)
                
                let baseClass = if inheritanceMatch.Success then Some inheritanceMatch.Groups.[1].Value else None
                
                let interfaces = 
                    interfaceMatches
                    |> Seq.cast<Match>
                    |> Seq.map (fun m -> m.Groups.[1].Value)
                    |> Seq.toList
                
                {
                    Name = typeName
                    Namespace = namespaceName
                    Methods = methods
                    Properties = properties
                    Modifiers = []
                    BaseClass = baseClass
                    Interfaces = interfaces
                }
            )
            |> Seq.toList
        with
        | ex ->
            logger.LogError(ex, "Error extracting classes from F# code")
            []
    
    /// <summary>
    /// Analyzes code to generate test cases.
    /// </summary>
    /// <param name="code">The code to analyze.</param>
    /// <param name="language">The language of the code.</param>
    /// <returns>The list of extracted methods and classes.</returns>
    member this.AnalyzeCode(code: string, language: string) =
        try
            logger.LogInformation("Analyzing {Language} code for test generation", language)
            
            // Extract methods and classes based on language
            match language.ToLowerInvariant() with
            | "csharp" ->
                let methods = this.ExtractMethodsFromCSharp(code)
                let classes = this.ExtractClassesFromCSharp(code)
                methods, classes
            | "fsharp" ->
                let methods = this.ExtractMethodsFromFSharp(code)
                let classes = this.ExtractClassesFromFSharp(code)
                methods, classes
            | _ ->
                logger.LogWarning("Unsupported language: {Language}", language)
                [], []
        with
        | ex ->
            logger.LogError(ex, "Error analyzing code for test generation")
            [], []
