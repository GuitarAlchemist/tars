namespace TarsEngine.FSharp.Core.Tests.Analysis

open System
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open Xunit
open TarsEngine.FSharp.Core.Analysis

/// <summary>
/// Tests for the PatternMatcherService class.
/// </summary>
module PatternMatcherServiceTests =
    
    /// <summary>
    /// Mock logger for testing.
    /// </summary>
    type MockLogger<'T>() =
        interface ILogger<'T> with
            member _.Log<'TState>(logLevel, eventId, state, ex, formatter) =
                // Do nothing
                ()
            
            member _.IsEnabled(logLevel) = true
            
            member _.BeginScope<'TState>(state) =
                { new IDisposable with
                    member _.Dispose() = ()
                }
    
    /// <summary>
    /// Test that the pattern matcher can find patterns in C# code.
    /// </summary>
    [<Fact>]
    let ``PatternMatcherService can find patterns in C# code``() = task {
        // Arrange
        let logger = MockLogger<PatternMatcherService>() :> ILogger<PatternMatcherService>
        
        // Define some patterns
        let patterns = [
            {
                Name = "Singleton Pattern"
                Description = "A design pattern that restricts the instantiation of a class to one object"
                Language = "csharp"
                Template = @"private\s+static\s+[a-zA-Z0-9_<>]+\s+instance\s*;\s*private\s+[a-zA-Z0-9_]+\s*\(\s*\)"
                Category = "Design Patterns"
                Tags = ["singleton"; "design pattern"; "creational"]
                AdditionalInfo = Map.empty
            }
            {
                Name = "Factory Method Pattern"
                Description = "A creational pattern that uses factory methods to deal with the problem of creating objects"
                Language = "csharp"
                Template = @"public\s+static\s+[a-zA-Z0-9_<>]+\s+Create[a-zA-Z0-9_]*\s*\("
                Category = "Design Patterns"
                Tags = ["factory"; "design pattern"; "creational"]
                AdditionalInfo = Map.empty
            }
        ]
        
        let service = PatternMatcherService(logger, patterns)
        
        let code = """
            using System;

            namespace DesignPatterns
            {
                public class Singleton
                {
                    private static Singleton instance;
                    private Singleton()
                    {
                        // Private constructor
                    }
                    
                    public static Singleton Instance
                    {
                        get
                        {
                            if (instance == null)
                            {
                                instance = new Singleton();
                            }
                            return instance;
                        }
                    }
                }
                
                public class Factory
                {
                    public static IProduct CreateProduct(string type)
                    {
                        switch (type)
                        {
                            case "A":
                                return new ProductA();
                            case "B":
                                return new ProductB();
                            default:
                                throw new ArgumentException("Invalid product type");
                        }
                    }
                }
                
                public interface IProduct
                {
                    void Operation();
                }
                
                public class ProductA : IProduct
                {
                    public void Operation()
                    {
                        Console.WriteLine("Product A");
                    }
                }
                
                public class ProductB : IProduct
                {
                    public void Operation()
                    {
                        Console.WriteLine("Product B");
                    }
                }
            }
            """
        
        // Act
        let! matches = service.FindPatternsAsync(code, "csharp")
        
        // Assert
        Assert.NotEmpty(matches)
        Assert.Contains(matches, fun m -> m.PatternName = "Singleton Pattern")
        Assert.Contains(matches, fun m -> m.PatternName = "Factory Method Pattern")
    }
    
    /// <summary>
    /// Test that the pattern matcher can find patterns in F# code.
    /// </summary>
    [<Fact>]
    let ``PatternMatcherService can find patterns in F# code``() = task {
        // Arrange
        let logger = MockLogger<PatternMatcherService>() :> ILogger<PatternMatcherService>
        
        // Define some patterns
        let patterns = [
            {
                Name = "Active Pattern"
                Description = "A pattern that transforms data into a form that is easier to match"
                Language = "fsharp"
                Template = @"let\s+\(\|[a-zA-Z0-9_]+\|\)\s+"
                Category = "F# Patterns"
                Tags = ["active pattern"; "pattern matching"; "f#"]
                AdditionalInfo = Map.empty
            }
            {
                Name = "Railway Oriented Programming"
                Description = "A functional approach to error handling"
                Language = "fsharp"
                Template = @"let\s+bind\s+f\s+result\s+=\s+match\s+result\s+with\s+\|\s+Ok\s+[a-zA-Z0-9_]+\s+->\s+f\s+[a-zA-Z0-9_]+\s+\|\s+Error\s+[a-zA-Z0-9_]+\s+->\s+Error\s+[a-zA-Z0-9_]+"
                Category = "F# Patterns"
                Tags = ["railway"; "error handling"; "f#"]
                AdditionalInfo = Map.empty
            }
        ]
        
        let service = PatternMatcherService(logger, patterns)
        
        let code = """
            module Patterns

            // Active pattern
            let (|Even|Odd|) n = if n % 2 = 0 then Even else Odd

            let testNumber n =
                match n with
                | Even -> printfn "%d is even" n
                | Odd -> printfn "%d is odd" n

            // Railway oriented programming
            type Result<'a, 'b> =
                | Ok of 'a
                | Error of 'b

            let bind f result =
                match result with
                | Ok x -> f x
                | Error e -> Error e

            let (>>=) result f = bind f result

            let validate1 x =
                if x > 0 then Ok x
                else Error "Value must be positive"

            let validate2 x =
                if x < 100 then Ok x
                else Error "Value must be less than 100"

            let validateBoth x =
                validate1 x >>= validate2
            """
        
        // Act
        let! matches = service.FindPatternsAsync(code, "fsharp")
        
        // Assert
        Assert.NotEmpty(matches)
        Assert.Contains(matches, fun m -> m.PatternName = "Active Pattern")
        Assert.Contains(matches, fun m -> m.PatternName = "Railway Oriented Programming")
    }
    
    /// <summary>
    /// Test that the pattern matcher can calculate similarity between code snippets.
    /// </summary>
    [<Fact>]
    let ``PatternMatcherService can calculate similarity between code snippets``() = task {
        // Arrange
        let logger = MockLogger<PatternMatcherService>() :> ILogger<PatternMatcherService>
        let service = PatternMatcherService(logger, [])
        
        let source = """
            public class Singleton
            {
                private static Singleton instance;
                private Singleton() {}
                
                public static Singleton Instance
                {
                    get
                    {
                        if (instance == null)
                        {
                            instance = new Singleton();
                        }
                        return instance;
                    }
                }
            }
            """
        
        let target = """
            public class MySingleton
            {
                private static MySingleton instance;
                private MySingleton() {}
                
                public static MySingleton GetInstance()
                {
                    if (instance == null)
                    {
                        instance = new MySingleton();
                    }
                    return instance;
                }
            }
            """
        
        // Act
        let! similarity = service.CalculateSimilarityAsync(source, target, "csharp")
        
        // Assert
        Assert.True(similarity > 0.7, $"Similarity should be high, but was {similarity}")
    }
    
    /// <summary>
    /// Test that the pattern matcher can find similar patterns.
    /// </summary>
    [<Fact>]
    let ``PatternMatcherService can find similar patterns``() = task {
        // Arrange
        let logger = MockLogger<PatternMatcherService>() :> ILogger<PatternMatcherService>
        
        // Define some patterns
        let patterns = [
            {
                Name = "Singleton Pattern"
                Description = "A design pattern that restricts the instantiation of a class to one object"
                Language = "csharp"
                Template = """
                    public class Singleton
                    {
                        private static Singleton instance;
                        private Singleton() {}
                        
                        public static Singleton Instance
                        {
                            get
                            {
                                if (instance == null)
                                {
                                    instance = new Singleton();
                                }
                                return instance;
                            }
                        }
                    }
                    """
                Category = "Design Patterns"
                Tags = ["singleton"; "design pattern"; "creational"]
                AdditionalInfo = Map.empty
            }
            {
                Name = "Factory Method Pattern"
                Description = "A creational pattern that uses factory methods to deal with the problem of creating objects"
                Language = "csharp"
                Template = """
                    public class Factory
                    {
                        public static IProduct CreateProduct(string type)
                        {
                            switch (type)
                            {
                                case "A":
                                    return new ProductA();
                                case "B":
                                    return new ProductB();
                                default:
                                    throw new ArgumentException("Invalid product type");
                            }
                        }
                    }
                    """
                Category = "Design Patterns"
                Tags = ["factory"; "design pattern"; "creational"]
                AdditionalInfo = Map.empty
            }
        ]
        
        let service = PatternMatcherService(logger, patterns)
        
        let code = """
            public class MySingleton
            {
                private static MySingleton instance;
                private MySingleton() {}
                
                public static MySingleton GetInstance()
                {
                    if (instance == null)
                    {
                        instance = new MySingleton();
                    }
                    return instance;
                }
            }
            """
        
        // Act
        let! similarPatterns = service.FindSimilarPatternsAsync(code, "csharp", 0.7)
        
        // Assert
        Assert.NotEmpty(similarPatterns)
        Assert.Contains(similarPatterns, fun (p, _) -> p.Name = "Singleton Pattern")
        
        // The singleton pattern should be more similar than the factory pattern
        let singletonSimilarity = similarPatterns |> List.find (fun (p, _) -> p.Name = "Singleton Pattern") |> snd
        let factorySimilarity = similarPatterns |> List.tryFind (fun (p, _) -> p.Name = "Factory Method Pattern") |> Option.map snd
        
        match factorySimilarity with
        | Some similarity -> Assert.True(singletonSimilarity > similarity)
        | None -> () // Factory pattern might not be in the results if similarity is too low
    }
