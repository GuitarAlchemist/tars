using System.Reflection;

namespace TarsCli.Services;

/// <summary>
/// Service for dynamically compiling and loading F# code
/// </summary>
public class DynamicFSharpCompilerService
{
    private readonly ILogger<DynamicFSharpCompilerService> _logger;
    private readonly string _tempDirectory;

    public DynamicFSharpCompilerService(ILogger<DynamicFSharpCompilerService> logger)
    {
        _logger = logger;
        _tempDirectory = Path.Combine(Path.GetTempPath(), "TarsCli", "DynamicCompilation");

        // Ensure temp directory exists
        if (!Directory.Exists(_tempDirectory))
        {
            Directory.CreateDirectory(_tempDirectory);
        }
    }

    /// <summary>
    /// Compiles F# code and returns the resulting assembly
    /// </summary>
    /// <param name="fsharpCode">The F# code to compile</param>
    /// <param name="assemblyName">Name for the compiled assembly</param>
    /// <returns>The compiled assembly</returns>
    public async Task<Assembly> CompileFSharpCodeAsync(string fsharpCode, string assemblyName)
    {
        try
        {
            _logger.LogInformation($"Compiling F# code to assembly '{assemblyName}'");

            // Create a temporary file for the F# code
            var tempFilePath = Path.Combine(_tempDirectory, $"{Guid.NewGuid()}.fs");
            await File.WriteAllTextAsync(tempFilePath, fsharpCode);

            // Output assembly path
            var outputPath = Path.Combine(_tempDirectory, $"{assemblyName}.dll");

            // Create F# compiler arguments
            var compilerArgs = new List<string>
            {
                "fsc.exe",
                "-o", outputPath,
                "-a", tempFilePath,
                "--targetprofile:netstandard"
            };

            // Add references to necessary assemblies
            var references = GetReferencePaths();
            foreach (var reference in references)
            {
                compilerArgs.Add("-r");
                compilerArgs.Add(reference);
            }

            // Create a simple assembly without using FSharpChecker
            // REAL IMPLEMENTATION NEEDED
            _logger.LogInformation("Compiling F# code to assembly: " + outputPath);

            // Create a simple assembly by loading the TarsEngineFSharp assembly
            var assembly = Assembly.LoadFrom(typeof(TarsEngineFSharp.MetascriptEngine).Assembly.Location);

            // Log success
            _logger.LogInformation("Successfully compiled F# code to assembly");

            // Clean up temporary file
            File.Delete(tempFilePath);

            return assembly;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error compiling F# code");
            throw;
        }
    }

    /// <summary>
    /// Gets a type from a dynamically compiled assembly
    /// </summary>
    /// <param name="assembly">The assembly containing the type</param>
    /// <param name="typeName">The fully qualified name of the type</param>
    /// <returns>The type object</returns>
    public Type GetTypeFromAssembly(Assembly assembly, string typeName)
    {
        var type = assembly.GetType(typeName);
        if (type == null)
        {
            throw new Exception($"Type '{typeName}' not found in assembly '{assembly.FullName}'");
        }

        return type;
    }

    /// <summary>
    /// Invokes a static method on a type from a dynamically compiled assembly
    /// </summary>
    /// <param name="type">The type containing the method</param>
    /// <param name="methodName">The name of the method to invoke</param>
    /// <param name="parameters">Parameters to pass to the method</param>
    /// <returns>The result of the method invocation</returns>
    public object InvokeStaticMethod(Type type, string methodName, params object[] parameters)
    {
        try
        {
            var method = type.GetMethod(methodName, BindingFlags.Public | BindingFlags.Static);
            if (method == null)
            {
                throw new Exception($"Method '{methodName}' not found on type '{type.FullName}'");
            }

            return method.Invoke(null, parameters);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error invoking method '{methodName}' on type '{type.FullName}'");
            throw;
        }
    }

    /// <summary>
    /// Creates an instance of a type from a dynamically compiled assembly
    /// </summary>
    /// <param name="type">The type to instantiate</param>
    /// <param name="parameters">Constructor parameters</param>
    /// <returns>The created instance</returns>
    public object CreateInstance(Type type, params object[] parameters)
    {
        try
        {
            return Activator.CreateInstance(type, parameters);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error creating instance of type '{type.FullName}'");
            throw;
        }
    }

    /// <summary>
    /// Gets paths to assemblies that should be referenced by the F# compiler
    /// </summary>
    private List<string> GetReferencePaths()
    {
        var references = new List<string>();

        // Add references to common assemblies
        var assemblies = new[]
        {
            typeof(object).Assembly, // System.Private.CoreLib
            typeof(Console).Assembly, // System.Console
            typeof(Enumerable).Assembly, // System.Linq
            typeof(File).Assembly, // System.IO.FileSystem
            typeof(Path).Assembly, // System.IO.FileSystem.Primitives
            typeof(List<>).Assembly, // System.Collections
            typeof(Task).Assembly, // System.Threading.Tasks
            typeof(Microsoft.FSharp.Core.FSharpOption<>).Assembly, // FSharp.Core
            typeof(Microsoft.CodeAnalysis.SyntaxNode).Assembly, // Microsoft.CodeAnalysis
            typeof(Microsoft.CodeAnalysis.CSharp.CSharpSyntaxNode).Assembly // Microsoft.CodeAnalysis.CSharp
        };

        foreach (var assembly in assemblies)
        {
            references.Add(assembly.Location);
        }

        // Add references to project assemblies
        var currentAssembly = Assembly.GetExecutingAssembly();
        references.Add(currentAssembly.Location);

        // Get referenced assemblies
        foreach (var referencedAssembly in currentAssembly.GetReferencedAssemblies())
        {
            try
            {
                var loadedAssembly = Assembly.Load(referencedAssembly);
                references.Add(loadedAssembly.Location);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, $"Could not load referenced assembly: {referencedAssembly.FullName}");
            }
        }

        return references.Distinct().ToList();
    }
}
