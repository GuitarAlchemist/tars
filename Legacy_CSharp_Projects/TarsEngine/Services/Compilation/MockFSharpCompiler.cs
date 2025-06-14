﻿using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Microsoft.CodeAnalysis.Scripting;

namespace TarsEngine.Services.Compilation
{
    /// <summary>
    /// REAL IMPLEMENTATION NEEDED
    /// </summary>
    public class MockFSharpCompiler : IFSharpCompiler
    {
        /// <inheritdoc/>
        public Task<CompilationResult> CompileAsync(string code, ScriptOptions options)
        {
            // Return a successful compilation result
            return Task.FromResult(new CompilationResult
            {
                Diagnostics = new List<CompilationDiagnostic>()
            });
        }
    }
}

