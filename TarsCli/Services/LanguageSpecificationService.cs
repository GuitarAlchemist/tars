using System.Text;
using Microsoft.Extensions.Configuration;

namespace TarsCli.Services;

/// <summary>
/// Service for generating language specifications and grammar definitions for TARS DSL
/// </summary>
public class LanguageSpecificationService
{
    private readonly ILogger<LanguageSpecificationService> _logger;
    private readonly IConfiguration _configuration;

    public LanguageSpecificationService(
        ILogger<LanguageSpecificationService> logger,
        IConfiguration configuration)
    {
        _logger = logger;
        _configuration = configuration;
    }

    /// <summary>
    /// Generate EBNF (Extended Backus-Naur Form) specification for TARS DSL
    /// </summary>
    public async Task<string> GenerateEbnfAsync()
    {
        _logger.LogInformation("Generating EBNF specification for TARS DSL");
            
        var ebnf = new StringBuilder();
            
        // Add header
        ebnf.AppendLine("(* TARS DSL - Extended Backus-Naur Form Specification *)");
        ebnf.AppendLine("(* Generated on: " + DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm:ss") + " UTC *)");
        ebnf.AppendLine();
            
        // Add top-level program structure
        ebnf.AppendLine("(* Top-level program structure *)");
        ebnf.AppendLine("<tars-program> ::= { <block> }");
        ebnf.AppendLine("<block> ::= <block-type> <block-name>? '{' <block-content> '}'");
        ebnf.AppendLine("<block-type> ::= 'CONFIG' | 'PROMPT' | 'ACTION' | 'TASK' | 'AGENT' | 'AUTO_IMPROVE' | 'DATA' | 'TOOLING'");
        ebnf.AppendLine("<block-name> ::= <identifier>");
        ebnf.AppendLine("<block-content> ::= { <property> | <statement> | <block> }");
        ebnf.AppendLine();
            
        // Add property definitions
        ebnf.AppendLine("(* Property definitions *)");
        ebnf.AppendLine("<property> ::= <identifier> ':' <value> ';'?");
        ebnf.AppendLine("<value> ::= <string> | <number> | <boolean> | <array> | <object> | <identifier>");
        ebnf.AppendLine("<string> ::= '\"' { <any-character-except-double-quote> | '\\\"' } '\"'");
        ebnf.AppendLine("<number> ::= <integer> | <float>");
        ebnf.AppendLine("<integer> ::= ['-'] <digit> { <digit> }");
        ebnf.AppendLine("<float> ::= <integer> '.' <digit> { <digit> }");
        ebnf.AppendLine("<boolean> ::= 'true' | 'false'");
        ebnf.AppendLine("<array> ::= '[' [ <value> { ',' <value> } ] ']'");
        ebnf.AppendLine("<object> ::= '{' [ <property> { ',' <property> } ] '}'");
        ebnf.AppendLine();
            
        // Add statement definitions
        ebnf.AppendLine("(* Statement definitions *)");
        ebnf.AppendLine("<statement> ::= <assignment> | <function-call> | <control-flow> | <return-statement>");
        ebnf.AppendLine("<assignment> ::= ['let'] <identifier> ['=' | ':='] <expression> ';'?");
        ebnf.AppendLine("<function-call> ::= <identifier> '(' [ <expression> { ',' <expression> } ] ')' ';'?");
        ebnf.AppendLine("<control-flow> ::= <if-statement> | <for-loop> | <while-loop>");
        ebnf.AppendLine("<if-statement> ::= 'if' '(' <expression> ')' '{' <block-content> '}' [ 'else' '{' <block-content> '}' ]");
        ebnf.AppendLine("<for-loop> ::= 'for' '(' <assignment> ';' <expression> ';' <expression> ')' '{' <block-content> '}'");
        ebnf.AppendLine("<while-loop> ::= 'while' '(' <expression> ')' '{' <block-content> '}'");
        ebnf.AppendLine("<return-statement> ::= 'return' <expression>? ';'?");
        ebnf.AppendLine();
            
        // Add expression definitions
        ebnf.AppendLine("(* Expression definitions *)");
        ebnf.AppendLine("<expression> ::= <value> | <identifier> | <function-call> | <binary-expression> | '(' <expression> ')'");
        ebnf.AppendLine("<binary-expression> ::= <expression> <operator> <expression>");
        ebnf.AppendLine("<operator> ::= '+' | '-' | '*' | '/' | '%' | '==' | '!=' | '<' | '>' | '<=' | '>=' | '&&' | '||'");
        ebnf.AppendLine();
            
        // Add identifier definition
        ebnf.AppendLine("(* Identifier definition *)");
        ebnf.AppendLine("<identifier> ::= <letter> { <letter> | <digit> | '_' }");
        ebnf.AppendLine("<letter> ::= 'A' | 'B' | ... | 'Z' | 'a' | 'b' | ... | 'z'");
        ebnf.AppendLine("<digit> ::= '0' | '1' | ... | '9'");
        ebnf.AppendLine();
            
        // Add specific block definitions
        ebnf.AppendLine("(* Specific block definitions *)");
        ebnf.AppendLine("<config-block> ::= 'CONFIG' '{' { <config-property> } '}'");
        ebnf.AppendLine("<config-property> ::= 'version' ':' <string> ';'? | 'author' ':' <string> ';'? | 'description' ':' <string> ';'?");
        ebnf.AppendLine();
            
        ebnf.AppendLine("<prompt-block> ::= 'PROMPT' [<identifier>] '{' <prompt-content> '}'");
        ebnf.AppendLine("<prompt-content> ::= <string> | { <property> }");
        ebnf.AppendLine();
            
        ebnf.AppendLine("<action-block> ::= 'ACTION' [<identifier>] '{' { <statement> } '}'");
        ebnf.AppendLine();
            
        ebnf.AppendLine("<task-block> ::= 'TASK' [<identifier>] '{' { <property> | <action-block> } '}'");
        ebnf.AppendLine();
            
        ebnf.AppendLine("<agent-block> ::= 'AGENT' [<identifier>] '{' { <property> | <task-block> | <communication-block> } '}'");
        ebnf.AppendLine("<communication-block> ::= 'COMMUNICATION' '{' { <property> } '}'");
        ebnf.AppendLine();
            
        ebnf.AppendLine("<auto-improve-block> ::= 'AUTO_IMPROVE' [<identifier>] '{' { <property> | <statement> } '}'");
        ebnf.AppendLine();
            
        ebnf.AppendLine("<data-block> ::= 'DATA' '{' { <assignment> | <statement> } '}'");
        ebnf.AppendLine();
            
        ebnf.AppendLine("<tooling-block> ::= 'TOOLING' '{' { <generate-grammar-block> | <diagnostics-block> | <instrumentation-block> } '}'");
        ebnf.AppendLine("<generate-grammar-block> ::= 'GENERATE_GRAMMAR' '{' { <property> } '}'");
        ebnf.AppendLine("<diagnostics-block> ::= 'DIAGNOSTICS' '{' { <property> } '}'");
        ebnf.AppendLine("<instrumentation-block> ::= 'INSTRUMENTATION' '{' { <property> } '}'");
            
        return ebnf.ToString();
    }

    /// <summary>
    /// Generate BNF (Backus-Naur Form) specification for TARS DSL
    /// </summary>
    public async Task<string> GenerateBnfAsync()
    {
        _logger.LogInformation("Generating BNF specification for TARS DSL");
            
        var bnf = new StringBuilder();
            
        // Add header
        bnf.AppendLine("# TARS DSL - Backus-Naur Form Specification");
        bnf.AppendLine("# Generated on: " + DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm:ss") + " UTC");
        bnf.AppendLine();
            
        // Add top-level program structure
        bnf.AppendLine("# Top-level program structure");
        bnf.AppendLine("<tars-program> ::= <block> | <tars-program> <block>");
        bnf.AppendLine("<block> ::= <block-type> <block-name> \"{\" <block-content> \"}\" | <block-type> \"{\" <block-content> \"}\"");
        bnf.AppendLine("<block-type> ::= \"CONFIG\" | \"PROMPT\" | \"ACTION\" | \"TASK\" | \"AGENT\" | \"AUTO_IMPROVE\" | \"DATA\" | \"TOOLING\"");
        bnf.AppendLine("<block-name> ::= <identifier>");
        bnf.AppendLine("<block-content> ::= <property> | <statement> | <block> | <block-content> <property> | <block-content> <statement> | <block-content> <block>");
        bnf.AppendLine();
            
        // Add property definitions
        bnf.AppendLine("# Property definitions");
        bnf.AppendLine("<property> ::= <identifier> \":\" <value> \";\" | <identifier> \":\" <value>");
        bnf.AppendLine("<value> ::= <string> | <number> | <boolean> | <array> | <object> | <identifier>");
        bnf.AppendLine("<string> ::= \"\\\"\" <string-content> \"\\\"\"");
        bnf.AppendLine("<string-content> ::= <empty> | <character> | <string-content> <character>");
        bnf.AppendLine("<number> ::= <integer> | <float>");
        bnf.AppendLine("<integer> ::= <digit> | <integer> <digit>");
        bnf.AppendLine("<float> ::= <integer> \".\" <integer>");
        bnf.AppendLine("<boolean> ::= \"true\" | \"false\"");
        bnf.AppendLine("<array> ::= \"[\" <array-content> \"]\" | \"[\" \"]\"");
        bnf.AppendLine("<array-content> ::= <value> | <array-content> \",\" <value>");
        bnf.AppendLine("<object> ::= \"{\" <object-content> \"}\" | \"{\" \"}\"");
        bnf.AppendLine("<object-content> ::= <property> | <object-content> \",\" <property>");
        bnf.AppendLine();
            
        // Add statement definitions
        bnf.AppendLine("# Statement definitions");
        bnf.AppendLine("<statement> ::= <assignment> | <function-call> | <control-flow> | <return-statement>");
        bnf.AppendLine("<assignment> ::= \"let\" <identifier> \"=\" <expression> \";\" | <identifier> \"=\" <expression> \";\" | \"let\" <identifier> \"=\" <expression> | <identifier> \"=\" <expression>");
        bnf.AppendLine("<function-call> ::= <identifier> \"(\" <argument-list> \")\" \";\" | <identifier> \"(\" \")\" \";\" | <identifier> \"(\" <argument-list> \")\" | <identifier> \"(\" \")\"");
        bnf.AppendLine("<argument-list> ::= <expression> | <argument-list> \",\" <expression>");
        bnf.AppendLine("<control-flow> ::= <if-statement> | <for-loop> | <while-loop>");
        bnf.AppendLine("<if-statement> ::= \"if\" \"(\" <expression> \")\" \"{\" <block-content> \"}\" | \"if\" \"(\" <expression> \")\" \"{\" <block-content> \"}\" \"else\" \"{\" <block-content> \"}\"");
        bnf.AppendLine("<for-loop> ::= \"for\" \"(\" <assignment> \";\" <expression> \";\" <expression> \")\" \"{\" <block-content> \"}\"");
        bnf.AppendLine("<while-loop> ::= \"while\" \"(\" <expression> \")\" \"{\" <block-content> \"}\"");
        bnf.AppendLine("<return-statement> ::= \"return\" <expression> \";\" | \"return\" \";\" | \"return\" <expression> | \"return\"");
        bnf.AppendLine();
            
        // Add expression definitions
        bnf.AppendLine("# Expression definitions");
        bnf.AppendLine("<expression> ::= <value> | <identifier> | <function-call> | <binary-expression> | \"(\" <expression> \")\"");
        bnf.AppendLine("<binary-expression> ::= <expression> <operator> <expression>");
        bnf.AppendLine("<operator> ::= \"+\" | \"-\" | \"*\" | \"/\" | \"%\" | \"==\" | \"!=\" | \"<\" | \">\" | \"<=\" | \">=\" | \"&&\" | \"||\"");
        bnf.AppendLine();
            
        // Add identifier definition
        bnf.AppendLine("# Identifier definition");
        bnf.AppendLine("<identifier> ::= <letter> | <identifier> <letter> | <identifier> <digit> | <identifier> \"_\"");
        bnf.AppendLine("<letter> ::= \"A\" | \"B\" | ... | \"Z\" | \"a\" | \"b\" | ... | \"z\"");
        bnf.AppendLine("<digit> ::= \"0\" | \"1\" | ... | \"9\"");
        bnf.AppendLine("<empty> ::= ");
        bnf.AppendLine("<character> ::= <letter> | <digit> | <special-character>");
        bnf.AppendLine("<special-character> ::= \" \" | \"!\" | \"#\" | \"$\" | ... | \"{\" | \"|\" | \"}\" | \"~\"");
            
        return bnf.ToString();
    }

    /// <summary>
    /// Generate a JSON schema for TARS DSL
    /// </summary>
    public async Task<string> GenerateJsonSchemaAsync()
    {
        _logger.LogInformation("Generating JSON schema for TARS DSL");
            
        var schema = @"{
  ""$schema"": ""http://json-schema.org/draft-07/schema#"",
  ""title"": ""TARS DSL Schema"",
  ""description"": ""JSON Schema for TARS DSL"",
  ""type"": ""object"",
  ""properties"": {
    ""blocks"": {
      ""type"": ""array"",
      ""description"": ""List of blocks in the TARS program"",
      ""items"": {
        ""type"": ""object"",
        ""properties"": {
          ""type"": {
            ""type"": ""string"",
            ""enum"": [""CONFIG"", ""PROMPT"", ""ACTION"", ""TASK"", ""AGENT"", ""AUTO_IMPROVE"", ""DATA"", ""TOOLING""],
            ""description"": ""Type of the block""
          },
          ""name"": {
            ""type"": ""string"",
            ""description"": ""Optional name of the block""
          },
          ""content"": {
            ""type"": ""object"",
            ""description"": ""Content of the block"",
            ""additionalProperties"": true
          }
        },
        ""required"": [""type"", ""content""]
      }
    }
  },
  ""required"": [""blocks""]
}";
            
        return schema;
    }

    /// <summary>
    /// Generate a markdown documentation for TARS DSL
    /// </summary>
    public async Task<string> GenerateMarkdownDocumentationAsync()
    {
        _logger.LogInformation("Generating markdown documentation for TARS DSL");
            
        var markdown = new StringBuilder();
            
        // Add header
        markdown.AppendLine("# TARS DSL Documentation");
        markdown.AppendLine();
        markdown.AppendLine("*Generated on: " + DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm:ss") + " UTC*");
        markdown.AppendLine();
            
        // Add introduction
        markdown.AppendLine("## Introduction");
        markdown.AppendLine();
        markdown.AppendLine("TARS DSL (Domain Specific Language) is a language designed for defining AI workflows, agent behaviors, and self-improvement processes. It provides a structured way to define prompts, actions, tasks, and agents within the TARS system.");
        markdown.AppendLine();
            
        // Add syntax overview
        markdown.AppendLine("## Syntax Overview");
        markdown.AppendLine();
        markdown.AppendLine("TARS DSL uses a block-based syntax with curly braces. Each block has a type, an optional name, and content. The content can include properties, statements, and nested blocks.");
        markdown.AppendLine();
        markdown.AppendLine("```");
        markdown.AppendLine("BLOCK_TYPE [name] {");
        markdown.AppendLine("    property1: value1;");
        markdown.AppendLine("    property2: value2;");
        markdown.AppendLine("    ");
        markdown.AppendLine("    NESTED_BLOCK {");
        markdown.AppendLine("        nestedProperty: nestedValue;");
        markdown.AppendLine("    }");
        markdown.AppendLine("}");
        markdown.AppendLine("```");
        markdown.AppendLine();
            
        // Add block types
        markdown.AppendLine("## Block Types");
        markdown.AppendLine();
            
        markdown.AppendLine("### CONFIG");
        markdown.AppendLine();
        markdown.AppendLine("The CONFIG block defines configuration settings for the TARS program.");
        markdown.AppendLine();
        markdown.AppendLine("```");
        markdown.AppendLine("CONFIG {");
        markdown.AppendLine("    version: \"1.0\";");
        markdown.AppendLine("    author: \"John Doe\";");
        markdown.AppendLine("    description: \"Example TARS program\";");
        markdown.AppendLine("}");
        markdown.AppendLine("```");
        markdown.AppendLine();
            
        markdown.AppendLine("### PROMPT");
        markdown.AppendLine();
        markdown.AppendLine("The PROMPT block defines a prompt to be sent to an AI model.");
        markdown.AppendLine();
        markdown.AppendLine("```");
        markdown.AppendLine("PROMPT {");
        markdown.AppendLine("    text: \"Generate a list of 5 ideas for improving code quality.\";");
        markdown.AppendLine("    model: \"gpt-4\";");
        markdown.AppendLine("    temperature: 0.7;");
        markdown.AppendLine("}");
        markdown.AppendLine("```");
        markdown.AppendLine();
            
        markdown.AppendLine("### ACTION");
        markdown.AppendLine();
        markdown.AppendLine("The ACTION block defines a set of actions to be performed.");
        markdown.AppendLine();
        markdown.AppendLine("```");
        markdown.AppendLine("ACTION {");
        markdown.AppendLine("    let result = processFile(\"example.cs\");");
        markdown.AppendLine("    print(result);");
        markdown.AppendLine("}");
        markdown.AppendLine("```");
        markdown.AppendLine();
            
        markdown.AppendLine("### TASK");
        markdown.AppendLine();
        markdown.AppendLine("The TASK block defines a task to be performed, which can include properties and actions.");
        markdown.AppendLine();
        markdown.AppendLine("```");
        markdown.AppendLine("TASK {");
        markdown.AppendLine("    id: \"task_001\";");
        markdown.AppendLine("    description: \"Process a file and print the result\";");
        markdown.AppendLine("    ");
        markdown.AppendLine("    ACTION {");
        markdown.AppendLine("        let result = processFile(\"example.cs\");");
        markdown.AppendLine("        print(result);");
        markdown.AppendLine("    }");
        markdown.AppendLine("}");
        markdown.AppendLine("```");
        markdown.AppendLine();
            
        markdown.AppendLine("### AGENT");
        markdown.AppendLine();
        markdown.AppendLine("The AGENT block defines an AI agent with capabilities, tasks, and communication settings.");
        markdown.AppendLine();
        markdown.AppendLine("```");
        markdown.AppendLine("AGENT {");
        markdown.AppendLine("    id: \"agent_001\";");
        markdown.AppendLine("    name: \"CodeAnalyzer\";");
        markdown.AppendLine("    capabilities: [\"code-analysis\", \"refactoring\"];");
        markdown.AppendLine("    ");
        markdown.AppendLine("    TASK {");
        markdown.AppendLine("        id: \"task_001\";");
        markdown.AppendLine("        description: \"Analyze code quality\";");
        markdown.AppendLine("    }");
        markdown.AppendLine("    ");
        markdown.AppendLine("    COMMUNICATION {");
        markdown.AppendLine("        protocol: \"HTTP\";");
        markdown.AppendLine("        endpoint: \"http://localhost:8080\";");
        markdown.AppendLine("    }");
        markdown.AppendLine("}");
        markdown.AppendLine("```");
        markdown.AppendLine();
            
        markdown.AppendLine("### AUTO_IMPROVE");
        markdown.AppendLine();
        markdown.AppendLine("The AUTO_IMPROVE block defines settings and actions for self-improvement processes.");
        markdown.AppendLine();
        markdown.AppendLine("```");
        markdown.AppendLine("AUTO_IMPROVE {");
        markdown.AppendLine("    target: \"code_quality\";");
        markdown.AppendLine("    frequency: \"daily\";");
        markdown.AppendLine("    ");
        markdown.AppendLine("    ACTION {");
        markdown.AppendLine("        let files = findFiles(\"*.cs\");");
        markdown.AppendLine("        foreach(file in files) {");
        markdown.AppendLine("            analyzeAndImprove(file);");
        markdown.AppendLine("        }");
        markdown.AppendLine("    }");
        markdown.AppendLine("}");
        markdown.AppendLine("```");
        markdown.AppendLine();
            
        markdown.AppendLine("### DATA");
        markdown.AppendLine();
        markdown.AppendLine("The DATA block defines data sources and operations.");
        markdown.AppendLine();
        markdown.AppendLine("```");
        markdown.AppendLine("DATA {");
        markdown.AppendLine("    let fileData = FILE(\"data/sample.csv\");");
        markdown.AppendLine("    let apiData = API(\"https://api.example.com/data\");");
        markdown.AppendLine("    let combined = combineData(fileData, apiData);");
        markdown.AppendLine("}");
        markdown.AppendLine("```");
        markdown.AppendLine();
            
        markdown.AppendLine("### TOOLING");
        markdown.AppendLine();
        markdown.AppendLine("The TOOLING block defines tools and utilities for working with TARS DSL.");
        markdown.AppendLine();
        markdown.AppendLine("```");
        markdown.AppendLine("TOOLING {");
        markdown.AppendLine("    GENERATE_GRAMMAR {");
        markdown.AppendLine("        format: \"BNF\";");
        markdown.AppendLine("        output: \"tars_grammar.bnf\";");
        markdown.AppendLine("    }");
        markdown.AppendLine("    ");
        markdown.AppendLine("    DIAGNOSTICS {");
        markdown.AppendLine("        level: \"detailed\";");
        markdown.AppendLine("        output: \"diagnostics.log\";");
        markdown.AppendLine("    }");
        markdown.AppendLine("}");
        markdown.AppendLine("```");
        markdown.AppendLine();
            
        // Add examples
        markdown.AppendLine("## Complete Examples");
        markdown.AppendLine();
            
        markdown.AppendLine("### Example 1: Simple Code Analysis");
        markdown.AppendLine();
        markdown.AppendLine("```");
        markdown.AppendLine("CONFIG {");
        markdown.AppendLine("    version: \"1.0\";");
        markdown.AppendLine("    description: \"Simple code analysis example\";");
        markdown.AppendLine("}");
        markdown.AppendLine("");
        markdown.AppendLine("PROMPT {");
        markdown.AppendLine("    text: \"Analyze the following code for potential improvements:\";");
        markdown.AppendLine("    model: \"gpt-4\";");
        markdown.AppendLine("}");
        markdown.AppendLine("");
        markdown.AppendLine("ACTION {");
        markdown.AppendLine("    let code = readFile(\"example.cs\");");
        markdown.AppendLine("    let analysis = analyzeCode(code);");
        markdown.AppendLine("    print(analysis);");
        markdown.AppendLine("}");
        markdown.AppendLine("```");
        markdown.AppendLine();
            
        markdown.AppendLine("### Example 2: Agent-Based Workflow");
        markdown.AppendLine();
        markdown.AppendLine("```");
        markdown.AppendLine("CONFIG {");
        markdown.AppendLine("    version: \"1.0\";");
        markdown.AppendLine("    description: \"Agent-based workflow example\";");
        markdown.AppendLine("}");
        markdown.AppendLine("");
        markdown.AppendLine("AGENT CodeAnalyzer {");
        markdown.AppendLine("    capabilities: [\"code-analysis\"];");
        markdown.AppendLine("    ");
        markdown.AppendLine("    TASK AnalyzeCode {");
        markdown.AppendLine("        description: \"Analyze code quality\";");
        markdown.AppendLine("        ");
        markdown.AppendLine("        ACTION {");
        markdown.AppendLine("            let code = readFile(\"example.cs\");");
        markdown.AppendLine("            return analyzeCode(code);");
        markdown.AppendLine("        }");
        markdown.AppendLine("    }");
        markdown.AppendLine("}");
        markdown.AppendLine("");
        markdown.AppendLine("AGENT CodeRefactorer {");
        markdown.AppendLine("    capabilities: [\"refactoring\"];");
        markdown.AppendLine("    ");
        markdown.AppendLine("    TASK RefactorCode {");
        markdown.AppendLine("        description: \"Refactor code based on analysis\";");
        markdown.AppendLine("        ");
        markdown.AppendLine("        ACTION {");
        markdown.AppendLine("            let analysis = getTaskResult(\"AnalyzeCode\");");
        markdown.AppendLine("            let code = readFile(\"example.cs\");");
        markdown.AppendLine("            let refactored = refactorCode(code, analysis);");
        markdown.AppendLine("            writeFile(\"example_refactored.cs\", refactored);");
        markdown.AppendLine("        }");
        markdown.AppendLine("    }");
        markdown.AppendLine("}");
        markdown.AppendLine("```");
            
        return markdown.ToString();
    }

    /// <summary>
    /// Save a language specification to a file
    /// </summary>
    public async Task<bool> SaveSpecificationToFileAsync(string content, string filePath)
    {
        try
        {
            _logger.LogInformation($"Saving specification to file: {filePath}");
                
            // Create directory if it doesn't exist
            var directory = Path.GetDirectoryName(filePath);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }
                
            // Write content to file
            await File.WriteAllTextAsync(filePath, content);
                
            _logger.LogInformation($"Specification saved to file: {filePath}");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error saving specification to file: {filePath}");
            return false;
        }
    }
}