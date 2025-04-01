using System;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.DSL;
using TarsCli.Common;

namespace TarsCli.Services
{
    /// <summary>
    /// Service for working with TARS DSL files
    /// </summary>
    public class DslService
    {
        private readonly ILogger<DslService> _logger;
        private readonly OllamaService _ollamaService;
        private readonly TemplateService _templateService;

        public DslService(
            ILogger<DslService> logger,
            OllamaService ollamaService,
            TemplateService templateService)
        {
            _logger = logger;
            _ollamaService = ollamaService;
            _templateService = templateService;
        }

        /// <summary>
        /// Run a TARS DSL file
        /// </summary>
        /// <param name="filePath">The path to the DSL file</param>
        /// <param name="verbose">Whether to show verbose output</param>
        /// <returns>The exit code</returns>
        public async Task<int> RunDslFileAsync(string filePath, bool verbose)
        {
            try
            {
                _logger.LogInformation($"Running DSL file: {filePath}");

                if (!File.Exists(filePath))
                {
                    _logger.LogError($"File not found: {filePath}");
                    return 1;
                }

                // Parse the DSL file
                var program = TarsEngine.DSL.Parser.parseFile(filePath);

                if (verbose)
                {
                    _logger.LogInformation($"Parsed {program.Blocks.Length} blocks from the DSL file");
                    foreach (var block in program.Blocks)
                    {
                        _logger.LogInformation($"Block type: {block.Type}");
                    }
                }

                // Initialize the agent registry
                var registry = new TarsEngine.DSL.AgentFramework.AgentRegistry();
                TarsEngine.DSL.Interpreter.setAgentRegistry(registry);

                // Execute the program
                var result = TarsEngine.DSL.Interpreter.execute(program);

                switch (result)
                {
                    case TarsEngine.DSL.Interpreter.ExecutionResult.Success value:
                        _logger.LogInformation($"Program executed successfully: {value.Item}");
                        return 0;
                    case TarsEngine.DSL.Interpreter.ExecutionResult.Error error:
                        _logger.LogError($"Error executing program: {error.Item}");
                        return 1;
                    default:
                        _logger.LogError("Unknown execution result");
                        return 1;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error running DSL file: {ex.Message}");
                return 1;
            }
        }

        /// <summary>
        /// Validate a TARS DSL file
        /// </summary>
        /// <param name="filePath">The path to the DSL file</param>
        /// <returns>The exit code</returns>
        public async Task<int> ValidateDslFileAsync(string filePath)
        {
            try
            {
                _logger.LogInformation($"Validating DSL file: {filePath}");

                if (!File.Exists(filePath))
                {
                    _logger.LogError($"File not found: {filePath}");
                    return 1;
                }

                // Parse the DSL file
                var program = TarsEngine.DSL.Parser.parseFile(filePath);

                _logger.LogInformation($"DSL file is valid. Contains {program.Blocks.Length} blocks.");

                return 0;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error validating DSL file: {ex.Message}");
                return 1;
            }
        }

        /// <summary>
        /// Generate a TARS DSL file template
        /// </summary>
        /// <param name="outputPath">The path to save the generated DSL file</param>
        /// <param name="templateName">The template to use</param>
        /// <returns>The exit code</returns>
        public async Task<int> GenerateDslTemplateAsync(string outputPath, string templateName)
        {
            try
            {
                _logger.LogInformation($"Generating DSL template: {templateName} to {outputPath}");

                string template = templateName.ToLower() switch
                {
                    "basic" => GetBasicTemplate(),
                    "chat" => GetChatTemplate(),
                    "agent" => GetAgentTemplate(),
                    _ => GetBasicTemplate()
                };

                // Create the directory if it doesn't exist
                var directory = Path.GetDirectoryName(outputPath);
                if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
                {
                    Directory.CreateDirectory(directory);
                }

                // Write the template to the file
                await File.WriteAllTextAsync(outputPath, template);

                _logger.LogInformation($"DSL template generated successfully: {outputPath}");

                return 0;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error generating DSL template: {ex.Message}");
                return 1;
            }
        }

        private string GetBasicTemplate()
        {
            return @"CONFIG {
    model: ""llama3""
    temperature: 0.7
    max_tokens: 1000
}

PROMPT {
    text: ""What is the capital of France?""
}

ACTION {
    type: ""generate""
    model: ""llama3""
}";
        }

        private string GetChatTemplate()
        {
            return @"CONFIG {
    model: ""llama3""
    temperature: 0.7
    max_tokens: 1000
}

PROMPT {
    text: ""You are a helpful assistant. Answer the user's questions accurately and concisely.""
    role: ""system""
}

PROMPT {
    text: ""What is the capital of France?""
    role: ""user""
}

ACTION {
    type: ""chat""
    model: ""llama3""
}";
        }

        private string GetAgentTemplate()
        {
            return @"CONFIG {
    model: ""llama3""
    temperature: 0.7
    max_tokens: 1000
}

AGENT {
    name: ""researcher""
    description: ""A research agent that can find information on the web""
    capabilities: [""search"", ""summarize"", ""analyze""]
}

TASK {
    description: ""Find information about the history of Paris""
    agent: ""researcher""
}

ACTION {
    type: ""execute""
    task: ""Find information about the history of Paris""
}";
        }

        /// <summary>
        /// Generates an EBNF (Extended Backus-Naur Form) grammar specification for the TARS DSL
        /// </summary>
        /// <param name="outputPath">The path to save the generated EBNF file</param>
        /// <returns>The exit code</returns>
        public async Task<int> GenerateEbnfAsync(string outputPath)
        {
            try
            {
                _logger.LogInformation($"Generating EBNF grammar specification to {outputPath}");

                // Generate the EBNF grammar
                Option<string> ebnfOption = GenerateEbnfGrammar();

                return await ebnfOption.Match(
                    some: async ebnf =>
                    {
                        // Create the directory if it doesn't exist
                        var directory = Path.GetDirectoryName(outputPath);
                        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
                        {
                            Directory.CreateDirectory(directory);
                        }

                        // Write the EBNF to the file
                        await File.WriteAllTextAsync(outputPath, ebnf);

                        _logger.LogInformation($"EBNF grammar specification generated successfully: {outputPath}");
                        return 0;
                    },
                    none: () =>
                    {
                        _logger.LogError("Failed to generate EBNF grammar specification");
                        return Task.FromResult(1);
                    });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error generating EBNF grammar specification: {ex.Message}");
                return 1;
            }
        }

        /// <summary>
        /// Generates the EBNF grammar for the TARS DSL
        /// </summary>
        /// <returns>An Option containing the EBNF grammar if successful</returns>
        private Option<string> GenerateEbnfGrammar()
        {
            try
            {
                // Create the EBNF grammar as a multi-line string to avoid escaping issues
                string ebnfGrammar = @"(* TARS DSL EBNF Grammar Specification *)

(* Top-level program *)
Program = Block, {Block};

(* Block types *)
Block = ConfigBlock | PromptBlock | ActionBlock | TaskBlock | AgentBlock | DescribeBlock | VariableBlock | IfBlock | ForBlock | WhileBlock | FunctionBlock | ReturnBlock | ImportBlock | ExportBlock | TarsBlock | CommunicationBlock | MessageBlock | SpawnAgentBlock | SelfImproveBlock | AutoImproveBlock;

(* Block structure *)
ConfigBlock = 'CONFIG', '{', [Properties], '}';
PromptBlock = 'PROMPT', '{', [Properties], '}';
ActionBlock = 'ACTION', '{', [Properties], '}';
TaskBlock = 'TASK', [Identifier], '{', [Properties], [NestedBlocks], '}';
AgentBlock = 'AGENT', [Identifier], '{', [Properties], [NestedBlocks], '}';
DescribeBlock = 'DESCRIBE', '{', [Properties], '}';
VariableBlock = 'VARIABLE', Identifier, '{', [Properties], '}';
IfBlock = 'IF', '{', [Properties], [NestedBlocks], '}';
ForBlock = 'FOR', '{', [Properties], [NestedBlocks], '}';
WhileBlock = 'WHILE', '{', [Properties], [NestedBlocks], '}';
FunctionBlock = 'FUNCTION', [Identifier], '{', [Properties], [NestedBlocks], '}';
ReturnBlock = 'RETURN', '{', [Properties], '}';
ImportBlock = 'IMPORT', '{', [Properties], '}';
ExportBlock = 'EXPORT', '{', [Properties], '}';
TarsBlock = 'TARS', '{', [NestedBlocks], '}';
CommunicationBlock = 'COMMUNICATION', '{', [Properties], '}';
MessageBlock = 'MESSAGE', '{', [Properties], '}';
SpawnAgentBlock = 'SPAWN_AGENT', '{', [Properties], '}';
SelfImproveBlock = 'SELF_IMPROVE', '{', [Properties], '}';
AutoImproveBlock = 'AUTO_IMPROVE', '{', [Properties], '}';

(* Properties *)
Properties = Property, {Property};
Property = Identifier, ':', PropertyValue;

(* Property values *)
PropertyValue = StringValue | NumberValue | BoolValue | ListValue | ObjectValue;
StringValue = '""', {Character - '""'}, '""';
NumberValue = ['-'], Digit, {Digit}, ['.', {Digit}];
BoolValue = 'true' | 'false';
ListValue = '[', [PropertyValue, {',', PropertyValue}], ']';
ObjectValue = '{', [Property, {',', Property}], '}';

(* Nested blocks *)
NestedBlocks = Block, {Block};

(* Basic elements *)
Identifier = Letter, {Letter | Digit | '_'};
Letter = 'A' | 'B' | ... | 'Z' | 'a' | 'b' | ... | 'z';
Digit = '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9';
Character = Letter | Digit | Symbol;
Symbol = '!' | '@' | '#' | '$' | '%' | '^' | '&' | '*' | '(' | ')' | ... ;

(* Expressions *)
Expression = Literal | Variable | FunctionCall | BinaryOp | UnaryOp;
Literal = StringValue | NumberValue | BoolValue | ListValue | ObjectValue;
Variable = '${', Identifier, '}';
FunctionCall = Identifier, '(', [Expression, {',', Expression}], ')';
BinaryOp = Expression, Operator, Expression;
UnaryOp = Operator, Expression;
Operator = '+' | '-' | '*' | '/' | '==' | '!=' | '>' | '<' | '>=' | '<=' | '&&' | '||';

(* Comments *)
Comment = '//', {Character - NewLine}, NewLine;
MultiLineComment = '/*', {Character - '*/'}, '*/';

(* Whitespace *)
Whitespace = ' ' | '\t' | '\r' | '\n';
";

                return Option.Some(ebnfGrammar);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error generating EBNF grammar: {ex.Message}");
                return Option.None<string>();
            }
        }
    }
}
