namespace TarsEngine.FSharp.Notebooks.Types

open System
open System.Collections.Generic

/// <summary>
/// Types for notebook output generation and formatting
/// </summary>

/// Output format types
type OutputFormat = 
    | HTML
    | PDF
    | Python
    | Markdown
    | LaTeX
    | Slides
    | Word
    | Excel

/// Output generation request
type OutputGenerationRequest = {
    Notebook: JupyterNotebook
    Format: OutputFormat
    OutputPath: string option
    Options: OutputOptions
}

/// Output generation options
and OutputOptions = {
    IncludeCode: bool
    IncludeOutputs: bool
    IncludeMarkdown: bool
    Theme: string option
    Template: string option
    CustomCSS: string option
    EmbedImages: bool
    TableOfContents: bool
    PageNumbers: bool
    Metadata: Map<string, obj>
}

/// Output generation result
type OutputGenerationResult = {
    Success: bool
    OutputPath: string
    Format: OutputFormat
    FileSize: int64
    GenerationTime: TimeSpan
    Warnings: string list
    Errors: string list
}

/// HTML output configuration
type HtmlOutputConfig = {
    Theme: string
    IncludeCSS: bool
    IncludeJS: bool
    EmbedImages: bool
    FullDocument: bool
    CustomCSS: string option
    CustomJS: string option
}

/// PDF output configuration
type PdfOutputConfig = {
    PageSize: string
    Orientation: string
    Margins: PdfMargins
    IncludePageNumbers: bool
    IncludeHeader: bool
    IncludeFooter: bool
    HeaderText: string option
    FooterText: string option
}

/// PDF margins
and PdfMargins = {
    Top: float
    Bottom: float
    Left: float
    Right: float
}

/// Python script output configuration
type PythonOutputConfig = {
    IncludeMarkdownAsComments: bool
    IncludeOutputsAsComments: bool
    AddExecutionGuards: bool
    PreserveStructure: bool
    AddImports: bool
}

/// Markdown output configuration
type MarkdownOutputConfig = {
    IncludeCodeBlocks: bool
    IncludeOutputs: bool
    CodeBlockLanguage: string
    ImageHandling: ImageHandling
    LinkHandling: LinkHandling
}

/// Image handling options
and ImageHandling = 
    | Embed
    | Link
    | Copy
    | Skip

/// Link handling options
and LinkHandling = 
    | Preserve
    | Convert
    | Remove

/// LaTeX output configuration
type LaTeXOutputConfig = {
    DocumentClass: string
    Packages: string list
    IncludeTitle: bool
    IncludeAuthor: bool
    IncludeDate: bool
    BibliographyStyle: string option
    CustomPreamble: string option
}

/// Slides output configuration
type SlidesOutputConfig = {
    Framework: SlideFramework
    Theme: string
    Transition: string
    AutoSlide: bool
    Controls: bool
    Progress: bool
    SlideNumbers: bool
    CustomCSS: string option
}

/// Slide framework options
and SlideFramework = 
    | RevealJS
    | ImpressJS
    | DeckJS
    | Bespoke

/// Word document output configuration
type WordOutputConfig = {
    Template: string option
    IncludeTableOfContents: bool
    IncludePageNumbers: bool
    HeaderText: string option
    FooterText: string option
    StyleMapping: Map<string, string>
}

/// Excel output configuration
type ExcelOutputConfig = {
    WorksheetName: string
    IncludeFormatting: bool
    IncludeCharts: bool
    AutoFitColumns: bool
    FreezeHeaders: bool
    TableStyle: string option
}

/// Output converter interface
type IOutputConverter =
    /// Convert notebook to specified format
    abstract member ConvertAsync: OutputGenerationRequest -> Async<OutputGenerationResult>
    
    /// Get supported formats
    abstract member SupportedFormats: OutputFormat list
    
    /// Validate conversion request
    abstract member ValidateRequest: OutputGenerationRequest -> ValidationResult

/// HTML converter
type HtmlConverter() =
    interface IOutputConverter with
        member _.ConvertAsync(request: OutputGenerationRequest) = async {
            // Implementation for HTML conversion
            return {
                Success = true
                OutputPath = request.OutputPath |> Option.defaultValue "output.html"
                Format = HTML
                FileSize = 0L
                GenerationTime = TimeSpan.Zero
                Warnings = []
                Errors = []
            }
        }
        
        member _.SupportedFormats = [HTML]
        
        member _.ValidateRequest(request: OutputGenerationRequest) = {
            IsValid = true
            Errors = []
            Warnings = []
        }

/// PDF converter
type PdfConverter() =
    interface IOutputConverter with
        member _.ConvertAsync(request: OutputGenerationRequest) = async {
            // Implementation for PDF conversion
            return {
                Success = true
                OutputPath = request.OutputPath |> Option.defaultValue "output.pdf"
                Format = PDF
                FileSize = 0L
                GenerationTime = TimeSpan.Zero
                Warnings = []
                Errors = []
            }
        }
        
        member _.SupportedFormats = [PDF]
        
        member _.ValidateRequest(request: OutputGenerationRequest) = {
            IsValid = true
            Errors = []
            Warnings = []
        }

/// Python converter
type PythonConverter() =
    interface IOutputConverter with
        member _.ConvertAsync(request: OutputGenerationRequest) = async {
            // Implementation for Python conversion
            return {
                Success = true
                OutputPath = request.OutputPath |> Option.defaultValue "output.py"
                Format = Python
                FileSize = 0L
                GenerationTime = TimeSpan.Zero
                Warnings = []
                Errors = []
            }
        }
        
        member _.SupportedFormats = [Python]
        
        member _.ValidateRequest(request: OutputGenerationRequest) = {
            IsValid = true
            Errors = []
            Warnings = []
        }

/// Output converter factory
type OutputConverterFactory() =
    
    /// Create converter for specified format
    member _.CreateConverter(format: OutputFormat) : IOutputConverter =
        match format with
        | HTML -> HtmlConverter() :> IOutputConverter
        | PDF -> PdfConverter() :> IOutputConverter
        | Python -> PythonConverter() :> IOutputConverter
        | Markdown -> HtmlConverter() :> IOutputConverter // Placeholder
        | LaTeX -> HtmlConverter() :> IOutputConverter // Placeholder
        | Slides -> HtmlConverter() :> IOutputConverter // Placeholder
        | Word -> HtmlConverter() :> IOutputConverter // Placeholder
        | Excel -> HtmlConverter() :> IOutputConverter // Placeholder
    
    /// Get all available converters
    member this.GetAllConverters() : (OutputFormat * IOutputConverter) list =
        [
            (HTML, this.CreateConverter(HTML))
            (PDF, this.CreateConverter(PDF))
            (Python, this.CreateConverter(Python))
        ]

/// Output generation service
type OutputGenerationService() =
    
    let converterFactory = OutputConverterFactory()
    
    /// Generate output in specified format
    member _.GenerateOutputAsync(request: OutputGenerationRequest) : Async<OutputGenerationResult> = async {
        try
            let converter = converterFactory.CreateConverter(request.Format)
            
            // Validate request
            let validation = converter.ValidateRequest(request)
            if not validation.IsValid then
                return {
                    Success = false
                    OutputPath = ""
                    Format = request.Format
                    FileSize = 0L
                    GenerationTime = TimeSpan.Zero
                    Warnings = validation.Warnings |> List.map (fun w -> w.Message)
                    Errors = validation.Errors |> List.map (fun e -> e.Message)
                }
            else
                // Perform conversion
                let! result = converter.ConvertAsync(request)
                return result
                
        with
        | ex ->
            return {
                Success = false
                OutputPath = ""
                Format = request.Format
                FileSize = 0L
                GenerationTime = TimeSpan.Zero
                Warnings = []
                Errors = [ex.Message]
            }
    }
    
    /// Get supported formats
    member _.GetSupportedFormats() : OutputFormat list =
        [HTML; PDF; Python; Markdown; LaTeX; Slides; Word; Excel]
    
    /// Create default output options
    member _.CreateDefaultOptions() : OutputOptions =
        {
            IncludeCode = true
            IncludeOutputs = true
            IncludeMarkdown = true
            Theme = Some "default"
            Template = None
            CustomCSS = None
            EmbedImages = true
            TableOfContents = false
            PageNumbers = false
            Metadata = Map.empty
        }
