
class MetascriptTemplateEngine:
    def __init__(self):
        self.templates = {}
        self.macros = {}
    
    def register_template(self, name, template):
        """Register a reusable metascript template"""
        self.templates[name] = template
    
    def register_macro(self, name, macro_func):
        """Register a code generation macro"""
        self.macros[name] = macro_func
    
    def generate_metascript(self, template_name, parameters):
        """Generate metascript from template"""
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        template = self.templates[template_name]
        
        # Process macros
        processed_template = self.process_macros(template)
        
        # Substitute parameters
        metascript = self.substitute_parameters(processed_template, parameters)
        
        # Validate generated metascript
        validation_result = self.validate_metascript(metascript)
        
        return {
            'metascript': metascript,
            'validation': validation_result,
            'template_used': template_name,
            'parameters': parameters
        }
    
    def process_macros(self, template):
        """Process macro expansions in template"""
        import re
        
        def macro_replacer(match):
            macro_name = match.group(1)
            macro_args = match.group(2).split(',') if match.group(2) else []
            
            if macro_name in self.macros:
                return self.macros[macro_name](*macro_args)
            else:
                return match.group(0)  # Leave unchanged if macro not found
        
        # Find and replace macro calls: @macro_name(args)
        return re.sub(r'@(\w+)\(([^)]*)\)', macro_replacer, template)

# Example templates
engine = MetascriptTemplateEngine()

# Register data processing template
engine.register_template('data_processor', """
# Data Processing Metascript
# Generated from template

let process{DataType} = fun input ->
    async {
        // Validation
        @validate_input({DataType})
        
        // Processing
        let processed = 
            input
            @apply_transformations({Transformations})
            @apply_filters({Filters})
            @apply_aggregations({Aggregations})
        
        // Output
        @format_output({OutputFormat})
        
        return processed
    }
""")

# Register macros
engine.register_macro('validate_input', lambda data_type: f"validateInput<{data_type}> input")
engine.register_macro('apply_transformations', lambda transforms: f"|> {' |> '.join(transforms.split(','))}")
