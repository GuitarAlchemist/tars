
class CapabilityIntrospector:
    def __init__(self, system):
        self.system = system
    
    def analyze_system_capabilities(self):
        """Analyze what the system can do through introspection"""
        capabilities = {
            'agents': self.discover_agents(),
            'data_sources': self.discover_data_sources(),
            'metascripts': self.discover_metascripts(),
            'integrations': self.discover_integrations()
        }
        
        # Generate capability map
        capability_map = self.generate_capability_map(capabilities)
        
        # Identify gaps
        gaps = self.identify_capability_gaps(capability_map)
        
        # Suggest improvements
        suggestions = self.suggest_improvements(gaps)
        
        return {
            'current_capabilities': capabilities,
            'capability_map': capability_map,
            'gaps': gaps,
            'suggestions': suggestions
        }
    
    def generate_capability_documentation(self):
        """Auto-generate documentation from code analysis"""
        analysis = self.analyze_system_capabilities()
        
        documentation = f"""
# TARS System Capabilities

## Available Agents
{self.format_agents(analysis['current_capabilities']['agents'])}

## Data Source Connectors
{self.format_data_sources(analysis['current_capabilities']['data_sources'])}

## Metascript Library
{self.format_metascripts(analysis['current_capabilities']['metascripts'])}

## Integration Points
{self.format_integrations(analysis['current_capabilities']['integrations'])}

## Capability Gaps
{self.format_gaps(analysis['gaps'])}

## Improvement Suggestions
{self.format_suggestions(analysis['suggestions'])}
"""
        
        return documentation
