
class AgentDiscoverySystem:
    def __init__(self):
        self.discovered_agents = {}
        self.agent_capabilities = {}
    
    def discover_agents(self, module_path):
        """Dynamically discover and load agents"""
        import importlib
        import inspect
        
        module = importlib.import_module(module_path)
        
        for name, obj in inspect.getmembers(module):
            if self.is_agent_class(obj):
                # Analyze agent capabilities
                capabilities = self.analyze_agent_capabilities(obj)
                
                # Register agent
                self.discovered_agents[name] = obj
                self.agent_capabilities[name] = capabilities
                
                print(f"🔍 Discovered agent: {name} with capabilities: {capabilities}")
    
    def is_agent_class(self, obj):
        """Check if object is a TARS agent"""
        return (inspect.isclass(obj) and 
                hasattr(obj, 'execute') and
                hasattr(obj, 'capabilities'))
    
    def analyze_agent_capabilities(self, agent_class):
        """Analyze agent capabilities through reflection"""
        capabilities = []
        
        # Check methods
        for method_name, method in inspect.getmembers(agent_class, inspect.isfunction):
            if method_name.startswith('can_'):
                capability = method_name[4:]  # Remove 'can_' prefix
                capabilities.append(capability)
        
        # Check attributes
        if hasattr(agent_class, 'capabilities'):
            capabilities.extend(agent_class.capabilities)
        
        return capabilities
    
    def create_agent_for_task(self, task_requirements):
        """Dynamically create best agent for task"""
        best_agent = None
        best_score = 0
        
        for agent_name, capabilities in self.agent_capabilities.items():
            score = self.calculate_capability_match(capabilities, task_requirements)
            if score > best_score:
                best_score = score
                best_agent = self.discovered_agents[agent_name]
        
        return best_agent() if best_agent else None
