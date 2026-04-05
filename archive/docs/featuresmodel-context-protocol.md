[33mcommit d8a77b07de2f7d6e62ab462b429d75e9677faa3a[m[33m ([m[1;36mHEAD[m[33m -> [m[1;32mmain[m[33m, [m[1;31morigin/main[m[33m)[m
Author: spareilleux <spareilleux@gmail.com>
Date:   Mon Mar 31 22:54:40 2025 -0400

    Implement TARS DSL Engine and Metascript Support
    
    This commit adds a simplified implementation of the TARS Domain Specific Language (DSL) engine and CLI commands for working with metascripts. The changes include:
    
    1. Fixed build errors in TarsEngine.DSL project:
       - Fixed BlockType.Run error in AgentCommunication.fs
       - Fixed Guid option errors in AgentCommunication.fs
       - Fixed type annotation errors in AgentLearning.fs
       - Fixed Map type mismatch in AgentLearning.fs
    
    2. Implemented a simplified DSL engine (SimpleDsl.fs):
       - Basic block structure (DESCRIBE, CONFIG, PROMPT, ACTION, etc.)
       - Variable substitution
       - Simple control flow (IF/ELSE)
       - Basic execution of prompts and actions
       - MCP integration for AI-to-AI collaboration
    
    3. Added CLI commands for metascripts:
       - 'metascript validate' to check metascript syntax
       - 'metascript execute' to run metascripts
       - Added proper error handling and verbose output
    
    4. Created example metascripts:
       - Added hello_world.tars example in Examples/metascripts directory
       - Added tars_augment_collaboration.tars for TARS-Augment collaboration via MCP
       - Added README.md with explanations of the examples
    
    5. Added metascript demo:
       - Created a demo that showcases the metascript capabilities
       - Integrated the demo into the existing demo command
    
    6. Updated documentation:
       - Created comprehensive documentation for the metascript feature
       - Updated the README.md file to include information about the metascript feature
    
    This implementation provides a foundation for the TARS DSL that can be extended with more advanced features in the future and enables collaboration between TARS and other AI systems like Augment Code via MCP.
