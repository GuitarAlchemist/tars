# Create TARS Spec

Create a detailed specification for a new TARS enhancement or autonomous capability with technical specifications and task breakdown.

Refer to the instructions located in this file:
@.agent-os/instructions/core/create-spec.md

## Usage

Use this command when you want to:
- Create detailed specs for TARS autonomous improvements
- Plan CUDA acceleration implementations
- Design new metascript capabilities
- Specify multi-agent coordination features
- Plan self-improvement enhancements

## TARS Spec Requirements

### Technical Specifications Must Include:
- **F# and C# implementation details** - Which language for which components
- **CUDA acceleration requirements** - GPU performance targets and WSL compilation needs
- **Metascript integration** - How the feature integrates with FLUX metascripts
- **Autonomous behavior** - How the feature enhances TARS self-improvement
- **Performance metrics** - Specific targets (e.g., 184M+ searches/second)
- **Testing requirements** - Real functionality validation, no simulations

### Task Breakdown Must Include:
- **Real implementation tasks** - No placeholders or simulations allowed
- **CUDA compilation steps** - WSL-specific compilation requirements
- **Testing and validation** - Concrete proof of functionality
- **Integration testing** - Verification with existing TARS components
- **Performance benchmarking** - Measurement of actual improvements

## Expected Outputs

This command will create a dated spec folder with:
- `srd.md` - Spec Requirements Document
- `technical-specs.md` - Detailed technical implementation
- `tasks.md` - Task breakdown with dependencies
- `performance-targets.md` - Specific performance goals and metrics
