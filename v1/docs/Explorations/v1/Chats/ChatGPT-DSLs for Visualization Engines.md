# DSLs for Visualization Engines

**User:** Stephane Pareilleux (spareilleux@gmail.com)  
**Created:** 3/31/2025 22:17  
**Updated:** 3/31/2025 22:27  
**Exported:** 4/1/2025 0:48  

## Prompt:
Please explore the DSLs for creating d3.js, three.js or any other WebGpu based engine for producing visualizations in TARS

## Response:
Raisonné sur les DSL pour visualisations pendant 47 secondes

Below is an exploration of how domain‐specific languages (DSLs) can be used to create visualizations—whether by generating d3.js charts, three.js 3D scenes, or leveraging emerging WebGPU engines—and how such DSLs might be integrated into a system like TARS.

* * *

Overview
--------

A DSL for visualizations is a specialized language that abstracts away the low‑level details of a rendering engine (such as d3.js, three.js, or WebGPU) and allows users to declare what they want to see in a more intuitive, higher‑level syntax. In a platform like TARS, incorporating such DSLs can empower users to quickly prototype, iterate, and deploy visualizations without having to manually write all the boilerplate code for each underlying engine.

* * *

DSLs for d3.js
--------------

### Declarative Grammars: Vega and Vega‑Lite

*   **Vega/Vega‑Lite:**  
    These are popular declarative grammars for creating a wide range of statistical visualizations. Although they aren’t d3.js in name, Vega is built on top of d3 and provides a JSON-based DSL. By writing a Vega‑Lite specification, users can define data sources, visual encoding channels, and interactions without directly handling d3’s imperative API.
    

### Higher‑Level Abstractions

*   **D3plus:**  
    Libraries like D3plus aim to simplify common tasks (such as creating bar charts, pie charts, or network diagrams) by providing a more concise DSL that internally manages the more complex aspects of d3.js.
    

_Example DSL snippet (conceptual):_

```json
{
  "chart": {
    "type": "bar",
    "data": "sales_data.csv",
    "encoding": {
      "x": "month",
      "y": "sales"
    },
    "style": {
      "color": "steelblue"
    }
  }
}
```

This style of declaration can be parsed and converted into the appropriate d3.js calls by a DSL interpreter within TARS.

* * *

DSLs for three.js
-----------------

### Declarative HTML‑like Markup: A‑Frame

*   **A‑Frame:**  
    Built on top of three.js, A‑Frame allows developers to construct 3D and VR scenes using custom HTML elements. Its declarative nature means that a scene can be defined with tags rather than JavaScript code, effectively serving as a DSL for 3D visualization.
    

_Example snippet using A‑Frame:_

```html
<a-scene>
  <a-box position="0 1.5 -3" rotation="0 45 0" color="#4CC3D9"></a-box>
  <a-sphere position="0 1.25 -5" radius="1.25" color="#EF2D5E"></a-sphere>
  <a-cylinder position="1 0.75 -3" radius="0.5" height="1.5" color="#FFC65D"></a-cylinder>
</a-scene>
```

### React‑Based Approaches

*   **React Three Fiber:**  
    Although it uses JSX (which itself is a kind of DSL for describing UI components), React Three Fiber provides a way to build three.js scenes using familiar React paradigms. The component-based approach can be seen as another flavor of DSL that abstracts away some of three.js’s lower-level intricacies.
    

* * *

DSLs for WebGPU‑Based Engines
-----------------------------

### Emerging DSL Concepts

*   **Shader DSLs:**  
    With WebGPU’s rise, DSLs are emerging to help write GPU shaders more efficiently. For example, WGSL (WebGPU Shading Language) is a low‑level DSL for writing shaders. While WGSL is focused on shader programming rather than full visualization pipelines, it’s an essential piece of the WebGPU puzzle.
    
*   **Higher‑Level Abstractions:**  
    As WebGPU matures, we can expect DSLs that allow developers to declare complex rendering pipelines or visualization logic without directly managing GPU buffers and pipelines. Such DSLs might eventually offer a unified syntax that compiles into WebGPU commands while retaining a higher level of abstraction similar to Vega or A‑Frame.
    

* * *

Integration into TARS
---------------------

Integrating DSLs into TARS for visualization purposes could involve several key design elements:

1.  **Unified Language Front‑End:**  
    TARS could provide a single DSL (or a family of DSLs) that allows users to specify their visualization intent regardless of the underlying engine. A directive within the DSL might specify whether the visualization should be rendered using d3.js, three.js, or a WebGPU‑based engine.
    
2.  **Backend Transpilers/Interpreters:**  
    Each DSL can have a corresponding transpiler or interpreter that converts the high‑level specification into engine‑specific code. For instance, a TARS DSL statement for a bar chart might be transpiled into a Vega‑Lite JSON specification (and then rendered via d3) or into custom d3.js code.
    
3.  **Interactive & Reactive Features:**  
    Given the trend toward interactive and real‑time visualizations, a TARS‑based DSL should allow reactive declarations that automatically update visualizations when underlying data changes. This might include bindings to data streams, event handlers, and state management features.
    
4.  **Customization & Extensibility:**  
    While DSLs are great for rapid development, there’s often a need for custom behavior. TARS can allow embedding of low‑level code snippets (or hooks) for scenarios where the DSL’s abstractions need to be bypassed, providing flexibility without sacrificing ease of use.
    

* * *

Challenges and Considerations
-----------------------------

*   **Balancing Abstraction and Control:**  
    The DSL should hide unnecessary complexity yet allow fine‑grained control when needed. Too much abstraction may limit customization, whereas too little can defeat the purpose of using a DSL.
    
*   **Performance Overheads:**  
    Converting a high‑level DSL into efficient low‑level code (especially for GPU‑intensive tasks) requires careful design to ensure that performance is not sacrificed.
    
*   **Ecosystem and Maintenance:**  
    DSLs that sit on top of fast-evolving libraries (like d3.js or WebGPU) must be maintained and updated regularly to remain compatible with underlying APIs.
    

* * *

Conclusion
----------

Exploring DSLs for visualizations in TARS involves considering how declarative, high‑level languages can simplify the creation of interactive and complex visualizations. Whether using established grammars like Vega‑Lite for d3.js, employing HTML‑based DSLs like A‑Frame for three.js, or pioneering new DSLs for WebGPU, the goal is to abstract low‑level details and let users focus on the “what” rather than the “how.”

By integrating such DSLs into TARS, developers and analysts could build sophisticated visualizations more quickly, easily switch between different rendering engines, and ultimately achieve a more flexible and powerful visualization pipeline.

Feel free to ask for more detailed examples or further exploration of any specific aspect of this integration!

## Prompt:
Explore more deeply applications and TARS DSL extensions or a sub- DSL for TARS

## Response:
Raisonnement terminé pendant 10 secondes

Below is a deeper exploration into how TARS can evolve its DSL ecosystem through targeted extensions and the creation of sub‑DSLs, along with some applications that benefit from these advancements.

* * *

1\. Advanced Applications of TARS DSL
-------------------------------------

**Rapid Prototyping and Iteration:**

*   **Interactive Dashboards & Analytics:** With TARS, analysts can quickly sketch out visualizations that pull from real‑time data, reducing the barrier to exploring business intelligence or scientific datasets.
    
*   **Simulation & Educational Tools:** Domain experts can use TARS to prototype interactive simulations (e.g., physical systems, financial models) without needing to write boilerplate code for each underlying rendering engine.
    

**Cross‑Engine Visualizations:**

*   **Unified Codebase:** By abstracting visualization details, TARS enables developers to specify visual intent once and deploy across multiple engines—be it d3.js for data‐driven charts, three.js for immersive 3D scenes, or even emerging WebGPU-based solutions for high-performance rendering.
    
*   **Seamless Transitions:** Users can experiment with different rendering paradigms without needing to overhaul their visualization logic, thanks to the DSL’s ability to target different backends.
    

* * *

2\. TARS DSL Extensions: Design Considerations
----------------------------------------------

**Modularity and Extensibility:**

*   **Plugin Architecture:** The TARS DSL can be designed with an extension framework that lets developers “plug in” new DSL modules. For instance, a dedicated plugin for advanced animation or interaction patterns could extend the core language.
    
*   **Sub‑DSLs for Specific Domains:**
    
    *   _Visualization Sub‑DSL:_ A tailored language for crafting data-driven charts, offering built-in syntax for common statistical operations, axis configuration, and color mapping.
        
    *   _3D Scene Sub‑DSL:_ A subset of TARS focused on 3D constructs, allowing declarative definitions of geometries, lighting, and camera behaviors that compile down to three.js or a WebGPU shader pipeline.
        

**Grammar Extensions and Macros:**

*   **Macro Systems:** Embedding a macro language within TARS enables users to define repetitive patterns or custom abstractions. For example, a macro might simplify the creation of interactive tooltips or animation sequences.
    
*   **Dynamic Code Generation:** The DSL extensions can include meta-programming facilities that translate high-level directives into optimized engine-specific code, ensuring that abstractions do not come at the cost of performance.
    

* * *

3\. Designing a TARS Sub‑DSL
----------------------------

**Goals for a Sub‑DSL:**

*   **Simplicity with Power:** The sub‑DSL should encapsulate common patterns in a domain (e.g., 3D scene setup) while allowing overrides for custom behavior.
    
*   **Interoperability:** It must integrate seamlessly with the TARS core so that users can combine different sub‑DSL components (like merging data-driven charts with 3D models in the same dashboard).
    

**Conceptual Example: A 3D Visualization Sub‑DSL**

Imagine a syntax where a user can define a 3D scene without directly interacting with three.js APIs:

```tars
scene {
  camera { type: "perspective", position: [0, 2, 5] }
  light { type: "ambient", intensity: 0.5 }
  
  object "cube" {
    geometry { type: "box", size: [1, 1, 1] }
    material { color: "#4CC3D9" }
    transform { position: [0, 0.5, 0], rotation: [0, 45, 0] }
  }
  
  object "sphere" {
    geometry { type: "sphere", radius: 0.75 }
    material { color: "#EF2D5E" }
    transform { position: [1, 1, -2] }
  }
}
```

In this sub‑DSL:

*   **Declarative Elements:** Users define scene components (camera, light, objects) in a way that abstracts low-level API calls.
    
*   **Interpreters/Transpilers:** The TARS engine processes this DSL snippet, converts it into the appropriate three.js or WebGPU calls, and manages the rendering pipeline.
    
*   **Customization Hooks:** Developers can extend or override components for specialized behaviors (e.g., custom shaders, event handling).
    

* * *

4\. Integration Strategies for TARS DSL Extensions
--------------------------------------------------

**Unified Front‑End with Modular Back‑Ends:**

*   **Directive Annotations:** TARS could allow users to annotate their DSL code with directives indicating which rendering engine to target. For example, a directive might specify whether to compile the visualization into d3.js, three.js, or a custom WebGPU pipeline.
    
*   **Hybrid Workflows:** Users might mix sub‑DSLs within a single TARS project. A data visualization dashboard might contain a Vega‑Lite-inspired module for charts alongside a 3D scene module for interactive spatial data, all orchestrated by TARS.
    

**Developer Tooling and IDE Support:**

*   **Syntax Highlighting and Validation:** Enhanced tooling that recognizes TARS DSL extensions and sub‑DSLs can help catch errors early in the development cycle.
    
*   **Live Preview and Debugging:** A built-in interpreter that provides live previews of DSL code changes can accelerate iterative development. Advanced debugging tools would help trace errors back to high-level DSL constructs rather than low-level engine calls.
    

* * *

5\. Challenges and Future Directions
------------------------------------

**Balancing Abstraction and Flexibility:**

*   **Overhead vs. Control:** One of the key challenges is ensuring that DSL abstractions do not incur significant performance overhead while still providing the flexibility to tweak low-level details when necessary.
    
*   **Evolving Standards:** As WebGPU and other rendering technologies continue to mature, the TARS DSL and its extensions must evolve, necessitating a robust versioning and migration strategy for DSL codebases.
    

**Community and Ecosystem Development:**

*   **Open Extensions:** By opening up the DSL extension framework, TARS can encourage community contributions—new sub‑DSLs for emerging visualization paradigms, custom plugins, or even integrations with other data processing languages.
    
*   **Documentation and Examples:** Comprehensive guides, examples, and best practices will be essential to help developers effectively leverage the DSL’s extensibility.
    

* * *

Conclusion
----------

Expanding TARS with DSL extensions and dedicated sub‑DSLs opens up a range of possibilities—from simplifying 3D scene creation to seamlessly integrating data-driven and high-performance visualizations. This approach empowers users to focus on creative expression and domain-specific insights while the underlying engine manages the complexities of rendering and performance optimization.

This deeper exploration highlights both the technical strategies and potential applications, paving the way for a more modular, flexible, and powerful visualization framework within TARS.

## Prompt:
Do the same exploration for music (Vexflow, tabs, midi, etc...)

## Response:
Raisonnement terminé pendant 13 secondes

Below is an in‑depth exploration of how DSLs can be applied to the musical domain—covering notation (using tools like VexFlow and ABC notation), tablature (for guitar and other stringed instruments), MIDI sequencing, and beyond—and how TARS might extend its DSL ecosystem to include a dedicated music sub‑DSL.

* * *

1\. Overview of Music DSLs
--------------------------

A domain‑specific language for music enables composers, educators, and developers to describe musical elements (notes, rhythms, chords, dynamics, etc.) in a high‑level, declarative fashion. Rather than manually crafting low‑level representations, a music DSL abstracts the underlying complexity. When integrated into TARS, such a DSL can serve multiple applications:

*   **Music Notation Rendering:** Generating visually accurate musical scores.
    
*   **Tablature Creation:** Producing guitar or bass tabs with intuitive syntax.
    
*   **MIDI Composition:** Sequencing musical events for playback or further processing.
    
*   **Interactive Educational Tools:** Allowing real‑time editing and visualization of musical ideas.
    

* * *

2\. DSLs for Music Notation
---------------------------

### VexFlow and Beyond

*   **VexFlow:**  
    VexFlow is a JavaScript library designed to render standard music notation on HTML5 canvas or SVG. A DSL built on top of VexFlow would let users declare notes, rests, beams, and articulations without having to manage the rendering details.
    
*   **ABC Notation & LilyPond:**  
    Both ABC notation and LilyPond are established DSLs for music notation. ABC notation uses simple text to represent music, while LilyPond focuses on producing beautifully engraved scores. These examples illustrate how a dedicated DSL can encapsulate complex musical semantics into concise code.
    

_Conceptual Example – Music Notation DSL:_

```tars
score {
  title: "Simple Melody"
  composer: "Unknown"

  instrument "piano" {
    clef: "treble"
    key: "C"
    time: "4/4"
  }

  measure 1 {
    note "C4" duration "quarter"
    note "D4" duration "quarter"
    note "E4" duration "quarter"
    note "F4" duration "quarter"
  }

  measure 2 {
    chord ["G4", "B4", "D5"] duration "half"
    rest duration "half"
  }
}
```

In this DSL snippet, the high‑level description is parsed and transformed into the corresponding VexFlow (or similar) rendering calls.

* * *

3\. DSLs for Tablature
----------------------

### Guitar Tabs and Chord Sheets

*   **Tablature DSLs:**  
    For guitarists and other string instrument players, DSLs can simplify the creation of tablature. A DSL might let users specify string positions, frets, and chord diagrams using a syntax that abstracts the underlying graphical representation.
    
*   **ChordPro and Similar Formats:**  
    Formats like ChordPro integrate chord annotations with lyrics. A TARS sub‑DSL could combine the ease of a textual DSL with rendering capabilities that generate both standard notation and tablature views.
    

_Conceptual Example – Tablature DSL:_

```tars
tablature {
  song "Strumming Fun" {
    instrument: "guitar"
    tuning: "EADGBE"
    measure 1 {
      string 1: fret 0
      string 2: fret 2
      string 3: fret 2
      string 4: fret 1
      string 5: fret 0
      string 6: fret x
    }
    chord "G" in measure 2 {
      positions: [3, 2, 0, 0, 0, 3]
    }
  }
}
```

This DSL allows users to declare notes and chords in tablature form, which the TARS engine could then render graphically or export as text.

* * *

4\. DSLs for MIDI Sequencing
----------------------------

### High‑Level MIDI Composition

*   **MIDI DSLs:**  
    Composing MIDI sequences involves specifying note events, velocities, and timings. A DSL for MIDI can encapsulate these details so that users write musical ideas that are later compiled into standard MIDI files or real‑time playback instructions.
    
*   **Integration with Notation:**  
    Ideally, the music DSL in TARS would allow composers to produce both the visual score (via VexFlow) and the corresponding MIDI sequence from the same high‑level code. This duality ensures consistency between what is seen and what is heard.
    

_Conceptual Example – MIDI DSL within a Score:_

```tars
score {
  title: "MIDI Melody"
  instrument "synth" {
    midiChannel: 1
  }

  measure 1 {
    note "C4" duration "quarter" velocity 80
    note "E4" duration "quarter" velocity 80
    note "G4" duration "quarter" velocity 80
    note "C5" duration "quarter" velocity 80
  }
}
```

This example highlights how note properties can be extended with MIDI‑specific attributes such as velocity and channel information.

* * *

5\. Integration into TARS as a Music Sub‑DSL
--------------------------------------------

### Unified Music Workflow

TARS could offer a modular music sub‑DSL that allows users to combine different musical representations in a single project:

*   **Interoperability:**  
    Allowing users to switch between or combine standard notation, tablature, and MIDI playback in one unified file.
    
*   **Modular Extensions:**  
    Similar to visualization sub‑DSLs, the music sub‑DSL could be modular—offering plugins for different instruments, performance effects, or even real‑time interactive composition.
    
*   **Live Preview & Editing:**  
    A built‑in live preview might show rendered notation alongside an interactive MIDI sequencer, giving immediate feedback as the user edits their DSL code.
    

### Example Integration Strategy

1.  **Unified Front‑End:**  
    A TARS music editor where users input their DSL code and see both a score and a playable MIDI preview.
    
2.  **Back‑End Transpilers:**  
    The TARS engine compiles DSL code into:
    
    *   VexFlow commands for score rendering.
        
    *   Tab generators for tablature views.
        
    *   MIDI instructions for audio playback.
        
3.  **Customization Hooks:**  
    Allow advanced users to embed raw MIDI or VexFlow commands if they need finer control over the output.
    

* * *

6\. Challenges and Future Directions
------------------------------------

### Balancing Abstraction and Detail

*   **Expressivity vs. Simplicity:**  
    The DSL should be expressive enough to capture complex musical ideas while remaining accessible to non‑programmers.
    
*   **Performance Considerations:**  
    Generating real‑time MIDI playback or rendering high‑resolution scores must be efficient and scalable.
    
*   **Evolving Standards:**  
    As musical technology evolves (e.g., new synthesizer standards, interactive performance tools), the DSL must be flexible enough to incorporate new features without disrupting existing projects.
    

### Community and Ecosystem

*   **Extensible Framework:**  
    A plugin or module system that invites community contributions—new instrument definitions, special effects, or custom notations.
    
*   **Robust Tooling:**  
    Integration with popular music production and editing software, along with features like syntax highlighting, validation, and live debugging.
    

* * *

7\. Conclusion
--------------

Expanding TARS with a dedicated music DSL—and sub‑DSLs for notation, tablature, and MIDI sequencing—opens a rich vein of creative potential. By abstracting the complexities of musical rendering and composition, TARS can empower musicians, educators, and developers to focus on creative expression rather than low‑level implementation details. Whether rendering a beautifully engraved score with VexFlow, generating intuitive guitar tabs, or composing interactive MIDI sequences, a unified music DSL within TARS promises a flexible, integrated, and powerful platform for digital music creation.

Feel free to ask for further details or specific examples on any aspect of this exploration!

## Prompt:
Could we envision an ad hoc creation of a DSL by TARS inference engine, then its incorporation into TARS memory (File or other support)

## Response:
Raisonnement terminé pendant 17 secondes

Yes, we can envision a system where the TARS inference engine dynamically generates DSL definitions on the fly and then integrates them into its persistent memory for later reuse. This kind of adaptive, self-modifying system could greatly enhance flexibility and rapid prototyping. Below is an in‑depth exploration of how such a mechanism might work.

* * *

1\. Conceptual Overview
-----------------------

**Ad Hoc DSL Generation:**  
The idea is that TARS’s inference engine—using its understanding of the user’s intent, recurring patterns in data, or domain-specific needs—could synthesize a new DSL or extend an existing one without manual intervention. For instance, if users repeatedly describe visualizations or music compositions in a particular pattern, TARS might infer a set of grammar rules that encapsulate these patterns.

**Persistent Integration:**  
Once generated, these DSL definitions aren’t transient. They could be stored in TARS’s memory as files or in a dedicated configuration database. This persistent storage means that the newly created DSL or DSL extensions become part of the system’s toolkit, available for future tasks and even subject to further refinement or versioning.

* * *

2\. Workflow and Architecture
-----------------------------

### A. Inference and Generation

1.  **Data Collection and Analysis:**
    
    *   The inference engine continuously monitors user interactions and project specifications.
        
    *   It identifies recurring patterns or specific needs that aren’t well-served by the existing DSLs.
        
2.  **DSL Synthesis:**
    
    *   Based on these insights, the engine constructs a new DSL specification or extends an existing one.
        
    *   This involves defining grammar rules, syntax constructs, and semantics tailored to the observed domain—whether it’s for visualizations, music, or any other application.
        
3.  **Validation and Testing:**
    
    *   Before full integration, the generated DSL is validated (either automatically or through user feedback) to ensure it correctly maps to the intended functionalities.
        
    *   Automated tests or simulated runs can help catch inconsistencies or inefficiencies.
        

### B. Incorporation into TARS Memory

1.  **File-Based Storage:**
    
    *   The DSL definition could be saved as a file (e.g., a JSON, YAML, or custom DSL file) within TARS’s file system.
        
    *   This file serves as a module that the system can load on startup or on demand.
        
2.  **Database or In-Memory Storage:**
    
    *   Alternatively, TARS might use a database or an in-memory data structure to store DSL definitions, allowing for version control, quick retrieval, and dynamic updates.
        
    *   This approach supports collaborative environments where DSL modules are shared or updated in real time.
        
3.  **Integration into the Compiler/Interpreter Pipeline:**
    
    *   Once stored, the DSL becomes part of the language processing pipeline.
        
    *   The TARS engine can now compile or interpret code written in the new DSL, translating it into low‑level commands for rendering engines (like d3.js, three.js, or WebGPU) or other back-end systems.
        

### C. Dynamic Evolution and Feedback

*   **Iterative Refinement:**  
    The system can continue to refine DSL definitions based on further user feedback or observed usage. Updates to the DSL are integrated seamlessly, maintaining backward compatibility through versioning.
    
*   **Hybrid DSL Environments:**  
    TARS might support a hybrid model where both manually authored and ad hoc generated DSLs coexist, giving users the choice to switch between standard solutions and those tailored to their unique workflow.
    

* * *

3\. Potential Applications and Benefits
---------------------------------------

*   **Rapid Prototyping:**  
    By automatically generating DSLs that encapsulate common patterns, developers and non‑technical users can prototype complex visualizations, musical scores, or other artifacts quickly without getting bogged down in low‑level details.
    
*   **Customization and Flexibility:**  
    As user needs evolve, TARS can create specialized DSL modules that adapt to new domains (e.g., an emerging trend in interactive visual media or novel musical notation styles).
    
*   **Learning and Adaptation:**  
    With each new DSL generated, TARS expands its own knowledge base. This adaptive learning approach can eventually lead to a highly personalized development environment that “understands” the user’s preferred patterns and practices.
    

* * *

4\. Challenges and Considerations
---------------------------------

*   **Correctness and Safety:**  
    Automatically generated DSLs must be rigorously validated to prevent errors in interpretation or execution, especially when they impact critical systems like real-time rendering or live performance.
    
*   **Performance Overhead:**  
    Dynamically creating and integrating DSLs adds a layer of complexity. Efficient parsing, compiling, and execution pipelines are essential to avoid latency or resource overhead.
    
*   **User Trust and Transparency:**  
    Users need visibility into how and why DSLs are generated. Providing options to review, modify, or override automatically generated definitions will help maintain trust in the system.
    
*   **Versioning and Compatibility:**  
    As DSLs evolve over time, maintaining compatibility with existing projects is crucial. A robust versioning system and migration strategies will be necessary.
    

* * *

5\. Conclusion
--------------

Envisioning an ad hoc creation of a DSL by the TARS inference engine—with subsequent incorporation into its persistent memory—is not only feasible but could transform the flexibility and adaptability of the platform. This approach leverages automated reasoning to create tailored language constructs that evolve with user needs, ultimately streamlining complex tasks like visualization, music composition, or any domain-specific application. By addressing challenges related to validation, performance, and transparency, TARS could offer a dynamic and highly customizable development environment that continuously learns and adapts.

Feel free to ask for further details or specific examples on any aspect of this envisioned process!



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)