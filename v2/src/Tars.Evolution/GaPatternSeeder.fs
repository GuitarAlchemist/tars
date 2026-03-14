namespace Tars.Evolution

/// Seeds TARS promotion pipeline with patterns discovered by static analysis
/// of the GA codebase. These are real, recurring structural patterns extracted
/// from production code — not simulated data.
module GaPatternSeeder =

    open System

    /// GA patterns discovered via cross-repo code analysis (2026-03-14).
    /// Each pattern has multiple synthetic "task IDs" representing the
    /// actual code locations where the pattern recurs in GA.
    let gaTraceArtifacts () : PromotionPipeline.TraceArtifact list =
        [
            // ── Pattern 1: Confidence-Evidence-Response ────────────────────
            // Recurs 11 times across all 5 agents + 6 skills.
            // Every agent/skill returns { Result, Confidence, Evidence[], Assumptions[] }.
            // This is a constrained output schema that could become an EBNF grammar rule.
            { TaskId = "ga-agent-theory"
              PatternName = "ga.confidence_evidence_response"
              PatternTemplate = "ga.confidence_evidence_response"
              Context = "TheoryAgent returns structured JSON with confidence score and evidence list"
              Score = 0.92
              Timestamp = DateTime(2026, 3, 14)
              RollbackExpansion = Some "step: parse_query (reason)\n  goal: Understand user's music theory question\n  output: Parsed intent and key entities\nstep: llm_call (tool)\n  tool: chat_client\n  output: Raw LLM response in JSON format\nstep: parse_response (reason)\n  goal: Extract {result, confidence, evidence[], assumptions[]} from JSON\n  output: Structured AgentResponse with validated confidence 0.0-1.0\nstep: fallback (reason)\n  goal: If JSON parse fails, wrap raw text with confidence=0.5\n  output: Degraded but valid AgentResponse" }
            { TaskId = "ga-agent-technique"
              PatternName = "ga.confidence_evidence_response"
              PatternTemplate = "ga.confidence_evidence_response"
              Context = "TechniqueAgent uses same structured response for guitar technique advice"
              Score = 0.88
              Timestamp = DateTime(2026, 3, 14)
              RollbackExpansion = None }
            { TaskId = "ga-agent-composer"
              PatternName = "ga.confidence_evidence_response"
              PatternTemplate = "ga.confidence_evidence_response"
              Context = "ComposerAgent uses same structured response for composition suggestions"
              Score = 0.85
              Timestamp = DateTime(2026, 3, 14)
              RollbackExpansion = None }
            { TaskId = "ga-agent-critic"
              PatternName = "ga.confidence_evidence_response"
              PatternTemplate = "ga.confidence_evidence_response"
              Context = "CriticAgent uses same structured response for music critique"
              Score = 0.90
              Timestamp = DateTime(2026, 3, 14)
              RollbackExpansion = None }
            { TaskId = "ga-skill-scaleinfo"
              PatternName = "ga.confidence_evidence_response"
              PatternTemplate = "ga.confidence_evidence_response"
              Context = "ScaleInfoSkill returns confidence=1.0 for domain-computed results"
              Score = 1.0
              Timestamp = DateTime(2026, 3, 14)
              RollbackExpansion = None }

            // ── Pattern 2: Domain Skill Fast-Path ──────────────────────────
            // Recurs 6 times across all IOrchestratorSkill implementations.
            // Shape: CanHandle(regex/keyword) → Parse → DomainCompute → Return(confidence=1.0)
            // Bypasses LLM entirely for deterministic domain answers.
            { TaskId = "ga-skill-scaleinfo-fastpath"
              PatternName = "ga.domain_skill_fastpath"
              PatternTemplate = "ga.domain_skill_fastpath"
              Context = "ScaleInfoSkill: regex match on key names → pure pitch-class lookup → confidence=1.0"
              Score = 0.95
              Timestamp = DateTime(2026, 3, 14)
              RollbackExpansion = Some "step: can_handle (reason)\n  goal: Check if query matches domain pattern via regex\n  output: Boolean — true if regex matches AND intent keywords present\nstep: parse_input (reason)\n  goal: Extract structured input from natural language (key name, scale type)\n  output: Domain object (e.g., Key, ChordProgression)\nstep: domain_compute (tool)\n  tool: domain_model\n  output: Deterministic result from music theory model (no LLM needed)\nstep: format_response (reason)\n  goal: Wrap domain result in AgentResponse with confidence=1.0\n  output: High-confidence response with factual evidence" }
            { TaskId = "ga-skill-fretspan"
              PatternName = "ga.domain_skill_fastpath"
              PatternTemplate = "ga.domain_skill_fastpath"
              Context = "FretSpanSkill: parse fret diagram → compute physical span → confidence=1.0"
              Score = 0.93
              Timestamp = DateTime(2026, 3, 14)
              RollbackExpansion = None }
            { TaskId = "ga-skill-chordsub"
              PatternName = "ga.domain_skill_fastpath"
              PatternTemplate = "ga.domain_skill_fastpath"
              Context = "ChordSubstitutionSkill: detect chord name → lookup substitutions → confidence=1.0"
              Score = 0.91
              Timestamp = DateTime(2026, 3, 14)
              RollbackExpansion = None }
            { TaskId = "ga-skill-keyid"
              PatternName = "ga.domain_skill_fastpath"
              PatternTemplate = "ga.domain_skill_fastpath"
              Context = "KeyIdentificationSkill: extract chords → compute key via Krumhansl-Schmuckler"
              Score = 0.88
              Timestamp = DateTime(2026, 3, 14)
              RollbackExpansion = None }
            { TaskId = "ga-skill-progression"
              PatternName = "ga.domain_skill_fastpath"
              PatternTemplate = "ga.domain_skill_fastpath"
              Context = "ProgressionCompletionSkill: detect partial progression → suggest completions"
              Score = 0.86
              Timestamp = DateTime(2026, 3, 14)
              RollbackExpansion = None }
            { TaskId = "ga-skill-voicing"
              PatternName = "ga.domain_skill_fastpath"
              PatternTemplate = "ga.domain_skill_fastpath"
              Context = "VoicingComfortSkill: score voicing ergonomics via hand-span model"
              Score = 0.90
              Timestamp = DateTime(2026, 3, 14)
              RollbackExpansion = None }

            // ── Pattern 3: Three-Tier Routing Fallback ─────────────────────
            // GA's SemanticRouter cascades: embedding similarity → LLM routing → keyword.
            // Each tier has a confidence threshold gate.
            // This is the routing pattern TARS lacks (keyword-only today).
            { TaskId = "ga-router-semantic"
              PatternName = "ga.routing_fallback_cascade"
              PatternTemplate = "ga.routing_fallback_cascade"
              Context = "SemanticRouter tier 1: cosine similarity on cached embeddings, gate at 0.85"
              Score = 0.88
              Timestamp = DateTime(2026, 3, 14)
              RollbackExpansion = Some "step: embed_query (tool)\n  tool: text_embeddings\n  output: Float[] embedding vector (cached 5min, 256-entry bound)\nstep: score_agents (reason)\n  goal: Compute cosine similarity between query and each agent's description embedding\n  output: (Agent, float)[] scores sorted descending\nstep: apply_bias (reason)\n  goal: Add learned routing feedback bias to each agent's score\n  output: Bias-corrected scores\nstep: threshold_gate (reason)\n  goal: If top score > 0.85, return immediately (high confidence)\n  output: RoutingResult or fall through to tier 2\nstep: llm_route (tool)\n  tool: chat_client\n  output: LLM picks best agent given descriptions + query (confidence gate 0.6)\nstep: keyword_fallback (reason)\n  goal: Match query keywords against agent capability lists\n  output: Final RoutingResult with method='keyword'" }
            { TaskId = "ga-router-debate"
              PatternName = "ga.routing_fallback_cascade"
              PatternTemplate = "ga.routing_fallback_cascade"
              Context = "Optional debate: run top-2 agents in parallel, CriticAgent adjudicates ties"
              Score = 0.82
              Timestamp = DateTime(2026, 3, 14)
              RollbackExpansion = None }
            { TaskId = "ga-orchestrator-streaming"
              PatternName = "ga.routing_fallback_cascade"
              PatternTemplate = "ga.routing_fallback_cascade"
              Context = "AnswerStreamingAsync uses same cascade: skills → route → agent-specific branch"
              Score = 0.85
              Timestamp = DateTime(2026, 3, 14)
              RollbackExpansion = None }
            { TaskId = "ga-orchestrator-nonstreaming"
              PatternName = "ga.routing_fallback_cascade"
              PatternTemplate = "ga.routing_fallback_cascade"
              Context = "AnswerAsync uses same cascade with full hook lifecycle at each stage"
              Score = 0.87
              Timestamp = DateTime(2026, 3, 14)
              RollbackExpansion = None }

            // ── Pattern 4: Hook Lifecycle State Machine ─────────────────────
            // Three independent hooks follow same FSM shape:
            // ConcurrentDict<Guid, State> keyed by CorrelationId, accumulate across lifecycle.
            { TaskId = "ga-hook-observability"
              PatternName = "ga.hook_lifecycle_fsm"
              PatternTemplate = "ga.hook_lifecycle_fsm"
              Context = "ObservabilityHook: start Activity on BeforeSkill, tag+dispose on AfterSkill"
              Score = 0.80
              Timestamp = DateTime(2026, 3, 14)
              RollbackExpansion = Some "step: on_request (reason)\n  goal: Initialize per-request state keyed by CorrelationId\n  output: State stored in ConcurrentDictionary\nstep: on_before_skill (reason)\n  goal: Start tracking (e.g., Activity span, timer)\n  output: Updated state with tracking handle\nstep: on_after_skill (reason)\n  goal: Accumulate result data into state (skill name, response, confidence)\n  output: Enriched state record\nstep: on_response_sent (reason)\n  goal: Emit final output (trace file, telemetry, memory write) and cleanup\n  output: Side effect + TryRemove state from dictionary" }
            { TaskId = "ga-hook-memory"
              PatternName = "ga.hook_lifecycle_fsm"
              PatternTemplate = "ga.hook_lifecycle_fsm"
              Context = "MemoryHook: enrich request with memory context, persist high-confidence responses"
              Score = 0.78
              Timestamp = DateTime(2026, 3, 14)
              RollbackExpansion = None }
            { TaskId = "ga-hook-tracebridge"
              PatternName = "ga.hook_lifecycle_fsm"
              PatternTemplate = "ga.hook_lifecycle_fsm"
              Context = "TraceBridgeHook: accumulate BridgeState across lifecycle, emit TARS-compatible JSON"
              Score = 0.82
              Timestamp = DateTime(2026, 3, 14)
              RollbackExpansion = None }

            // ── Pattern 5: Orchestrator Multi-Stage Pipeline ────────────────
            // ProductionOrchestrator has 6 stages with early-exit branches.
            // Skills checked first (fast path), then parallel routing+filtering, then agent dispatch.
            { TaskId = "ga-orchestrator-pipeline"
              PatternName = "ga.orchestrator_pipeline"
              PatternTemplate = "ga.orchestrator_pipeline"
              Context = "6-stage pipeline: hooks → skills → parallel(route,filter) → agent branch → hooks → return"
              Score = 0.90
              Timestamp = DateTime(2026, 3, 14)
              RollbackExpansion = Some "step: pre_hooks (reason)\n  goal: Run OnRequestReceived hooks (sanitize, rate-limit, auth)\n  output: Mutated message or Cancel with BlockedResponse\nstep: skill_scan (reason)\n  goal: Check each IOrchestratorSkill.CanHandle in registration order\n  output: First match short-circuits to skill execution (confidence=1.0)\nstep: parallel_classify (tool)\n  tool: task_when_all\n  output: (QueryFilters, RoutingResult) computed concurrently\nstep: agent_dispatch (reason)\n  goal: Branch by agent ID and intent: Tab→PathOptimization|TabAnalysis, else→RAG\n  output: ChatResponse from selected agent\nstep: post_hooks (reason)\n  goal: Run OnResponseSent hooks (memory, observability, trace bridge)\n  output: Side effects emitted, final response returned" }
            { TaskId = "ga-orchestrator-streaming-pipeline"
              PatternName = "ga.orchestrator_pipeline"
              PatternTemplate = "ga.orchestrator_pipeline"
              Context = "Streaming variant: same 6 stages but uses token callback for real-time output"
              Score = 0.88
              Timestamp = DateTime(2026, 3, 14)
              RollbackExpansion = None }
            { TaskId = "ga-orchestrator-agui"
              PatternName = "ga.orchestrator_pipeline"
              PatternTemplate = "ga.orchestrator_pipeline"
              Context = "AG-UI controller maps SSE events to same pipeline stages"
              Score = 0.84
              Timestamp = DateTime(2026, 3, 14)
              RollbackExpansion = None }
        ]

    /// Run the promotion pipeline on GA-discovered patterns.
    let seed (minOccurrences: int) : PromotionPipeline.PipelineResult list =
        let artifacts = gaTraceArtifacts ()
        PromotionPipeline.run minOccurrences artifacts
