# TARS Autonomous Instruction

**Task**: Implement Advanced Harmonic Progression Analyzer for Guitar Alchemist
**Priority**: High
**Estimated Duration**: 3-4 hours
**Complexity**: Expert
**Dependencies**: Guitar Alchemist codebase, Music theory knowledge, TARS meta-learning, Hurwitz quaternion framework

---

## OBJECTIVE

**Primary Goal**: Create an intelligent harmonic progression analyzer that can identify, analyze, and suggest chord progressions with advanced music theory understanding

**Success Criteria**:
- [ ] Real-time chord progression identification from audio input
- [ ] Functional harmony analysis (Roman numeral analysis)
- [ ] Voice leading optimization suggestions
- [ ] Modal interchange and borrowed chord detection
- [ ] Chord substitution recommendations
- [ ] Integration with Guitar Alchemist's quaternion harmonic framework
- [ ] Machine learning-based progression prediction
- [ ] Export capabilities for DAW integration

**Expected Outputs**:
- Harmonic Analyzer Engine: Real-time progression analysis system
- Roman Numeral Analyzer: Functional harmony identification
- Voice Leading Optimizer: Smooth voice leading suggestions
- Chord Substitution Engine: Advanced harmonic substitution recommendations
- Progression Predictor: ML-based next chord prediction
- Integration API: Clean interface for Guitar Alchemist integration

## CONTEXT

**Background**: Guitar Alchemist needs sophisticated harmonic analysis capabilities to help musicians understand complex chord progressions, optimize voice leading, and discover new harmonic possibilities through advanced music theory analysis.

**Constraints**:
- Real-time: Analysis must complete within 100ms for live performance use
- Accuracy: >90% accuracy in chord identification and functional analysis
- Complexity: Handle extended chords, modal interchange, and advanced harmony
- Integration: Seamless integration with existing quaternion-based harmonic framework

**Integration Points**:
- Hurwitz Quaternion Framework: Use for mathematical harmonic modeling
- Audio Processing Pipeline: Connect with real-time audio analysis
- Chord Database: Integrate with fretboard analysis system
- Machine Learning: Apply TARS meta-learning for pattern recognition

## WORKFLOW

### Phase 1: Harmonic Theory Foundation
**Objective**: Establish comprehensive harmonic analysis knowledge base
**Duration**: 45 minutes

**Steps**:
1. **Functional Harmony System**: Implement Roman numeral analysis framework
   - Action: Create comprehensive functional harmony mapping (I, ii, V7, etc.)
   - Validation: Verify all major and minor key functions are correctly mapped
   - Error Handling: Fall back to basic triadic analysis if extended harmony fails

2. **Modal System Integration**: Add modal harmony and interchange analysis
   - Action: Implement all seven modes with their characteristic chord functions
   - Validation: Verify modal chord identification accuracy
   - Error Handling: Default to parallel major/minor if modal analysis fails

3. **Advanced Harmony Concepts**: Implement extended and altered chord analysis
   - Action: Add support for 9th, 11th, 13th chords and alterations
   - Validation: Verify extended chord identification and function assignment
   - Error Handling: Simplify to basic chord types if extension analysis fails

### Phase 2: Real-Time Chord Identification
**Objective**: Implement real-time chord recognition from audio input
**Duration**: 1 hour

**Steps**:
1. **Audio Feature Extraction**: Extract harmonic content from audio signal
   - Action: Implement chromatic pitch class profile analysis
   - Validation: Verify pitch detection accuracy >85% for clean audio
   - Error Handling: Use previous chord if current analysis is uncertain

2. **Chord Template Matching**: Match audio features to chord templates
   - Action: Create comprehensive chord template database with confidence scoring
   - Validation: Verify chord identification accuracy >90% for standard progressions
   - Error Handling: Return "unknown" chord with confidence score if no match

3. **Temporal Smoothing**: Apply intelligent smoothing to reduce false positives
   - Action: Implement temporal consistency checking and chord duration analysis
   - Validation: Verify smoothing reduces false positives without missing real changes
   - Error Handling: Disable smoothing if it causes significant detection delays

### Phase 3: Functional Harmony Analysis
**Objective**: Implement intelligent Roman numeral and functional analysis
**Duration**: 1 hour

**Steps**:
1. **Key Detection Algorithm**: Identify the tonal center and key signature
   - Action: Implement Krumhansl-Schmuckler key-finding algorithm
   - Validation: Verify key detection accuracy >85% for tonal music
   - Error Handling: Use C major as default if key detection fails

2. **Roman Numeral Assignment**: Assign functional Roman numerals to identified chords
   - Action: Map chords to their functional roles within the detected key
   - Validation: Verify Roman numeral assignments match music theory principles
   - Error Handling: Use chord symbols if functional analysis is uncertain

3. **Secondary Dominants and Tonicization**: Detect temporary key changes
   - Action: Identify V/V, V/vi and other secondary dominant relationships
   - Validation: Verify secondary dominant detection enhances harmonic understanding
   - Error Handling: Treat as chromatic chords if tonicization analysis fails

### Phase 4: Voice Leading Analysis
**Objective**: Implement intelligent voice leading optimization
**Duration**: 45 minutes

**Steps**:
1. **Voice Leading Calculator**: Analyze movement between chord voicings
   - Action: Calculate voice leading efficiency and smoothness scores
   - Validation: Verify voice leading scores correlate with musical smoothness
   - Error Handling: Use simple distance metrics if complex analysis fails

2. **Optimization Suggestions**: Recommend improved voice leading options
   - Action: Generate alternative voicings with better voice leading
   - Validation: Verify suggestions improve voice leading while maintaining harmony
   - Error Handling: Keep original voicing if optimization fails

3. **Counterpoint Analysis**: Identify parallel motion and voice leading errors
   - Action: Detect parallel fifths, octaves, and other voice leading issues
   - Validation: Verify counterpoint analysis matches traditional rules
   - Error Handling: Skip counterpoint analysis if chord voicing is incomplete

### Phase 5: Advanced Harmonic Features
**Objective**: Implement chord substitution and progression prediction
**Duration**: 45 minutes

**Steps**:
1. **Chord Substitution Engine**: Generate harmonic substitution suggestions
   - Action: Implement tritone substitutions, chromatic mediants, and modal interchange
   - Validation: Verify substitutions maintain harmonic function and musical sense
   - Error Handling: Limit to basic substitutions if advanced analysis fails

2. **Progression Prediction**: Use machine learning to predict next chords
   - Action: Train model on common progression patterns and user preferences
   - Validation: Verify predictions are musically sensible and stylistically appropriate
   - Error Handling: Use statistical analysis if ML prediction fails

3. **Style Analysis**: Identify harmonic style and genre characteristics
   - Action: Analyze progression patterns to identify jazz, classical, pop, etc. styles
   - Validation: Verify style identification matches musical characteristics
   - Error Handling: Use "unknown style" if classification is uncertain

## VALIDATION

**Automated Tests**:
- [ ] Chord identification accuracy >90% on test audio samples
- [ ] Roman numeral analysis matches music theory textbook examples
- [ ] Voice leading scores correlate with musical smoothness ratings
- [ ] Chord substitutions maintain harmonic function
- [ ] Progression predictions are musically sensible
- [ ] Real-time performance <100ms latency
- [ ] Integration with quaternion framework functional

**Manual Verification**:
- [ ] Analysis results match expert music theory analysis
- [ ] Suggestions are musically useful and appropriate
- [ ] User interface is intuitive for musicians
- [ ] Audio input handling is robust across different sources

**Quality Gates**:
- Accuracy: >90% chord identification, >85% functional analysis
- Performance: <100ms real-time analysis latency
- Usability: Musicians can understand and apply analysis results
- Integration: Seamless operation with Guitar Alchemist ecosystem

## ERROR HANDLING

**Common Errors**:
- **Audio Input Failure**: No audio signal or corrupted input
  - Detection: Audio level monitoring and signal quality checks
  - Recovery: Use MIDI input or manual chord entry mode
  - Escalation: If audio system completely fails

- **Chord Recognition Failure**: Cannot identify chord from audio
  - Detection: Low confidence scores across all chord templates
  - Recovery: Display "unknown chord" with partial analysis
  - Escalation: If recognition fails for >50% of input

- **Key Detection Ambiguity**: Multiple possible keys detected
  - Detection: Similar confidence scores for multiple keys
  - Recovery: Use most likely key based on context and previous analysis
  - Escalation: If key ambiguity persists throughout entire piece

**Rollback Procedures**:
1. Maintain analysis state history for rollback capability
2. Provide manual override for all automatic analysis decisions
3. Save user corrections to improve future analysis accuracy
4. Implement graceful degradation to simpler analysis modes

## AUTONOMOUS DECISION POINTS

**AUTONOMOUS DECISION**: Analysis Complexity Level
- **Option A**: Full advanced analysis with all features enabled
- **Option B**: Simplified analysis for better performance
- **Option C**: Adaptive complexity based on input complexity
- **Decision Criteria**: Real-time performance requirements and input complexity
- **Confidence Threshold**: 85% confidence required for complexity decisions

**AUTONOMOUS DECISION**: Chord Substitution Aggressiveness
- **Option A**: Conservative substitutions maintaining original harmonic function
- **Option B**: Moderate substitutions with some harmonic reinterpretation
- **Option C**: Aggressive substitutions exploring advanced harmonic possibilities
- **Decision Criteria**: User skill level and musical context
- **Confidence Threshold**: 80% confidence required for substitution suggestions

## LEARNING OBJECTIVES

**Knowledge to Acquire**:
- [ ] Genre-specific harmonic patterns and progressions
- [ ] User preference patterns for chord substitutions
- [ ] Real-time audio processing optimization techniques
- [ ] Advanced voice leading principles and applications

**Meta-Learning Applications**:
- Apply existing music theory knowledge from TARS knowledge base
- Transfer learning from other harmonic analysis domains
- Continuous improvement of chord recognition accuracy
- Pattern recognition for style and genre identification

**Self-Improvement Targets**:
- Optimize real-time analysis algorithms for better performance
- Improve chord recognition accuracy through user feedback
- Enhance progression prediction through pattern learning
- Develop personalized analysis based on user preferences

---

**This instruction enables TARS to autonomously create a sophisticated harmonic progression analyzer that provides professional-level music theory analysis while maintaining real-time performance for live musical applications.**
