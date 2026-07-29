# Quiz: Verification Horizon

## Questions

1. **The Horizon**: What is the "Verification Horizon"?
2. **Symptoms**: What is a "Summary Wall," and why is it a sign that the verification horizon has been crossed?
3. **Evidence**: You are reviewing a claim: "The new sorting algorithm is 20% faster." What specific raw evidence would bring this claim "Inside the Horizon"?
4. **Beyond the Horizon**: Give an example of an AI task that is naturally "Beyond the Horizon" for a human maintainer without specialized tools.
5. **Tooling**: How do **Evidence Bundles** help a human reviewer stay in control when an AI is doing complex work?
6. **False Security**: Why is "High AI Accuracy" not a substitute for managing the verification horizon?

## Answer Key

1. The point where a human can no longer verify an AI's conclusion based only on a summary; they need raw, grounded evidence.
2. A "Summary Wall" is when an agent provides only a high-level conclusion (the "what") without the supporting "how" or "why." It's a sign because the human cannot see the steps taken to reach the conclusion.
3. Benchmark results (before/after), the test data used, and the specific code diff showing the algorithmic change.
4. Large-scale refactoring of 100+ files, or complex mathematical proofs/optimizations where human visual inspection is impossible.
5. Evidence bundles package the raw data, sources, and intermediate steps so the human can "drill down" into the AI's reasoning at will.
6. Because the human's role is **governance**. Even if the AI is right, the human must understand *why* to maintain the system and make informed future decisions.
