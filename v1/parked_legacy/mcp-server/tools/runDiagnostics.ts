import { getComprehensiveDiagnostics, calculateOverallHealthScore } from '../src/diagnostics.js';

async function main() {
  try {
    const repoPath = process.argv[2] ?? '..';
    const diagnostics = await getComprehensiveDiagnostics(repoPath);

    // Ensure overallHealthScore is consistent (function already sets it, but double-check)
    diagnostics.overallHealthScore = calculateOverallHealthScore(diagnostics);

    console.log(JSON.stringify(diagnostics, null, 2));
  } catch (error) {
    console.error('Failed to collect diagnostics:', error);
    process.exitCode = 1;
  }
}

main();
