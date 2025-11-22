/**
 * Augment Code Integration with TARS via MCP
 *
 * This script provides functions for Augment Code to interact with TARS via the MCP protocol.
 */

import fetch from 'node-fetch';
import { promises as fs } from 'fs';
import path from 'path';

// Configuration
const TARS_MCP_URL = 'http://localhost:9000/';
const EXPLORATION_DIRS = [
  'docs/Explorations/v1/Chats',
  'docs/Explorations/Reflections'
];
const TARGET_DIRS = [
  'TarsCli/Services',
  'TarsCli/Commands',
  'TarsCli/Models'
];

/**
 * Send a request to the TARS MCP server
 * @param {Object} requestBody - The request body
 * @returns {Promise<Object>} - The response from the server
 */
async function sendMcpRequest(requestBody) {
  try {
    const response = await fetch(TARS_MCP_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(requestBody)
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error sending MCP request:', error);
    throw error;
  }
}

/**
 * Start the autonomous improvement process
 * @param {string[]} explorationDirs - Directories containing exploration files
 * @param {string[]} targetDirs - Directories to target with improvements
 * @param {number} durationMinutes - Duration of the improvement process in minutes
 * @param {string} model - Model to use for improvement
 * @param {boolean} autoCommit - Whether to automatically commit improvements
 * @param {boolean} createPullRequest - Whether to create a pull request for improvements
 * @returns {Promise<Object>} - The result of the operation
 */
async function startAutonomousImprovement(
  explorationDirs = EXPLORATION_DIRS,
  targetDirs = TARGET_DIRS,
  durationMinutes = 60,
  model = 'llama3',
  autoCommit = false,
  createPullRequest = false
) {
  console.log('Starting autonomous improvement process...');
  console.log(`Exploration directories: ${explorationDirs.join(', ')}`);
  console.log(`Target directories: ${targetDirs.join(', ')}`);
  console.log(`Duration: ${durationMinutes} minutes`);
  console.log(`Model: ${model}`);
  console.log(`Auto-commit: ${autoCommit}`);
  console.log(`Create PR: ${createPullRequest}`);

  const requestBody = {
    action: 'tars',
    operation: 'autonomous_improvement',
    exploration_dirs: explorationDirs,
    target_dirs: targetDirs,
    duration_minutes: durationMinutes,
    model: model,
    auto_commit: autoCommit,
    create_pull_request: createPullRequest
  };

  try {
    const response = await sendMcpRequest(requestBody);
    console.log('Autonomous improvement process started successfully');
    return response;
  } catch (error) {
    console.error('Error starting autonomous improvement process:', error);
    throw error;
  }
}

/**
 * Get the status of the autonomous improvement process
 * @returns {Promise<Object>} - The status of the autonomous improvement process
 */
async function getAutonomousImprovementStatus() {
  console.log('Getting autonomous improvement status...');

  const requestBody = {
    action: 'tars',
    operation: 'autonomous_improvement_status'
  };

  try {
    const response = await sendMcpRequest(requestBody);
    console.log('Autonomous improvement status retrieved successfully');
    return response;
  } catch (error) {
    console.error('Error getting autonomous improvement status:', error);
    throw error;
  }
}

/**
 * Stop the autonomous improvement process
 * @returns {Promise<Object>} - The result of the operation
 */
async function stopAutonomousImprovement() {
  console.log('Stopping autonomous improvement process...');

  const requestBody = {
    action: 'tars',
    operation: 'stop_autonomous_improvement'
  };

  try {
    const response = await sendMcpRequest(requestBody);
    console.log('Autonomous improvement process stopped successfully');
    return response;
  } catch (error) {
    console.error('Error stopping autonomous improvement process:', error);
    throw error;
  }
}

/**
 * Extract knowledge from a file
 * @param {string} filePath - Path to the file
 * @param {string} model - Model to use for extraction
 * @returns {Promise<Object>} - The extracted knowledge
 */
async function extractKnowledge(filePath, model = 'llama3') {
  console.log(`Extracting knowledge from: ${filePath}`);

  const requestBody = {
    action: 'knowledge',
    operation: 'extract',
    filePath: filePath,
    model: model
  };

  try {
    const response = await sendMcpRequest(requestBody);
    console.log('Knowledge extracted successfully');
    return response;
  } catch (error) {
    console.error('Error extracting knowledge:', error);
    throw error;
  }
}

/**
 * Apply knowledge to improve a file
 * @param {string} filePath - Path to the file
 * @param {string} model - Model to use for improvement
 * @returns {Promise<Object>} - The result of the operation
 */
async function applyKnowledge(filePath, model = 'llama3') {
  console.log(`Applying knowledge to: ${filePath}`);

  const requestBody = {
    action: 'knowledge',
    operation: 'apply',
    filePath: filePath,
    model: model
  };

  try {
    const response = await sendMcpRequest(requestBody);
    console.log('Knowledge applied successfully');
    return response;
  } catch (error) {
    console.error('Error applying knowledge:', error);
    throw error;
  }
}

/**
 * Run a knowledge improvement cycle
 * @param {string} explorationDir - Directory containing exploration files
 * @param {string} targetDir - Directory to target with improvements
 * @param {string} pattern - File pattern to match
 * @param {string} model - Model to use for the cycle
 * @returns {Promise<Object>} - The result of the operation
 */
async function runKnowledgeImprovementCycle(
  explorationDir,
  targetDir,
  pattern = '*.cs',
  model = 'llama3'
) {
  console.log(`Running knowledge improvement cycle: ${explorationDir} -> ${targetDir}`);

  const requestBody = {
    action: 'knowledge',
    operation: 'cycle',
    explorationDirectory: explorationDir,
    targetDirectory: targetDir,
    pattern: pattern,
    model: model
  };

  try {
    const response = await sendMcpRequest(requestBody);
    console.log('Knowledge improvement cycle completed successfully');
    return response;
  } catch (error) {
    console.error('Error running knowledge improvement cycle:', error);
    throw error;
  }
}

/**
 * Generate a retroaction report
 * @param {string} explorationDir - Directory containing exploration files
 * @param {string} targetDir - Directory containing target files
 * @param {string} model - Model to use for the report
 * @returns {Promise<Object>} - The result of the operation
 */
async function generateRetroactionReport(
  explorationDir,
  targetDir,
  model = 'llama3'
) {
  console.log(`Generating retroaction report: ${explorationDir} -> ${targetDir}`);

  const requestBody = {
    action: 'knowledge',
    operation: 'retroaction',
    explorationDirectory: explorationDir,
    targetDirectory: targetDir,
    model: model
  };

  try {
    const response = await sendMcpRequest(requestBody);
    console.log('Retroaction report generated successfully');
    return response;
  } catch (error) {
    console.error('Error generating retroaction report:', error);
    throw error;
  }
}

/**
 * Run a TARS metascript
 * @param {string} metascriptPath - Path to the metascript
 * @param {boolean} verbose - Whether to enable verbose logging
 * @returns {Promise<Object>} - The result of the operation
 */
async function runMetascript(metascriptPath, verbose = true) {
  console.log(`Running metascript: ${metascriptPath}`);

  const requestBody = {
    action: 'tars',
    operation: 'run_metascript',
    metascript_path: metascriptPath,
    verbose: verbose
  };

  try {
    const response = await sendMcpRequest(requestBody);
    console.log('Metascript executed successfully');
    return response;
  } catch (error) {
    console.error('Error running metascript:', error);
    throw error;
  }
}

/**
 * Main function to run the collaborative improvement process
 */
async function runCollaborativeImprovement() {
  try {
    console.log('Starting collaborative improvement process...');

    // Step 1: Start the autonomous improvement process
    await startAutonomousImprovement(
      EXPLORATION_DIRS,
      TARGET_DIRS,
      60,
      'llama3',
      false,
      false
    );

    // Step 2: Monitor the autonomous improvement process
    let isRunning = true;
    let monitoringAttempts = 0;

    while (isRunning && monitoringAttempts < 12) {
      await new Promise(resolve => setTimeout(resolve, 5 * 60 * 1000)); // Wait 5 minutes
      monitoringAttempts++;

      const status = await getAutonomousImprovementStatus();
      console.log(`Autonomous improvement status (${monitoringAttempts}):`, status);

      if (!status.is_running) {
        isRunning = false;
      }
    }

    // Step 3: Run a knowledge improvement cycle for each exploration directory
    for (const explorationDir of EXPLORATION_DIRS) {
      for (const targetDir of TARGET_DIRS) {
        await runKnowledgeImprovementCycle(explorationDir, targetDir);
      }
    }

    // Step 4: Generate retroaction reports
    for (const explorationDir of EXPLORATION_DIRS) {
      for (const targetDir of TARGET_DIRS) {
        await generateRetroactionReport(explorationDir, targetDir);
      }
    }

    // Step 5: Run the autonomous improvement metascript
    await runMetascript('TarsCli/Metascripts/autonomous_improvement.tars');

    console.log('Collaborative improvement process completed successfully');
  } catch (error) {
    console.error('Error in collaborative improvement process:', error);
  }
}

// Export functions
export {
  startAutonomousImprovement,
  getAutonomousImprovementStatus,
  stopAutonomousImprovement,
  extractKnowledge,
  applyKnowledge,
  runKnowledgeImprovementCycle,
  generateRetroactionReport,
  runMetascript,
  runCollaborativeImprovement
};

// Run the collaborative improvement process if this script is executed directly
runCollaborativeImprovement();
