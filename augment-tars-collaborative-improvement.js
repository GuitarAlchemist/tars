/**
 * Augment Code and TARS CLI Collaborative Autonomous Improvement
 * 
 * This script coordinates the collaboration between Augment Code and TARS CLI
 * to autonomously improve TARS documentation and codebase using metascripts.
 */

const fs = require('fs').promises;
const path = require('path');
const tars = require('./augment-tars-integration');

// Configuration
const EXPLORATION_DIRS = [
  'C:/Users/spare/source/repos/tars/docs/Explorations/v1/Chats',
  'C:/Users/spare/source/repos/tars/docs/Explorations/Reflections'
];
const TARGET_DIRS = [
  'C:/Users/spare/source/repos/tars/TarsCli/Services',
  'C:/Users/spare/source/repos/tars/TarsCli/Commands',
  'C:/Users/spare/source/repos/tars/TarsCli/Models'
];
const IMPROVEMENT_CYCLE_DURATION_MINUTES = 60;
const MODEL = 'llama3';
const LOG_FILE = 'collaborative-improvement-log.md';

/**
 * Main function to run the collaborative improvement process
 */
async function runCollaborativeImprovement() {
  try {
    // Step 1: Log the start of the process
    await logMessage('# Augment Code and TARS CLI Collaborative Improvement\n');
    await logMessage(`Started at: ${new Date().toISOString()}\n`);
    
    // Step 2: Check TARS MCP capabilities
    await logMessage('## Checking TARS MCP Capabilities\n');
    const capabilities = await tars.getCapabilities();
    await logMessage(`Available capabilities: ${JSON.stringify(Object.keys(capabilities.capabilities))}\n`);
    
    if (!capabilities.capabilities.knowledge) {
      throw new Error('Knowledge capability not available in TARS MCP');
    }
    
    // Step 3: Extract knowledge from exploration directories
    await logMessage('## Extracting Knowledge from Exploration Directories\n');
    const knowledgeItems = await extractKnowledgeFromDirectories(EXPLORATION_DIRS);
    await logMessage(`Extracted ${knowledgeItems.length} knowledge items\n`);
    
    // Step 4: Generate a knowledge report
    await logMessage('## Generating Knowledge Report\n');
    const reportResult = await tars.generateKnowledgeReport();
    await logMessage(`Knowledge report generated: ${reportResult.reportPath}\n`);
    
    // Step 5: Run improvement cycles for each target directory
    await logMessage('## Running Improvement Cycles\n');
    for (const targetDir of TARGET_DIRS) {
      await logMessage(`### Improving ${targetDir}\n`);
      
      // Step 5.1: Generate a knowledge metascript
      await logMessage(`Generating knowledge metascript for ${targetDir}...\n`);
      const metascriptResult = await tars.generateKnowledgeMetascript(targetDir, '*.cs', MODEL);
      await logMessage(`Metascript generated: ${metascriptResult.metascriptPath}\n`);
      
      // Step 5.2: Run a knowledge improvement cycle
      for (const explorationDir of EXPLORATION_DIRS) {
        await logMessage(`Running knowledge improvement cycle: ${explorationDir} -> ${targetDir}...\n`);
        const cycleResult = await tars.runKnowledgeImprovementCycle(explorationDir, targetDir, '*.cs', MODEL);
        await logMessage(`Improvement cycle completed: ${cycleResult.reportPath}\n`);
      }
    }
    
    // Step 6: Generate a retroaction report
    await logMessage('## Generating Retroaction Report\n');
    for (const explorationDir of EXPLORATION_DIRS) {
      for (const targetDir of TARGET_DIRS) {
        await logMessage(`Generating retroaction report: ${explorationDir} -> ${targetDir}...\n`);
        const retroactionResult = await tars.generateRetroactionReport(explorationDir, targetDir, MODEL);
        await logMessage(`Retroaction report generated: ${retroactionResult.reportPath}\n`);
      }
    }
    
    // Step 7: Start TARS self-improvement
    await logMessage('## Starting TARS Self-Improvement\n');
    const selfImprovementResult = await tars.startSelfImprovement(IMPROVEMENT_CYCLE_DURATION_MINUTES, true);
    await logMessage(`Self-improvement started: ${selfImprovementResult.message}\n`);
    
    // Step 8: Monitor self-improvement status
    await logMessage('## Monitoring Self-Improvement Status\n');
    let selfImprovementCompleted = false;
    let monitoringAttempts = 0;
    
    while (!selfImprovementCompleted && monitoringAttempts < 12) {
      await new Promise(resolve => setTimeout(resolve, 5 * 60 * 1000)); // Wait 5 minutes
      monitoringAttempts++;
      
      const statusResult = await tars.getSelfImprovementStatus();
      await logMessage(`Self-improvement status (${monitoringAttempts}): ${JSON.stringify(statusResult.status)}\n`);
      
      if (!statusResult.status.isRunning) {
        selfImprovementCompleted = true;
      }
    }
    
    // Step 9: Finalize the process
    await logMessage('## Finalizing Collaborative Improvement\n');
    await logMessage(`Completed at: ${new Date().toISOString()}\n`);
    await logMessage('Collaborative improvement process completed successfully.\n');
    
    console.log('Collaborative improvement process completed successfully.');
    return true;
  } catch (error) {
    await logMessage(`## ERROR: ${error.message}\n`);
    console.error('Error in collaborative improvement process:', error);
    return false;
  }
}

/**
 * Extract knowledge from all files in the specified directories
 * @param {string[]} directories - The directories to extract knowledge from
 * @returns {Promise<Array>} - The extracted knowledge items
 */
async function extractKnowledgeFromDirectories(directories) {
  const knowledgeItems = [];
  
  for (const directory of directories) {
    await logMessage(`Extracting knowledge from ${directory}...\n`);
    
    try {
      const files = await getMarkdownFiles(directory);
      await logMessage(`Found ${files.length} markdown files\n`);
      
      for (const file of files) {
        await logMessage(`Extracting knowledge from ${file}...\n`);
        
        try {
          const result = await tars.extractKnowledge(file, MODEL);
          
          if (result.success && result.knowledge) {
            knowledgeItems.push(result.knowledge);
            await logMessage(`Successfully extracted knowledge: ${result.knowledge.Title}\n`);
          } else {
            await logMessage(`Failed to extract knowledge from ${file}: ${JSON.stringify(result)}\n`);
          }
        } catch (error) {
          await logMessage(`Error extracting knowledge from ${file}: ${error.message}\n`);
        }
      }
    } catch (error) {
      await logMessage(`Error processing directory ${directory}: ${error.message}\n`);
    }
  }
  
  return knowledgeItems;
}

/**
 * Get all markdown files in a directory
 * @param {string} directory - The directory to search
 * @returns {Promise<string[]>} - The paths of the markdown files
 */
async function getMarkdownFiles(directory) {
  const files = [];
  
  try {
    const entries = await fs.readdir(directory, { withFileTypes: true });
    
    for (const entry of entries) {
      const fullPath = path.join(directory, entry.name);
      
      if (entry.isDirectory()) {
        const subFiles = await getMarkdownFiles(fullPath);
        files.push(...subFiles);
      } else if (entry.isFile() && entry.name.endsWith('.md')) {
        files.push(fullPath);
      }
    }
  } catch (error) {
    console.error(`Error reading directory ${directory}:`, error);
  }
  
  return files;
}

/**
 * Log a message to the log file
 * @param {string} message - The message to log
 */
async function logMessage(message) {
  try {
    console.log(message);
    await fs.appendFile(LOG_FILE, message);
  } catch (error) {
    console.error('Error writing to log file:', error);
  }
}

// Run the collaborative improvement process
runCollaborativeImprovement()
  .then(success => {
    if (success) {
      console.log('Collaborative improvement process completed successfully.');
    } else {
      console.error('Collaborative improvement process failed.');
    }
  })
  .catch(error => {
    console.error('Unhandled error in collaborative improvement process:', error);
  });
