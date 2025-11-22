/**
 * Augment Code Integration with TARS
 *
 * This script provides functions for Augment Code to interact with TARS via the MCP protocol.
 */

// Configuration
const TARS_MCP_URL = 'http://localhost:9000/';

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
 * Get the capabilities of the TARS MCP server
 * @returns {Promise<Object>} - The capabilities of the server
 */
async function getCapabilities() {
  const requestBody = {
    action: 'tars',
    operation: 'capabilities'
  };

  return await sendMcpRequest(requestBody);
}

/**
 * Extract knowledge from a file
 * @param {string} filePath - The path to the file
 * @param {string} model - The model to use (default: 'llama3')
 * @returns {Promise<Object>} - The extracted knowledge
 */
async function extractKnowledge(filePath, model = 'llama3') {
  const requestBody = {
    action: 'knowledge',
    operation: 'extract',
    filePath,
    model
  };

  return await sendMcpRequest(requestBody);
}

/**
 * Apply knowledge to improve a file
 * @param {string} filePath - The path to the file
 * @param {string} model - The model to use (default: 'llama3')
 * @returns {Promise<Object>} - The result of the operation
 */
async function applyKnowledge(filePath, model = 'llama3') {
  const requestBody = {
    action: 'knowledge',
    operation: 'apply',
    filePath,
    model
  };

  return await sendMcpRequest(requestBody);
}

/**
 * Generate a knowledge report
 * @returns {Promise<Object>} - The result of the operation
 */
async function generateKnowledgeReport() {
  const requestBody = {
    action: 'knowledge',
    operation: 'report'
  };

  return await sendMcpRequest(requestBody);
}

/**
 * Generate a knowledge metascript
 * @param {string} targetDirectory - The directory to target
 * @param {string} pattern - The file pattern to match (default: '*.cs')
 * @param {string} model - The model to use (default: 'llama3')
 * @returns {Promise<Object>} - The result of the operation
 */
async function generateKnowledgeMetascript(targetDirectory, pattern = '*.cs', model = 'llama3') {
  const requestBody = {
    action: 'knowledge',
    operation: 'metascript',
    targetDirectory,
    pattern,
    model
  };

  return await sendMcpRequest(requestBody);
}

/**
 * Run a knowledge improvement cycle
 * @param {string} explorationDirectory - The directory containing exploration files
 * @param {string} targetDirectory - The directory to target
 * @param {string} pattern - The file pattern to match (default: '*.cs')
 * @param {string} model - The model to use (default: 'llama3')
 * @returns {Promise<Object>} - The result of the operation
 */
async function runKnowledgeImprovementCycle(explorationDirectory, targetDirectory, pattern = '*.cs', model = 'llama3') {
  const requestBody = {
    action: 'knowledge',
    operation: 'cycle',
    explorationDirectory,
    targetDirectory,
    pattern,
    model
  };

  return await sendMcpRequest(requestBody);
}

/**
 * Generate a retroaction report
 * @param {string} explorationDirectory - The directory containing exploration files
 * @param {string} targetDirectory - The directory containing target files
 * @param {string} model - The model to use (default: 'llama3')
 * @returns {Promise<Object>} - The result of the operation
 */
async function generateRetroactionReport(explorationDirectory, targetDirectory, model = 'llama3') {
  const requestBody = {
    action: 'knowledge',
    operation: 'retroaction',
    explorationDirectory,
    targetDirectory,
    model
  };

  return await sendMcpRequest(requestBody);
}

/**
 * List all knowledge items
 * @returns {Promise<Object>} - The result of the operation
 */
async function listKnowledgeItems() {
  const requestBody = {
    action: 'knowledge',
    operation: 'list'
  };

  return await sendMcpRequest(requestBody);
}

/**
 * Generate text with a model
 * @param {string} prompt - The prompt to generate from
 * @param {string} model - The model to use (default: 'llama3')
 * @returns {Promise<Object>} - The result of the operation
 */
async function generateText(prompt, model = 'llama3') {
  const requestBody = {
    action: 'ollama',
    operation: 'generate',
    prompt,
    model
  };

  return await sendMcpRequest(requestBody);
}

/**
 * Get available models
 * @returns {Promise<Object>} - The result of the operation
 */
async function getModels() {
  const requestBody = {
    action: 'ollama',
    operation: 'models'
  };

  return await sendMcpRequest(requestBody);
}

/**
 * Start self-improvement
 * @param {number} duration - Duration in minutes
 * @param {boolean} autoAccept - Whether to auto-accept improvements (default: false)
 * @returns {Promise<Object>} - The result of the operation
 */
async function startSelfImprovement(duration, autoAccept = false) {
  const requestBody = {
    action: 'self-improve',
    operation: 'start',
    duration,
    autoAccept
  };

  return await sendMcpRequest(requestBody);
}

/**
 * Get self-improvement status
 * @returns {Promise<Object>} - The result of the operation
 */
async function getSelfImprovementStatus() {
  const requestBody = {
    action: 'self-improve',
    operation: 'status'
  };

  return await sendMcpRequest(requestBody);
}

/**
 * Stop self-improvement
 * @returns {Promise<Object>} - The result of the operation
 */
async function stopSelfImprovement() {
  const requestBody = {
    action: 'self-improve',
    operation: 'stop'
  };

  return await sendMcpRequest(requestBody);
}

// Export all functions
module.exports = {
  getCapabilities,
  extractKnowledge,
  applyKnowledge,
  generateKnowledgeReport,
  generateKnowledgeMetascript,
  runKnowledgeImprovementCycle,
  generateRetroactionReport,
  listKnowledgeItems,
  generateText,
  getModels,
  startSelfImprovement,
  getSelfImprovementStatus,
  stopSelfImprovement
};
