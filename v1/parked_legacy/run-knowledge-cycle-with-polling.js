import fetch from 'node-fetch';

const TARS_MCP_URL = 'http://localhost:9000/';
const POLL_INTERVAL_MS = 5000; // 5 seconds

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

async function runKnowledgeImprovementCycle(explorationDir, targetDir) {
  console.log(`Running knowledge improvement cycle: ${explorationDir} -> ${targetDir}`);
  
  const requestBody = {
    action: 'knowledge',
    operation: 'cycle',
    explorationDirectory: explorationDir,
    targetDirectory: targetDir,
    pattern: '*.cs',
    model: 'llama3'
  };
  
  try {
    // Start the knowledge improvement cycle
    const startResponse = await sendMcpRequest(requestBody);
    console.log('Knowledge improvement cycle started!');
    console.log('Initial response:', JSON.stringify(startResponse, null, 2));
    
    // Poll for status updates
    let isRunning = true;
    let pollCount = 0;
    
    while (isRunning && pollCount < 60) { // Poll for up to 5 minutes
      await new Promise(resolve => setTimeout(resolve, POLL_INTERVAL_MS));
      pollCount++;
      
      try {
        // Check the status of the knowledge improvement cycle
        const statusResponse = await sendMcpRequest({
          action: 'tars',
          operation: 'status'
        });
        
        console.log(`Status update (${pollCount}):`, JSON.stringify(statusResponse, null, 2));
        
        // Check if the cycle is still running
        if (statusResponse.status && statusResponse.status.knowledgeCycleRunning === false) {
          isRunning = false;
          console.log('Knowledge improvement cycle completed!');
        }
      } catch (error) {
        console.warn(`Error getting status update (${pollCount}):`, error.message);
      }
    }
    
    if (isRunning) {
      console.log('Knowledge improvement cycle is still running after polling timeout');
    }
    
    return startResponse;
  } catch (error) {
    console.error('Error running knowledge improvement cycle:', error);
    throw error;
  }
}

// Run a knowledge improvement cycle with polling
runKnowledgeImprovementCycle('docs/Explorations/v1/Chats', 'TarsCli/Services')
  .then(data => {
    console.log('Knowledge improvement cycle process completed');
  })
  .catch(error => {
    console.error('Knowledge improvement cycle process failed:', error);
  });
