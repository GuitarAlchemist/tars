import fetch from 'node-fetch';
import fs from 'fs';
import path from 'path';

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
    
    // Check if we have a report path
    if (startResponse.reportPath) {
      // Poll for the report file to be created
      let reportExists = false;
      let pollCount = 0;
      
      while (!reportExists && pollCount < 60) { // Poll for up to 5 minutes
        await new Promise(resolve => setTimeout(resolve, POLL_INTERVAL_MS));
        pollCount++;
        
        try {
          // Check if the report file exists
          if (fs.existsSync(startResponse.reportPath)) {
            reportExists = true;
            console.log(`Report file created after ${pollCount} polls!`);
            
            // Read the report file
            const reportContent = fs.readFileSync(startResponse.reportPath, 'utf8');
            console.log('Report content:', reportContent);
          } else {
            console.log(`Waiting for report file to be created (poll ${pollCount})...`);
          }
        } catch (error) {
          console.warn(`Error checking report file (${pollCount}):`, error.message);
        }
      }
      
      if (!reportExists) {
        console.log('Report file was not created after polling timeout');
      }
    } else {
      console.log('No report path provided in the response');
      
      // Poll for any changes in the target directory
      let pollCount = 0;
      
      while (pollCount < 60) { // Poll for up to 5 minutes
        await new Promise(resolve => setTimeout(resolve, POLL_INTERVAL_MS));
        pollCount++;
        
        try {
          // Check for any .bak files in the target directory
          const files = fs.readdirSync(targetDir);
          const bakFiles = files.filter(file => file.endsWith('.bak'));
          
          if (bakFiles.length > 0) {
            console.log(`Found ${bakFiles.length} .bak files in the target directory after ${pollCount} polls!`);
            console.log('Backup files:', bakFiles);
            break;
          } else {
            console.log(`No .bak files found in the target directory (poll ${pollCount})...`);
          }
        } catch (error) {
          console.warn(`Error checking target directory (${pollCount}):`, error.message);
        }
      }
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
