import fetch from 'node-fetch';

const TARS_MCP_URL = 'http://localhost:9000/';

async function runKnowledgeImprovementCycle(explorationDir, targetDir) {
  try {
    console.log(`Running knowledge improvement cycle: ${explorationDir} -> ${targetDir}`);
    
    const response = await fetch(TARS_MCP_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        action: 'knowledge',
        operation: 'cycle',
        explorationDirectory: explorationDir,
        targetDirectory: targetDir,
        pattern: '*.cs',
        model: 'llama3'
      })
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    console.log('Knowledge improvement cycle successful!');
    console.log('Cycle result:', JSON.stringify(data, null, 2));
    
    return data;
  } catch (error) {
    console.error('Error running knowledge improvement cycle:', error);
    throw error;
  }
}

// Run a knowledge improvement cycle
runKnowledgeImprovementCycle('docs/Explorations/v1/Chats', 'TarsCli/Services')
  .then(data => {
    console.log('Knowledge improvement cycle completed successfully');
  })
  .catch(error => {
    console.error('Knowledge improvement cycle failed:', error);
  });
