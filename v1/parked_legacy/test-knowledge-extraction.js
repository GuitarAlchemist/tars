import fetch from 'node-fetch';

const TARS_MCP_URL = 'http://localhost:9000/';

async function extractKnowledge(filePath) {
  try {
    console.log(`Extracting knowledge from: ${filePath}`);
    
    const response = await fetch(TARS_MCP_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        action: 'knowledge',
        operation: 'extract',
        filePath: filePath,
        model: 'llama3'
      })
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    console.log('Knowledge extraction successful!');
    console.log('Extracted knowledge:', JSON.stringify(data, null, 2));
    
    return data;
  } catch (error) {
    console.error('Error extracting knowledge:', error);
    throw error;
  }
}

// Extract knowledge from an exploration file
extractKnowledge('docs/Explorations/v1/Chats/ChatGPT-TARS Project Implications.md')
  .then(data => {
    console.log('Knowledge extraction completed successfully');
  })
  .catch(error => {
    console.error('Knowledge extraction failed:', error);
  });
