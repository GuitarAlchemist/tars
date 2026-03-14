import fetch from 'node-fetch';

const TARS_MCP_URL = 'http://localhost:9000/';

async function getKnowledgeStatus() {
  try {
    console.log('Getting knowledge improvement status...');
    
    const response = await fetch(TARS_MCP_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        action: 'knowledge',
        operation: 'status'
      })
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    console.log('Knowledge status retrieved successfully!');
    console.log('Status:', JSON.stringify(data, null, 2));
    
    return data;
  } catch (error) {
    console.error('Error getting knowledge status:', error);
    throw error;
  }
}

// Get knowledge status
getKnowledgeStatus()
  .then(data => {
    console.log('Knowledge status check completed successfully');
  })
  .catch(error => {
    console.error('Knowledge status check failed:', error);
  });
