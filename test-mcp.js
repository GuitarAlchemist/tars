import fetch from 'node-fetch';

const TARS_MCP_URL = 'http://localhost:9000/';

async function testMcpConnection() {
  try {
    console.log('Testing connection to TARS MCP server...');
    
    const response = await fetch(TARS_MCP_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        action: 'tars',
        operation: 'capabilities'
      })
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    console.log('Connection successful!');
    console.log('TARS capabilities:', JSON.stringify(data, null, 2));
    
    return data;
  } catch (error) {
    console.error('Error connecting to TARS MCP server:', error);
    throw error;
  }
}

testMcpConnection()
  .then(data => {
    console.log('Test completed successfully');
  })
  .catch(error => {
    console.error('Test failed:', error);
  });
