// Interactive MCP Example
class MCPInteractiveDemo {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    if (!this.container) {
      console.error(`Container with ID ${containerId} not found`);
      return;
    }
    
    this.initializeUI();
    this.bindEvents();
  }
  
  initializeUI() {
    this.container.innerHTML = `
      <div class="mcp-demo">
        <div class="mcp-demo-header">
          <h3>Interactive MCP Demo</h3>
          <p>Experience how the Model Context Protocol works with Scoras</p>
        </div>
        
        <div class="mcp-demo-content">
          <div class="mcp-server-section">
            <h4>MCP Server</h4>
            <div class="mcp-server-tools">
              <h5>Available Tools:</h5>
              <ul class="mcp-tools-list">
                <li data-tool="calculator">
                  <strong>calculator</strong>: Perform calculations
                  <span class="complexity-badge simple">Simple</span>
                </li>
                <li data-tool="weather">
                  <strong>weather</strong>: Get weather information
                  <span class="complexity-badge standard">Standard</span>
                </li>
                <li data-tool="search">
                  <strong>search</strong>: Search for information
                  <span class="complexity-badge complex">Complex</span>
                </li>
              </ul>
            </div>
            <div class="mcp-server-log">
              <h5>Server Log:</h5>
              <div class="log-container" id="${containerId}-server-log"></div>
            </div>
          </div>
          
          <div class="mcp-client-section">
            <h4>MCP Client</h4>
            <div class="mcp-client-input">
              <select id="${containerId}-tool-select" class="tool-select">
                <option value="calculator">calculator</option>
                <option value="weather">weather</option>
                <option value="search">search</option>
              </select>
              
              <div class="parameter-container" id="${containerId}-calculator-params">
                <select id="${containerId}-operation" class="param-input">
                  <option value="add">add</option>
                  <option value="subtract">subtract</option>
                  <option value="multiply">multiply</option>
                  <option value="divide">divide</option>
                </select>
                <input type="number" id="${containerId}-a" placeholder="a" class="param-input" value="5">
                <input type="number" id="${containerId}-b" placeholder="b" class="param-input" value="7">
              </div>
              
              <div class="parameter-container" id="${containerId}-weather-params" style="display:none;">
                <input type="text" id="${containerId}-location" placeholder="location" class="param-input" value="New York">
              </div>
              
              <div class="parameter-container" id="${containerId}-search-params" style="display:none;">
                <input type="text" id="${containerId}-query" placeholder="query" class="param-input" value="quantum computing">
                <input type="number" id="${containerId}-max-results" placeholder="max_results" class="param-input" value="3">
              </div>
              
              <button id="${containerId}-execute-btn" class="execute-btn">Execute Tool</button>
            </div>
            
            <div class="mcp-client-result">
              <h5>Result:</h5>
              <div class="result-container" id="${containerId}-result"></div>
            </div>
          </div>
        </div>
        
        <div class="mcp-demo-footer">
          <div class="complexity-score">
            <h5>Complexity Score:</h5>
            <div class="score-container">
              <div class="score-meter">
                <div class="score-indicator" id="${containerId}-score-indicator"></div>
              </div>
              <div class="score-value" id="${containerId}-score-value">0</div>
            </div>
            <div class="score-rating" id="${containerId}-score-rating">Simple</div>
          </div>
        </div>
      </div>
    `;
    
    // Initialize the log
    this.serverLog = document.getElementById(`${containerId}-server-log`);
    this.resultContainer = document.getElementById(`${containerId}-result`);
    this.scoreIndicator = document.getElementById(`${containerId}-score-indicator`);
    this.scoreValue = document.getElementById(`${containerId}-score-value`);
    this.scoreRating = document.getElementById(`${containerId}-score-rating`);
    
    // Add initial log entry
    this.addLogEntry("Server started and ready to accept requests");
  }
  
  bindEvents() {
    const toolSelect = document.getElementById(`${this.container.id}-tool-select`);
    const executeBtn = document.getElementById(`${this.container.id}-execute-btn`);
    
    // Tool selection change
    toolSelect.addEventListener('change', () => {
      this.showParametersForTool(toolSelect.value);
    });
    
    // Execute button click
    executeBtn.addEventListener('click', () => {
      this.executeSelectedTool();
    });
  }
  
  showParametersForTool(toolName) {
    // Hide all parameter containers
    const paramContainers = this.container.querySelectorAll('.parameter-container');
    paramContainers.forEach(container => {
      container.style.display = 'none';
    });
    
    // Show the selected tool's parameters
    const selectedContainer = document.getElementById(`${this.container.id}-${toolName}-params`);
    if (selectedContainer) {
      selectedContainer.style.display = 'flex';
    }
  }
  
  executeSelectedTool() {
    const toolSelect = document.getElementById(`${this.container.id}-tool-select`);
    const selectedTool = toolSelect.value;
    
    let parameters = {};
    let result = null;
    let complexity = 0;
    
    // Get parameters based on selected tool
    if (selectedTool === 'calculator') {
      const operation = document.getElementById(`${this.container.id}-operation`).value;
      const a = parseFloat(document.getElementById(`${this.container.id}-a`).value);
      const b = parseFloat(document.getElementById(`${this.container.id}-b`).value);
      
      parameters = { operation, a, b };
      
      // Simulate calculation
      if (operation === 'add') result = a + b;
      else if (operation === 'subtract') result = a - b;
      else if (operation === 'multiply') result = a * b;
      else if (operation === 'divide') {
        if (b === 0) {
          this.addLogEntry("Error: Cannot divide by zero", "error");
          this.resultContainer.innerHTML = `<div class="error-result">Error: Cannot divide by zero</div>`;
          return;
        }
        result = a / b;
      }
      
      complexity = 1.4; // Simple tool
    }
    else if (selectedTool === 'weather') {
      const location = document.getElementById(`${this.container.id}-location`).value;
      
      parameters = { location };
      
      // Simulate weather data
      result = {
        location: location,
        temperature: Math.floor(Math.random() * 30) + 50, // 50-80°F
        conditions: ["Sunny", "Cloudy", "Rainy", "Partly Cloudy"][Math.floor(Math.random() * 4)],
        humidity: Math.floor(Math.random() * 50) + 30, // 30-80%
        wind_speed: Math.floor(Math.random() * 15) + 1 // 1-15 mph
      };
      
      complexity = 2.0; // Standard tool
    }
    else if (selectedTool === 'search') {
      const query = document.getElementById(`${this.container.id}-query`).value;
      const maxResults = parseInt(document.getElementById(`${this.container.id}-max-results`).value);
      
      parameters = { query, max_results: maxResults };
      
      // Simulate search results
      result = [];
      for (let i = 0; i < maxResults; i++) {
        result.push({
          title: `Result ${i+1} for "${query}"`,
          snippet: `This is a simulated search result about ${query}. It contains relevant information that would be useful for the user.`
        });
      }
      
      complexity = 3.0; // Complex tool
    }
    
    // Log the request
    this.addLogEntry(`Received request for tool: ${selectedTool}`);
    this.addLogEntry(`Parameters: ${JSON.stringify(parameters)}`);
    
    // Update the result
    this.resultContainer.innerHTML = `<pre>${JSON.stringify(result, null, 2)}</pre>`;
    
    // Log the response
    this.addLogEntry(`Sent response: ${JSON.stringify(result)}`);
    
    // Update complexity score
    this.updateComplexityScore(complexity);
  }
  
  addLogEntry(message, type = "info") {
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    
    const timestamp = new Date().toLocaleTimeString();
    entry.innerHTML = `<span class="timestamp">${timestamp}</span> ${message}`;
    
    this.serverLog.appendChild(entry);
    this.serverLog.scrollTop = this.serverLog.scrollHeight;
  }
  
  updateComplexityScore(additionalScore) {
    // Get current score
    let currentScore = parseFloat(this.scoreValue.textContent);
    
    // Add new score
    currentScore += additionalScore;
    
    // Update score display
    this.scoreValue.textContent = currentScore.toFixed(1);
    
    // Update score indicator
    let percentage = Math.min((currentScore / 20) * 100, 100);
    this.scoreIndicator.style.width = `${percentage}%`;
    
    // Update rating
    let rating = "Simple";
    if (currentScore >= 25) rating = "Extremely Complex";
    else if (currentScore >= 10) rating = "Complex";
    else if (currentScore >= 5) rating = "Moderate";
    
    this.scoreRating.textContent = rating;
    
    // Update color
    if (currentScore >= 25) {
      this.scoreIndicator.style.backgroundColor = "#f44336"; // Red
    } else if (currentScore >= 10) {
      this.scoreIndicator.style.backgroundColor = "#ff9800"; // Orange
    } else if (currentScore >= 5) {
      this.scoreIndicator.style.backgroundColor = "#ffc107"; // Yellow
    } else {
      this.scoreIndicator.style.backgroundColor = "#4caf50"; // Green
    }
  }
}

// Initialize the demo when the page loads
document.addEventListener('DOMContentLoaded', function() {
  const mcpDemoContainers = document.querySelectorAll('.mcp-interactive-demo');
  mcpDemoContainers.forEach((container, index) => {
    new MCPInteractiveDemo(container.id || `mcp-demo-${index}`);
  });
});
