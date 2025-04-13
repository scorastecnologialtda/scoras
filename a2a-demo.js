// Interactive A2A Example
class A2AInteractiveDemo {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    if (!this.container) {
      console.error(`Container with ID ${containerId} not found`);
      return;
    }
    
    this.initializeUI();
    this.bindEvents();
    this.conversationHistory = [];
    this.totalScore = 0;
  }
  
  initializeUI() {
    this.container.innerHTML = `
      <div class="a2a-demo">
        <div class="a2a-demo-header">
          <h3>Interactive A2A Demo</h3>
          <p>Experience how the Agent-to-Agent Protocol works with Scoras</p>
        </div>
        
        <div class="a2a-demo-content">
          <div class="a2a-agents-section">
            <div class="a2a-agent-card">
              <h4>Math Agent</h4>
              <div class="agent-skills">
                <h5>Skills:</h5>
                <ul>
                  <li>
                    <strong>Mathematics</strong>
                    <span class="complexity-badge standard">Standard</span>
                    <p>Perform mathematical calculations and solve problems</p>
                  </li>
                </ul>
              </div>
              <div class="agent-status" id="${containerId}-math-status">Idle</div>
            </div>
            
            <div class="a2a-agent-card">
              <h4>Research Agent</h4>
              <div class="agent-skills">
                <h5>Skills:</h5>
                <ul>
                  <li>
                    <strong>Research</strong>
                    <span class="complexity-badge complex">Complex</span>
                    <p>Find and analyze information on various topics</p>
                  </li>
                </ul>
              </div>
              <div class="agent-status" id="${containerId}-research-status">Idle</div>
            </div>
          </div>
          
          <div class="a2a-interaction-section">
            <div class="a2a-conversation">
              <h4>Conversation</h4>
              <div class="conversation-container" id="${containerId}-conversation"></div>
            </div>
            
            <div class="a2a-input">
              <select id="${containerId}-agent-select" class="agent-select">
                <option value="coordinator">Coordinator</option>
                <option value="math">Math Agent</option>
                <option value="research">Research Agent</option>
              </select>
              <input type="text" id="${containerId}-message-input" class="message-input" placeholder="Type your message...">
              <button id="${containerId}-send-btn" class="send-btn">Send</button>
            </div>
          </div>
        </div>
        
        <div class="a2a-demo-footer">
          <div class="task-status">
            <h5>Current Task:</h5>
            <div class="task-id" id="${containerId}-task-id">No active task</div>
            <div class="task-state" id="${containerId}-task-state">-</div>
          </div>
          
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
    
    // Initialize UI elements
    this.conversationContainer = document.getElementById(`${containerId}-conversation`);
    this.messageInput = document.getElementById(`${containerId}-message-input`);
    this.mathStatus = document.getElementById(`${containerId}-math-status`);
    this.researchStatus = document.getElementById(`${containerId}-research-status`);
    this.taskId = document.getElementById(`${containerId}-task-id`);
    this.taskState = document.getElementById(`${containerId}-task-state`);
    this.scoreIndicator = document.getElementById(`${containerId}-score-indicator`);
    this.scoreValue = document.getElementById(`${containerId}-score-value`);
    this.scoreRating = document.getElementById(`${containerId}-score-rating`);
    
    // Add welcome message
    this.addMessage("system", "Welcome to the A2A Protocol Demo. You can interact with the Coordinator, Math Agent, or Research Agent.");
  }
  
  bindEvents() {
    const sendBtn = document.getElementById(`${this.container.id}-send-btn`);
    const messageInput = document.getElementById(`${this.container.id}-message-input`);
    
    // Send button click
    sendBtn.addEventListener('click', () => {
      this.sendMessage();
    });
    
    // Enter key press in input
    messageInput.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') {
        this.sendMessage();
      }
    });
  }
  
  sendMessage() {
    const agentSelect = document.getElementById(`${this.container.id}-agent-select`);
    const selectedAgent = agentSelect.value;
    const message = this.messageInput.value.trim();
    
    if (!message) return;
    
    // Add user message to conversation
    this.addMessage("user", message);
    
    // Clear input
    this.messageInput.value = "";
    
    // Process message based on selected agent
    this.processMessage(selectedAgent, message);
  }
  
  processMessage(agent, message) {
    // Generate a task ID if needed
    if (!this.currentTaskId) {
      this.currentTaskId = `task-${Date.now()}`;
      this.taskId.textContent = this.currentTaskId;
      this.taskState.textContent = "in_progress";
    }
    
    // Update agent status
    if (agent === "math" || agent === "coordinator") {
      this.mathStatus.textContent = "Processing";
      this.mathStatus.className = "agent-status processing";
    }
    
    if (agent === "research" || agent === "coordinator") {
      this.researchStatus.textContent = "Processing";
      this.researchStatus.className = "agent-status processing";
    }
    
    // Simulate processing delay
    setTimeout(() => {
      let response = "";
      let complexity = 0;
      
      if (agent === "math") {
        response = this.generateMathResponse(message);
        complexity = 2.0; // Standard complexity
        
        // Update math agent status
        this.mathStatus.textContent = "Idle";
        this.mathStatus.className = "agent-status";
      }
      else if (agent === "research") {
        response = this.generateResearchResponse(message);
        complexity = 3.0; // Complex complexity
        
        // Update research agent status
        this.researchStatus.textContent = "Idle";
        this.researchStatus.className = "agent-status";
      }
      else if (agent === "coordinator") {
        // Determine which agent to delegate to
        if (this.isMathQuery(message)) {
          // Delegate to math agent
          this.addMessage("coordinator", `I'll delegate this math question to our Math Agent.`);
          
          setTimeout(() => {
            response = this.generateMathResponse(message);
            this.addMessage("math", response);
            
            // Update math agent status
            this.mathStatus.textContent = "Idle";
            this.mathStatus.className = "agent-status";
            
            // Update complexity score for math agent
            this.updateComplexityScore(2.0);
            
            // Update task state
            this.taskState.textContent = "completed";
          }, 1500);
          
          complexity = 1.5; // Coordinator complexity
        }
        else {
          // Delegate to research agent
          this.addMessage("coordinator", `I'll delegate this research question to our Research Agent.`);
          
          setTimeout(() => {
            response = this.generateResearchResponse(message);
            this.addMessage("research", response);
            
            // Update research agent status
            this.researchStatus.textContent = "Idle";
            this.researchStatus.className = "agent-status";
            
            // Update complexity score for research agent
            this.updateComplexityScore(3.0);
            
            // Update task state
            this.taskState.textContent = "completed";
          }, 2000);
          
          complexity = 1.5; // Coordinator complexity
        }
      }
      
      // Add response to conversation if not from coordinator
      if (agent !== "coordinator" || !this.isMathQuery(message)) {
        this.addMessage(agent, response);
      }
      
      // Update complexity score
      this.updateComplexityScore(complexity);
      
    }, 1000);
  }
  
  generateMathResponse(message) {
    // Simple math response generation
    if (message.toLowerCase().includes("calculate") || 
        message.match(/[0-9]+\s*[\+\-\*\/]\s*[0-9]+/)) {
      
      // Extract numbers and operation
      const mathMatch = message.match(/([0-9]+)\s*([\+\-\*\/])\s*([0-9]+)/);
      if (mathMatch) {
        const a = parseFloat(mathMatch[1]);
        const op = mathMatch[2];
        const b = parseFloat(mathMatch[3]);
        
        let result;
        switch(op) {
          case '+': result = a + b; break;
          case '-': result = a - b; break;
          case '*': result = a * b; break;
          case '/': 
            if (b === 0) return "I cannot divide by zero.";
            result = a / b; 
            break;
        }
        
        return `The result of ${a} ${op} ${b} is ${result}.`;
      }
    }
    
    if (message.toLowerCase().includes("area") && message.toLowerCase().includes("circle")) {
      const radiusMatch = message.match(/radius\s*(?:of|is|=)?\s*([0-9]+)/i);
      if (radiusMatch) {
        const radius = parseFloat(radiusMatch[1]);
        const area = Math.PI * radius * radius;
        return `The area of a circle with radius ${radius} is ${area.toFixed(2)} square units.`;
      }
    }
    
    return "I can help with mathematical calculations. Please provide a specific math problem.";
  }
  
  generateResearchResponse(message) {
    // Simple research response generation
    if (message.toLowerCase().includes("quantum computing")) {
      return "Quantum computing is a type of computation that harnesses quantum mechanical phenomena. It uses qubits instead of classical bits, allowing for superposition and entanglement. This enables quantum computers to potentially solve certain problems much faster than classical computers.";
    }
    
    if (message.toLowerCase().includes("artificial intelligence") || message.toLowerCase().includes("ai")) {
      return "Artificial Intelligence (AI) refers to systems that can perform tasks that typically require human intelligence. These include learning, reasoning, problem-solving, perception, and language understanding. Modern AI often uses machine learning techniques, particularly deep learning, to achieve these capabilities.";
    }
    
    if (message.toLowerCase().includes("climate change")) {
      return "Climate change refers to long-term shifts in temperatures and weather patterns. Human activities have been the main driver of climate change since the 1800s, primarily due to burning fossil fuels like coal, oil, and gas, which produces heat-trapping gases. The effects include rising temperatures, more severe weather events, rising sea levels, and ecosystem disruption.";
    }
    
    return "I can research various topics for you. Please specify what information you're looking for.";
  }
  
  isMathQuery(message) {
    // Determine if a message is likely a math query
    const mathKeywords = ["calculate", "solve", "equation", "math", "formula", "area", "volume", "perimeter"];
    const mathSymbols = /[\+\-\*\/\=\^]/;
    
    const lowerMessage = message.toLowerCase();
    
    return mathKeywords.some(keyword => lowerMessage.includes(keyword)) || 
           mathSymbols.test(message) ||
           /[0-9]+\s*[\+\-\*\/]\s*[0-9]+/.test(message);
  }
  
  addMessage(sender, text) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    
    let senderName = sender.charAt(0).toUpperCase() + sender.slice(1);
    if (sender === "user") senderName = "You";
    
    messageDiv.innerHTML = `
      <div class="message-header">
        <span class="sender">${senderName}</span>
        <span class="timestamp">${new Date().toLocaleTimeString()}</span>
      </div>
      <div class="message-content">${text}</div>
    `;
    
    this.conversationContainer.appendChild(messageDiv);
    this.conversationContainer.scrollTop = this.conversationContainer.scrollHeight;
    
    // Add to conversation history
    this.conversationHistory.push({
      role: sender,
      content: text,
      timestamp: new Date()
    });
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
  const a2aDemoContainers = document.querySelectorAll('.a2a-interactive-demo');
  a2aDemoContainers.forEach((container, index) => {
    new A2AInteractiveDemo(container.id || `a2a-demo-${index}`);
  });
});
