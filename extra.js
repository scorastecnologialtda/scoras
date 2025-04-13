// Scoras custom JavaScript

document.addEventListener('DOMContentLoaded', function() {
  // Initialize code copy buttons with custom styling
  document.querySelectorAll('pre > code').forEach(function(codeBlock) {
    const container = codeBlock.parentNode;
    const copyButton = document.createElement('button');
    copyButton.className = 'md-clipboard md-icon';
    copyButton.title = 'Copy to clipboard';
    copyButton.setAttribute('data-clipboard-target', 'pre > code');
    
    copyButton.addEventListener('click', function() {
      const code = codeBlock.textContent;
      navigator.clipboard.writeText(code).then(function() {
        copyButton.title = 'Copied!';
        setTimeout(function() {
          copyButton.title = 'Copy to clipboard';
        }, 2000);
      });
    });
    
    container.appendChild(copyButton);
  });
  
  // Add syntax highlighting for Scoras-specific code
  document.querySelectorAll('code').forEach(function(codeBlock) {
    const text = codeBlock.textContent;
    
    // Highlight Scoras-specific keywords
    const scorasKeywords = [
      'Agent', 'tool', 'WorkflowGraph', 'Document', 'SimpleRAG', 'SemanticRAG', 
      'ContextualRAG', 'ExpertAgent', 'CreativeAgent', 'RAGAgent', 'MultiAgentSystem',
      'ToolChain', 'ToolRouter', 'complexity_score', 'MCPClient', 'A2AClient'
    ];
    
    scorasKeywords.forEach(function(keyword) {
      const regex = new RegExp(`\\b${keyword}\\b`, 'g');
      codeBlock.innerHTML = codeBlock.innerHTML.replace(
        regex, 
        `<span class="scoras-keyword">${keyword}</span>`
      );
    });
  });
  
  // Add interactive tabs for protocol examples
  const protocolTabs = document.querySelectorAll('.scoras-protocol-tabs');
  protocolTabs.forEach(function(tabContainer) {
    const tabs = tabContainer.querySelectorAll('.scoras-tab');
    const contents = tabContainer.querySelectorAll('.scoras-tab-content');
    
    tabs.forEach(function(tab, index) {
      tab.addEventListener('click', function() {
        // Remove active class from all tabs and contents
        tabs.forEach(t => t.classList.remove('active'));
        contents.forEach(c => c.classList.remove('active'));
        
        // Add active class to clicked tab and corresponding content
        tab.classList.add('active');
        contents[index].classList.add('active');
      });
    });
    
    // Activate first tab by default
    if (tabs.length > 0) {
      tabs[0].click();
    }
  });
  
  // Add smooth scrolling for navigation
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
      e.preventDefault();
      
      const targetId = this.getAttribute('href');
      const targetElement = document.querySelector(targetId);
      
      if (targetElement) {
        window.scrollTo({
          top: targetElement.offsetTop - 100,
          behavior: 'smooth'
        });
      }
    });
  });
  
  // Add complexity score visualization
  document.querySelectorAll('.scoras-complexity-meter').forEach(function(meter) {
    const score = parseFloat(meter.getAttribute('data-score'));
    const maxScore = parseFloat(meter.getAttribute('data-max-score') || '100');
    const percentage = (score / maxScore) * 100;
    
    const indicator = document.createElement('div');
    indicator.className = 'scoras-complexity-indicator';
    indicator.style.width = `${percentage}%`;
    
    // Set color based on complexity
    if (percentage < 30) {
      indicator.style.backgroundColor = '#4caf50'; // Green for simple
    } else if (percentage < 60) {
      indicator.style.backgroundColor = '#ff9800'; // Orange for moderate
    } else {
      indicator.style.backgroundColor = '#f44336'; // Red for complex
    }
    
    meter.appendChild(indicator);
  });
});
