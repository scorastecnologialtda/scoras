// Update MkDocs configuration to include demo files
document.addEventListener('DOMContentLoaded', function() {
  // Add demo CSS to the page
  const demoCssLink = document.createElement('link');
  demoCssLink.rel = 'stylesheet';
  demoCssLink.href = '../assets/stylesheets/demo.css';
  document.head.appendChild(demoCssLink);
  
  // Add demo script tags
  const mcpDemoScript = document.createElement('script');
  mcpDemoScript.src = '../assets/javascripts/mcp-demo.js';
  document.body.appendChild(mcpDemoScript);
  
  const a2aDemoScript = document.createElement('script');
  a2aDemoScript.src = '../assets/javascripts/a2a-demo.js';
  document.body.appendChild(a2aDemoScript);
});
