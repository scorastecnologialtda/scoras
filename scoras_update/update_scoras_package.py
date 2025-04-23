#!/usr/bin/env python3
"""
Script to update the Scoras package with the complete implementation of missing classes and functions.
This script will:
1. Update core.py, rag.py, and tools.py with the new implementations
2. Update __init__.py to expose all the new classes and functions
3. Update the version number to 0.3.1
"""

import os
import shutil
import sys

def update_file(source_file, target_file, backup=True):
    """Update a file with new content, creating a backup if requested."""
    if backup and os.path.exists(target_file):
        backup_file = f"{target_file}.bak"
        print(f"Creating backup of {target_file} to {backup_file}")
        shutil.copy2(target_file, backup_file)
    
    print(f"Updating {target_file} with content from {source_file}")
    shutil.copy2(source_file, target_file)

def update_init_file(init_file, version="0.3.1"):
    """Update the __init__.py file with the new version number and imports."""
    with open(init_file, 'r') as f:
        content = f.read()
    
    # Update version number
    content = content.replace('__version__ = "0.2.0"', f'__version__ = "{version}"')
    content = content.replace('__version__ = "0.3.0"', f'__version__ = "{version}"')
    
    # Ensure all classes are imported and exposed
    imports = [
        "from scoras.core import Graph, Node, Edge, Message, Tool, RAG, ScoreTracker, ScorasConfig, WorkflowGraph, ScoringMixin",
        "from scoras.agents import Agent, ExpertAgent, CreativeAgent, MultiAgentSystem, AgentTeam",
        "from scoras.rag import SimpleRAG, ContextualRAG, Document, Retriever",
        "from scoras.tools import tool, ToolChain, ToolRouter, ToolBuilder, ToolResult",
        "from scoras.mcp import MCPServer, MCPClient, MCPSkill, create_mcp_server, create_mcp_client",
        "from scoras.a2a import A2AAgent, A2ANetwork, A2AHub, create_a2a_agent, create_a2a_network"
    ]
    
    # Replace import lines
    for i, line in enumerate(imports):
        if i == 0:  # First import line
            content = content.split("from scoras.core")[0] + line + "\n"
            continue
        
        module_name = line.split("from scoras.")[1].split(" import")[0]
        if f"from scoras.{module_name}" in content:
            # Replace existing import line
            lines = content.split("\n")
            for j, l in enumerate(lines):
                if f"from scoras.{module_name}" in l:
                    lines[j] = line
                    break
            content = "\n".join(lines)
        else:
            # Add new import line after the last import
            last_import_idx = content.rfind("from scoras.")
            last_import_line_end = content.find("\n", last_import_idx)
            content = content[:last_import_line_end+1] + line + "\n" + content[last_import_line_end+1:]
    
    with open(init_file, 'w') as f:
        f.write(content)
    
    print(f"Updated {init_file} with version {version} and all necessary imports")

def main():
    """Main function to update the Scoras package."""
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Source files (new implementations)
    core_update = os.path.join(script_dir, "core_update.py")
    rag_update = os.path.join(script_dir, "rag_update.py")
    tools_update = os.path.join(script_dir, "tools_update.py")
    
    # Target files (in the package)
    package_dir = os.path.expanduser("~/scoras")
    if not os.path.exists(package_dir):
        print(f"Error: Package directory {package_dir} not found")
        sys.exit(1)
    
    scoras_dir = os.path.join(package_dir, "scoras")
    core_file = os.path.join(scoras_dir, "core.py")
    rag_file = os.path.join(scoras_dir, "rag.py")
    tools_file = os.path.join(scoras_dir, "tools.py")
    init_file = os.path.join(scoras_dir, "__init__.py")
    
    # Update files
    update_file(core_update, core_file)
    update_file(rag_update, rag_file)
    update_file(tools_update, tools_file)
    update_init_file(init_file, version="0.3.1")
    
    # Update setup.py version
    setup_file = os.path.join(package_dir, "setup.py")
    if os.path.exists(setup_file):
        with open(setup_file, 'r') as f:
            setup_content = f.read()
        
        # Update version number
        setup_content = setup_content.replace('version="0.2.0"', 'version="0.3.1"')
        setup_content = setup_content.replace('version="0.2.1"', 'version="0.3.1"')
        setup_content = setup_content.replace('version="0.2.2"', 'version="0.3.1"')
        setup_content = setup_content.replace('version="0.2.3"', 'version="0.3.1"')
        setup_content = setup_content.replace('version="0.3.0"', 'version="0.3.1"')
        
        with open(setup_file, 'w') as f:
            f.write(setup_content)
        
        print(f"Updated {setup_file} with version 0.3.1")
    
    # Update pyproject.toml version if it exists
    pyproject_file = os.path.join(package_dir, "pyproject.toml")
    if os.path.exists(pyproject_file):
        with open(pyproject_file, 'r') as f:
            pyproject_content = f.read()
        
        # Check if version is specified in pyproject.toml
        if 'version = ' in pyproject_content:
            # Update version number
            import re
            pyproject_content = re.sub(r'version = "[0-9]+\.[0-9]+\.[0-9]+"', 'version = "0.3.1"', pyproject_content)
            
            with open(pyproject_file, 'w') as f:
                f.write(pyproject_content)
            
            print(f"Updated {pyproject_file} with version 0.3.1")
    
    print("\nScoras package has been successfully updated to version 0.3.1")
    print("The package now includes all the missing classes and functions:")
    print("- Message, Tool, RAG, ScoreTracker, ScorasConfig, WorkflowGraph in core.py")
    print("- Document, SimpleRAG in rag.py")
    print("- tool decorator in tools.py")
    print("\nTo build and publish the updated package, run:")
    print("cd ~/scoras")
    print("rm -rf dist/ build/ *.egg-info/")
    print("python -m build")
    print("twine upload dist/*")

if __name__ == "__main__":
    main()
