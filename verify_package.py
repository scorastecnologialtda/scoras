"""
This module checks for all required dependencies and verifies that the package is correctly installed.
Run this script to ensure that the Scoras package is functioning properly.
"""

import importlib
import sys

def check_dependency(module_name):
    """Check if a dependency is installed."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

def main():
    """Verify package functionality and dependencies."""
    print("Scoras Package Verification")
    print("==========================")
    
    # Check core package
    print("\nChecking core package...")
    try:
        import scoras
        print(f"✓ Scoras version: {scoras.__version__}")
    except ImportError:
        print("✗ Failed to import scoras package")
        sys.exit(1)
    
    # Check required dependencies
    required_deps = [
        "pydantic", 
        "httpx", 
        "numpy", 
        "typing_extensions", 
        "asyncio", 
        "aiohttp"
    ]
    
    print("\nChecking required dependencies...")
    all_deps_ok = True
    for dep in required_deps:
        if check_dependency(dep):
            print(f"✓ {dep}")
        else:
            print(f"✗ {dep} - MISSING")
            all_deps_ok = False
    
    # Check optional dependencies
    optional_deps = ["openai", "anthropic"]
    
    print("\nChecking optional dependencies...")
    for dep in optional_deps:
        if check_dependency(dep):
            print(f"✓ {dep}")
        else:
            print(f"- {dep} (optional)")
    
    # Check module imports
    print("\nChecking module imports...")
    modules = [
        "scoras.core", 
        "scoras.agents", 
        "scoras.rag", 
        "scoras.tools", 
        "scoras.mcp", 
        "scoras.a2a"
    ]
    
    all_modules_ok = True
    for module in modules:
        try:
            importlib.import_module(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module} - ERROR: {str(e)}")
            all_modules_ok = False
    
    # Final verdict
    print("\nVerification result:")
    if all_deps_ok and all_modules_ok:
        print("✓ All checks passed! Scoras is correctly installed and ready to use.")
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
