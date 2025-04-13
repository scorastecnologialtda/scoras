# Installation

Getting started with Scoras is simple. This guide will walk you through the installation process and basic setup.

## Requirements

Scoras requires Python 3.8 or higher and works on Windows, macOS, and Linux.

## Installation via pip

The recommended way to install Scoras is via pip:

```bash
pip install scoras
```

This will install Scoras and its core dependencies.

## Installation with Optional Dependencies

To install Scoras with support for specific LLM providers:

```bash
# For OpenAI support
pip install "scoras[openai]"

# For Anthropic support
pip install "scoras[anthropic]"

# For Google Gemini support
pip install "scoras[gemini]"

# For all providers
pip install "scoras[all]"
```

## Installing from Source

To install the latest development version from GitHub:

```bash
git clone https://github.com/scoras/scoras.git
cd scoras
pip install -e .
```

## Environment Setup

Scoras uses environment variables for API keys. Set these up before using the corresponding providers:

```bash
# For OpenAI
export OPENAI_API_KEY="your-api-key"

# For Anthropic
export ANTHROPIC_API_KEY="your-api-key"

# For Google Gemini
export GOOGLE_API_KEY="your-api-key"
```

On Windows, use:

```cmd
set OPENAI_API_KEY=your-api-key
```

Or permanently set them in your environment variables through system settings.

## Verifying Installation

To verify that Scoras is installed correctly, run:

```python
import scoras

print(f"Scoras version: {scoras.__version__}")
```

If this runs without errors, you're ready to start using Scoras!
