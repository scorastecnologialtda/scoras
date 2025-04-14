from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="scoras",
    version="0.2.1",
    author="Anderson L. Amaral",
    author_email="info@scorastecnologia.com",
    description="Intelligent Agent Framework with Complexity Scoring",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/scorastecnologialtda/scoras",
    project_urls={
        "Bug Tracker": "https://github.com/scorastecnologialtda/scoras/issues",
        "Documentation": "https://github.com/scorastecnologialtda/scoras/blob/main/README.md",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
    ],
    packages=find_packages() ,  # Changed this line - removed where="."
    python_requires=">=3.8",
    install_requires=[
        "pydantic>=2.0.0",
        "httpx>=0.24.0",
        "numpy>=1.20.0",
        "typing-extensions>=4.0.0",
        "asyncio>=3.4.3",
        "aiohttp>=3.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.20.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
        ],
        "openai": ["openai>=1.0.0"],
        "anthropic": ["anthropic>=0.5.0"],
    },
) 
