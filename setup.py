from setuptools import setup, find_packages

setup(
    name="scoras",
    version="0.1.0",
    author="Anderson L. Amaral",
    author_email="anderson.amaral@example.com",
    description="Uma biblioteca simples para criação de agentes, RAGs e ferramentas",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/andersonamaral/scoras",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pydantic>=2.0.0",
        "requests>=2.25.0",
    ],
    extras_require={
        "openai": ["openai>=1.0.0"],
        "anthropic": ["anthropic>=0.5.0"],
        "all": [
            "openai>=1.0.0",
            "anthropic>=0.5.0",
        ],
    },
)
