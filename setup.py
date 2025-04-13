from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8")  as fh:
    long_description = fh.read()

setup(
    name="scoras",
    version="0.1.0",
    author="Anderson L. Amaral",
    author_email="luis.anderson.sp@gmail.com",
    description="Intelligent Agent Framework with Complexity Scoring",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/andersonamaral2/scoras",
    project_urls={
        "Documentation": "https://yrgwfhnc.manus.space/",
        "Bug Tracker": "https://github.com/andersonamaral2/scoras/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=find_packages() ,
    python_requires=">=3.8",
    install_requires=[
        "pydantic>=2.0.0",
        "httpx>=0.24.0",
        "typing-extensions>=4.0.0",
        "aiohttp>=3.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
        ],
    },
) 
