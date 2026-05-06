# ResearchScholar_AI: Automated Research Paper Analysis System

An AI-powered system designed to automate the synthesis and evaluation of academic research papers. By leveraging a multi-agent framework powered by LangChain and LangGraph, ResearchScholar_AI improves the efficiency of literature reviews and ensures precise understanding of complex methodologies and results.

## 1. Introduction / Objective

ResearchScholar_AI is an intelligent research paper analysis platform that:
- Automates synthesis and evaluation of academic research papers
- Uses a multi-agent framework for specialized analysis
- Improves efficiency of literature reviews
- Provides precise understanding of complex methodologies and results
- Generates structured, high-level summaries for quick comprehension

## 2. Methodology / Workflow

### Input Phase
- Users upload research PDFs directly through the interface
- Integration with academic databases (ArXiv, PubMed, etc.)
- Support for multiple document formats (PDF, DOCX)

### AI Planning Phase
- Coordinator agent analyzes and routes paper sections
- Assigns specific sections to specialized domain agents
- Determines analysis strategy based on paper type and content

### Analysis Phase
- Domain agents perform multi-turn discussions
- Validates if data supports paper's conclusions
- Comprehensive evaluation of methodologies and results
- Cross-referencing of claims with evidence

### Reporting Phase
- Synthesizes findings into structured summaries
- Generates high-level overviews for quick understanding
- Creates detailed analytical reports
- Provides actionable insights and recommendations

## Project Structure

```
.
├── venv/                  # Virtual environment (created by setup.sh)
├── main.py               # Main entry point
├── requirements.txt      # Project dependencies
├── setup.sh             # Setup script to initialize environment
├── .env.example         # Example environment variables
├── .gitignore           # Git ignore file
└── README.md            # This file
```

## 3. Setup Instructions

### Prerequisites
- Python 3.8 or higher
- bash shell
- pip package manager

### Quick Setup

1. **Clone or navigate to the project directory:**
   ```bash
   cd /path/to/ResearchScholar_AI
   ```

2. **Run the setup script:**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```

4. **Configure environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

### Manual Setup (if setup.sh doesn't work)

1. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   ```

2. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 4. Usage

### Running the Application

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Run the main application
python main.py

# Or run with Streamlit interface
streamlit run main.py
```

### Processing Research Papers

1. Start the application
2. Upload a research paper (PDF or DOCX format)
3. Wait for the AI planning phase to assign sections to agents
4. Review analysis results from domain agents
5. Generate and download the comprehensive report

### Installing Additional Packages

```bash
# Install a new package (within venv)
pip install package_name

# Update requirements.txt
pip freeze > requirements.txt
```

### Deactivating the Environment

```bash
deactivate
```

## 5. Technology Stack

- **Language**: Python 3.x
- **LLM Orchestration**: LangChain & LangGraph
- **Large Language Model**: OpenAI API (GPT-4 / GPT-3.5-turbo)
- **Vector Database**: Pinecone (for storing paper embeddings)
- **Document Parsing**: PyPDF2 (PDF), python-docx (Word documents)
- **User Interface**: Streamlit (interactive web interface)
- **Data Processing**: Pandas, NumPy
- **HTTP Requests**: Requests library
- **Testing**: pytest, pytest-cov
- **Code Quality**: black, flake8, pylint
- **Environment Management**: python-dotenv

## 6. Development

### Code Quality Tools

The project includes several tools for maintaining code quality:

- **black**: Code formatter
  ```bash
  black .
  ```

- **flake8**: Linter
  ```bash
  flake8 .
  ```

- **pylint**: Code analysis
  ```bash
  pylint *.py
  ```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov
```

## 7. Environment Variables

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your configuration values:
   - **OPENAI_API_KEY**: Your OpenAI API key
   - **PINECONE_API_KEY**: Your Pinecone API key
   - **PINECONE_ENVIRONMENT**: Your Pinecone environment
   - **PINECONE_INDEX_NAME**: Name of your Pinecone index
   - **DEBUG**: Set to True for development

3. Load environment variables in your Python code:
   ```python
   from dotenv import load_dotenv
   import os
   
   load_dotenv()
   openai_api_key = os.getenv('OPENAI_API_KEY')
   pinecone_api_key = os.getenv('PINECONE_API_KEY')
   ```

## 8. Key Features

- **Multi-Agent Analysis**: Specialized agents for different research domains
- **Automated Literature Review**: Saves time on research paper analysis
- **Intelligent Synthesis**: Combines findings from multiple papers
- **Data Validation**: Verifies if conclusions are supported by data
- **Structured Reports**: High-level summaries and detailed analyses
- **Vector Database Integration**: Efficient paper embedding storage and retrieval
- **User-Friendly Interface**: Streamlit-based web application

## 9. Workflow Diagram

```
User Upload → Input Processing → AI Planning Phase
                                       ↓
                            Section Assignment to Agents
                                       ↓
                            Domain Agent Analysis (Multi-turn)
                                       ↓
                            Validation & Cross-referencing
                                       ↓
                            Report Generation
                                       ↓
                            User Download/View
```

## 10. Common Issues

### Virtual Environment Not Activating
- Make sure you're in the project directory
- Check that `venv/` folder exists
- Try: `source venv/bin/activate`

### Module Not Found Errors
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt` again
- Check for typos in import statements

### API Key Issues
- Verify OPENAI_API_KEY is set in `.env`
- Verify PINECONE_API_KEY is set in `.env`
- Check that keys have proper permissions

## 11. Project Dependencies

All project dependencies are listed in `requirements.txt`. Main packages include:

- **langchain**: LLM orchestration framework
- **langgraph**: Graph-based workflow orchestration
- **openai**: OpenAI API client
- **pinecone-client**: Vector database client
- **PyPDF2**: PDF document parsing
- **python-docx**: Word document parsing
- **streamlit**: Web UI framework
- **pandas**: Data analysis library
- **pytest**: Testing framework

## 12. Next Steps

1. Configure API keys in `.env` file
2. Implement coordinator agent logic
3. Develop domain-specific analysis agents
4. Build the Streamlit UI for paper uploads
5. Implement vector database integration
6. Add comprehensive test coverage
7. Optimize performance for large document sets

## 13. License

[Add your license information here]

## 14. Contact & Support

[Add your contact information here]

## 15. Contributing

We welcome contributions! Please follow these steps:

1. Create a new branch for your feature
2. Make your changes
3. Run tests and code quality checks
4. Submit a pull request

---

**Last Updated**: March 2, 2026
**Version**: 1.0.0

