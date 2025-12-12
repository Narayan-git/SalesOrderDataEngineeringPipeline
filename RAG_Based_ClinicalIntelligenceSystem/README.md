# Clinical Intelligence System: RAG-based Capstone Project

## 📋 Overview

This project implements a **Retrieval-Augmented Generation (RAG) pipeline** for building a Clinical Intelligence System. The system leverages a knowledge base of 100 clinical documents from the National Library of Medicine (NIH) as the exclusive source of truth and uses advanced retrieval and generation techniques to answer clinical questions with high accuracy and reliability.

The RAG approach combines semantic document retrieval with large language models (Azure OpenAI GPT-4) to provide evidence-based clinical answers while minimizing hallucinations and maintaining factual grounding.

---

## 🎯 Project Goals

1. Build an intelligent question-answering system for clinical and genetic information
2. Implement multiple retrieval strategies (semantic, hybrid, BM25)
3. Generate answers strictly based on clinical documents
4. Validate system performance using DeepEval metrics
5. Handle complex clinical queries with nuanced context understanding

---

## 📁 Project Structure

```
RAG_Based_ClinicalIntelligenceSystem/
├── README.md                              # This file
├── capstone_setup.ipynb                   # Environment and dependency setup
├── code.ipynb                             # Main RAG pipeline implementation
├── usis.env                               # Environment configuration (template)
├── Capstone_1_Learner_Instruction_card.pdf # Project requirements and guidelines
└── Data/
    ├── capstone1_rag_dataset.csv          # 100+ clinical documents (knowledge base)
    ├── capstone1_rag_test_questions.csv   # Test questions for evaluation
    ├── capstone1_rag_validation.csv       # Validation set with reference answers
    └── output_data.csv                    # Generated predictions/results
```

---

## 🔧 Technology Stack

### Core Libraries
- **LangChain**: RAG framework and document processing
- **Azure OpenAI**: GPT-4o-mini for generation, Text-Embedding-3-Small for embeddings
- **ChromaDB**: Vector database for semantic search and document storage
- **Pandas**: Data manipulation and analysis
- **DeepEval**: LLM evaluation metrics and validation

### Environment
- **Python 3.10+**
- **Databricks Runtime** (for production deployment)
- **Azure OpenAI API** with authentication
- **UHG AIML Training Infrastructure**

---

## 💾 Data Description

### Dataset Structure
The project uses three main datasets:

#### 1. **capstone1_rag_dataset.csv** (Knowledge Base)
- **Records**: 100+ clinical documents
- **Columns**:
  - `document_id`: Unique identifier (numeric)
  - `document_url`: Source URL (NIH/GHR)
  - `context`: Full clinical text content with disease descriptions, genetic causes, inheritance patterns, etc.
- **Sources**: National Center for Biotechnology Information (NCBI), Genetic and Rare Diseases Information Center
- **Topics**: Genetic disorders, clinical manifestations, inheritance patterns, genetic mutations

**Example Documents**:
- Autosomal Dominant Epilepsy with Auditory Features (ADEAF)
- Nephronophthisis
- Wagner Syndrome
- And 97+ more clinical conditions

#### 2. **capstone1_rag_test_questions.csv**
- **Purpose**: Test dataset for RAG system evaluation
- **Columns**:
  - `question`: Clinical questions to be answered by the system
  - `retrieved_documents`: Retrieved evidence (populated by pipeline)
  - `generated_answer`: System-generated answer (populated by pipeline)

#### 3. **capstone1_rag_validation.csv**
- **Purpose**: Ground truth for model validation and evaluation
- **Columns**:
  - `question`: Clinical questions
  - `reference_context`: Human-curated relevant context
  - `reference_answer`: Expert-verified answer

---

## 🚀 Quick Start Guide

### Prerequisites
- Access to Azure OpenAI API (GPT-4o-mini and Text-Embedding-3-Small models)
- UHG AIML Training authentication credentials
- Python 3.10+ environment
- Databricks workspace (for deployment)

### Installation & Setup

#### Step 1: Configure Environment
```bash
# Create .env file with Azure credentials
cp usis.env .env

# Fill in the following fields in .env:
# MODEL_ENDPOINT=<your-azure-endpoint>
# API_VERSION=2024-08-01-preview
# EMBEDDINGS_MODEL_NAME=text-embedding-3-small_1
# CHAT_MODEL_NAME=gpt-4o-mini_2024-07-18
# PROJECT_ID=<your-project-id>
```

#### Step 2: Install Dependencies
Open `capstone_setup.ipynb` and run to install:
- langchain
- langchain-openai
- chromadb
- deepeval
- tenacity
- pandas

```python
# Run in Databricks notebook
%run ./capstone_setup
```

#### Step 3: Run the Pipeline
Open `code.ipynb` and execute cells sequentially:
1. **Setup Phase**: Load libraries and authentication
2. **Data Loading**: Load clinical documents from CSV
3. **Embedding & Storage**: Create vector embeddings and store in ChromaDB
4. **Retrieval**: Test semantic and BM25 retrieval strategies
5. **Generation**: Generate answers based on retrieved context
6. **Validation**: Evaluate using DeepEval metrics

---

## 🔍 Core Components & Pipeline

### 1. **Data Loading & Preprocessing**
```python
def load_csv_with_langchain(csv_path, source_column=None):
    """Load CSV documents into LangChain format"""
```
- Converts CSV clinical documents into LangChain Document objects
- Preserves metadata (source URL, document ID)
- Returns: List of Document objects ready for embedding

### 2. **Embedding & Vector Storage**
```python
def store_embeddings(persist_directory, docs=None):
    """Create and persist vector embeddings using ChromaDB"""
```
- Uses Azure OpenAI Text-Embedding-3-Small model
- Creates semantic vectors for each document chunk
- Stores in ChromaDB with persistent storage
- Reuses existing embeddings if available (MD5 hash validation)

**Key Features**:
- Efficient caching with file hash validation
- Automatic vector store creation and persistence
- Error handling with exponential backoff retry logic

### 3. **Retrieval Strategies**

#### 3a. Semantic Retrieval (Vector Similarity)
```python
def semantic_retrieval(query, vectorstore, top_k=3):
    """Fetch top-k most similar documents using vector similarity"""
```
- Uses cosine similarity for semantic matching
- Filters duplicates for result diversity
- Parameters: `top_k` (default 3) controls result count

#### 3b. Hybrid/BM25 Retrieval
```python
def bm25_rag_pipeline(query, vectorstore, top_k=3):
    """Combine semantic and keyword search for diverse results"""
```
- Combines semantic vectors with BM25 keyword matching
- Balances semantic understanding with exact keyword matching
- Produces diverse, high-quality retrieval results

### 4. **Answer Generation**
```python
def generate_answer(query, top_chunks, model_name=CHAT_DEPLOYMENT_NAME):
    """Generate answer strictly based on retrieved context"""
```
- Uses Azure OpenAI GPT-4o-mini model
- Enforces context-only responses (no model hallucination)
- Fallback message: "No relevant content found in the provided csv data."
- Temperature: 0 (deterministic responses)

### 5. **End-to-End Pipeline**
```python
def csv_chatbot_pipeline(csv_path, user_query, persist_directory):
    """Complete pipeline: Load → Embed → Retrieve → Generate → Return"""
```
**Flow**:
1. Check for cached embeddings
2. Load/reuse vector store
3. Retrieve relevant documents
4. Generate contextualized answer
5. Return context, question, and response

---

## 📊 Evaluation Metrics

The project uses **DeepEval** for comprehensive LLM evaluation:

### Retrieval Metrics
- **Contextual Precision**: Measures proportion of retrieved documents relevant to query
- **Contextual Recall**: Measures coverage of relevant information in retrieved documents
- **Contextual Relevancy**: Overall relevance of retrieved context to question

### Generation Metrics
- **Answer Relevancy**: How well answer addresses the question
- **Faithfulness**: Answer consistency with retrieved context (hallucination detection)
- **Hallucination Score**: Measures factual accuracy vs. generated content

### Custom Metrics
- **GEval**: Can define custom evaluation criteria

**Evaluation Example**:
```python
from deepeval.metrics import ContextualPrecisionMetric, HallucinationMetric
from deepeval import evaluate

metric = ContextualPrecisionMetric(threshold=0.6, model=wrapped_model)
results = evaluate(test_cases, [metric])
```

---

## 📝 Usage Examples

### Example 1: Single Query
```python
csv_path = "./Data/capstone1_rag_dataset.csv"
persist_directory = "./capstone_chroma.db/"

query = "What are the key features of autosomal dominant epilepsy with auditory features?"

result = csv_chatbot_pipeline(csv_path, query, persist_directory)

print("Question:", result['question'])
print("Retrieved Context:", result['context'][:2])  # First 2 documents
print("AI Response:", result['AI_generated_response'])
```

### Example 2: Batch Processing
```python
def batch_generate_answers_from_csv(csv_path, question_csv_path, persist_directory):
    """Generate answers for multiple questions from CSV"""
    questions = load_questions_from_csv(question_csv_path)
    results = []
    for q in questions:
        pipeline_result = csv_chatbot_pipeline(csv_path, q, persist_directory)
        results.append({
            'question': q,
            'AI_generated_response': pipeline_result['AI_generated_response']
        })
    return results

# Run batch processing
results = batch_generate_answers_from_csv(
    csv_path="./Data/capstone1_rag_dataset.csv",
    question_csv_path="./Data/capstone1_rag_validation.csv",
    persist_directory="./capstone_chroma.db/"
)
```

### Example 3: Model Validation
```python
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualPrecisionMetric
from deepeval import evaluate

# Create test cases
test_cases = [
    LLMTestCase(
        input="What are the main eye-related symptoms of Wagner syndrome?",
        actual_output=model_answer,
        expected_output=reference_answer,
        retrieval_context=retrieved_docs
    )
]

# Evaluate
metric = ContextualPrecisionMetric(threshold=0.6, model=wrapped_model)
results = evaluate(test_cases, [metric])

# Print results
for res in results.test_results:
    print(f"Success: {res.metrics_data[0].success}")
    print(f"Score: {res.metrics_data[0].score}")
    print(f"Reason: {res.metrics_data[0].reason}")
```

---

## 🔐 Security & Configuration

### Environment Variables
The system requires Azure OpenAI credentials configured in `usis.env`:

```env
MODEL_ENDPOINT="https://<your-resource>.openai.azure.com/"
API_VERSION="2024-08-01-preview"
EMBEDDINGS_MODEL_NAME="text-embedding-3-small_1"
CHAT_MODEL_NAME="gpt-4o-mini_2024-07-18"
PROJECT_ID="<your-project-id>"
```

### UHG Compliance
- **Telemetry**: ChromaDB telemetry disabled for compliance
- **Authentication**: OAuth2 token-based authentication with UHG API
- **Secrets**: Credentials stored in Databricks secrets scope `AIML_Training`

---

## 🧠 Key Insights & Best Practices

### Design Decisions

1. **Hybrid Retrieval**: BM25 + semantic search provides better results than pure semantic search
   - Semantic: Captures meaning and context
   - BM25: Captures exact keywords and entity names

2. **Context-Constrained Generation**: Prompt explicitly instructs model to use only provided context
   - Reduces hallucination risk
   - Improves factual accuracy
   - Provides fallback for unanswerable queries

3. **Persistent Vector Store**: ChromaDB caching with MD5 validation
   - Avoids redundant embedding computation
   - Speeds up pipeline iterations
   - Maintains consistency across runs

4. **Evaluation-Driven Approach**: Uses human-annotated validation set
   - Enables quantitative performance tracking
   - Identifies model weaknesses
   - Guides optimization efforts

### Optimization Tips

1. **Retrieval Tuning**:
   - Adjust `top_k` parameter (3-5 typically optimal)
   - Experiment with different retrieval strategies
   - Use reranking for further refinement

2. **Embedding Optimization**:
   - Consider document chunking for large documents
   - Experiment with different embedding models
   - Validate vector store quality

3. **Generation Tuning**:
   - Adjust prompt engineering for better answers
   - Use different temperature settings for exploratory vs. deterministic tasks
   - Consider model selection based on latency/quality tradeoff

---

## 📈 Performance Benchmarks

### System Characteristics
- **Retrieval Latency**: ~100-500ms per query (depends on vector store size)
- **Generation Latency**: ~1-3s per query (depends on response length)
- **Total E2E Latency**: ~2-5s per query
- **Vector Store Size**: ~50-100MB for 100 documents
- **Accuracy**: Varies by domain (validated against reference answers)

---

## 🐛 Troubleshooting

### Common Issues

1. **Authentication Error**
   ```
   Issue: "Invalid credentials for Azure OpenAI"
   Solution: Verify API_KEY, ENDPOINT, and API_VERSION in .env file
   ```

2. **ChromaDB Persistence Error**
   ```
   Issue: "Vector store directory not found"
   Solution: Ensure persist_directory is writable and has sufficient disk space
   ```

3. **Out of Memory on Large Datasets**
   ```
   Issue: "Memory exceeded during embedding"
   Solution: Process documents in batches, reduce batch size, or use smaller embedding model
   ```

4. **No Relevant Context Found**
   ```
   Issue: "Model returns fallback message"
   Solution: Review query quality, adjust retrieval top_k, consider query reformulation
   ```

---

## 📚 References & Further Reading

### Documentation
- [LangChain Official Documentation](https://python.langchain.com/)
- [Azure OpenAI API Reference](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [ChromaDB Vector Database](https://www.trychroma.com/)
- [DeepEval LLM Testing Framework](https://www.deepeval.com/)

### Clinical Data Sources
- [NIH Genetic and Rare Diseases Information Center (GARD)](https://rarediseases.info.nih.gov/)
- [National Center for Biotechnology Information (NCBI)](https://www.ncbi.nlm.nih.gov/)
- [National Library of Medicine (NLM)](https://www.nlm.nih.gov/)

### Research Papers
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)
- "Dense Passage Retrieval for Open-Domain Question Answering" (Karpukhin et al., 2020)

---

## 📋 Requirements & Dependencies

```
langchain>=0.1.0
langchain-openai>=0.0.1
langchain-community>=0.0.1
chromadb>=0.3.0
openai>=1.0.0
azure-identity>=1.0.0
deepeval>=0.20.0
tenacity>=8.0.0
pandas>=1.5.0
python-dotenv>=0.19.0
```

---

## 🎓 Learning Outcomes

By completing this project, you will:

✅ Understand RAG (Retrieval-Augmented Generation) architecture
✅ Learn vector embedding and semantic search concepts
✅ Implement production-ready LLM applications
✅ Apply LLM evaluation metrics and validation techniques
✅ Handle healthcare/clinical data responsibly
✅ Integrate multiple AI services (embeddings + LLM generation)
✅ Deploy scalable data pipelines on Databricks

---

## 👥 Contact & Support

For questions or issues:
1. Review the Capstone instruction card: `Capstone_1_Learner_Instruction_card.pdf`
2. Check the troubleshooting section above
3. Consult DeepEval documentation for evaluation issues
4. Review LangChain documentation for pipeline questions

---

## 📄 License & Usage

This is an educational capstone project developed as part of the AIML Training program. The clinical data is sourced from public NIH/NCBI databases and is used for educational purposes only.

---

## 🔄 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Dec 2025 | Initial implementation with semantic and BM25 retrieval |
| - | - | Comprehensive evaluation using DeepEval metrics |
| - | - | Production-ready Databricks deployment |

---

**Last Updated**: December 12, 2025  
**Project Status**: ✅ Complete and Validated

