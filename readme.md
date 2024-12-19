# A simple RAG System Implementation

![](images/overview.png)

A comprehensive Retrieval-Augmented Generation (RAG) system designed to enhance Large Language Model (LLM) responses by incorporating relevant context from a document database.

## Quick Start
Set up the environment and install the required packages, our code is tested on Python 3.11.

```bash
pip install -r requirements.txt
```

Run the following command to initialize the database and start the web interface:

```bash
python init_db.py --docs_dir raw_docs
python web.py
```

Put your documents in the `raw_docs` folder and access the web interface at `http://localhost:7860`.

We implement Qiwen-25 model in `model.py`, you can modify the model to your own model. Only requirement is to implement the `_call` in your own model class. System prompt is also customizable in `model.py`.

You can find more hyperparameters settings in the following introduction.

To evaluate with given file (default is `dataset/qa.xlsx`), run the following command:

```bash
python evaluate.py --data_path dataset/qa.xlsxdataset/qa.xlsx
```

## System Architecture

The system consists of four main Python modules:
- `init_db.py`: Document processing and database initialization
- `model.py`: LLM interface implementation
- `web.py`: Web interface using Gradio
- `evaluate.py`: System evaluation

### Document Processing and Database Initialization

The system supports multiple document formats and implements efficient document processing:

```python
def process_documents():
    # Initialize document loaders for different file types
    loaders = {
        ".pdf": PDFPlumberLoader,
        ".txt": TextLoader,
        ".md": TextLoader
    }
    
    # Document processing pipeline
    documents = []
    for file_path in glob.glob("raw_docs/**/*.*", recursive=True):
        ext = os.path.splitext(file_path)[1].lower()
        if ext in loaders:
            loader = loaders[ext](file_path)
            documents.extend(loader.load())

    # Text chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)

    # Vector database initialization
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./vectorDB/data"
    )
    vectordb.persist()
```

### Model Implementation

The system supports both a dummy LLM for testing and the Qwen-25 model for production:

```python
class Qwen25LLM(BaseLLM):
    def __init__(self):
        super().__init__()
        self.client = Client("http://127.0.0.1:8001/")
        
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.client.predict(
            prompt,
            api_name="/predict"
        )
        return response
```

## Setup and Configuration

### Environment Requirements

```bash
PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
MODEL_API_BASE="http://localhost:8001"
```

### Directory Structure

```
project_root/
├── init_db.py        # Database initialization
├── model.py          # LLM implementation
├── web.py           # Web interface
├── evaluate.py       # Evaluation system
├── raw_docs/         # Source documents
├── vectorDB/         # Vector database storage
│   └── data/
└── dataset/          # Evaluation datasets
```

## Evaluation System

The evaluation system includes:
- Custom dataset with 20 human-constructed questions
- Reference answers and passages
- Automated evaluation script
- Performance metrics tracking

### Evaluation Script

```python
if __name__ == "__main__":
    args = parser.parse_args()
    model_center = Model_center()

    def inference_row(row, debug=False, comparison_mode=False):
        context, ((q, a),) = model_center(row["Question"], [], debug,comparison_mode)
        result = {"Question": q}
        if comparison_mode:
            answer, naive_answer = a.split("**Naive result:**")
            answer = answer.strip("\n")[16:]
            result.update({"Answer": answer, "Passage": context,
                    "Naive Answer": naive_answer})
        else:
            result.update({"Answer": a, "Passage": context})
        result.update({'Reference Answer': row['Reference Answer'],
                'Reference Passage': row['Passage']})
        return pd.Series(result)

    df_eval = pd.read_excel(args.data_path)
    df_result = df_eval.apply(inference_row, axis=1, debug=args.debug,
            comparison_mode=args.comparison_mode)
    df_result.to_excel("eval_result.xlsx", index=False)
```

## Performance Optimization

### Vector Database Optimization
- Chunk size: 500 characters
- Overlap: 50 characters
- Similarity threshold: 0.3

### Response Generation Optimization
- Context limitation to most relevant documents
- Optimized prompt templates
- Response caching when applicable

## Authors

- Xinjie Shen 
- Haoquan Zhang
- Yaoshi Chen

## Claim

This is a course project. Codes are released under GPL-3.0 License.