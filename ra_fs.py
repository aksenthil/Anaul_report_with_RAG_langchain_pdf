### Documentation for the Code

This Python code utilizes `LangChain`, a framework for building applications with LLMs (large language models) and other AI tools. It demonstrates how to load a PDF, split it into chunks, embed the text using Hugging Face embeddings, create vector representations with FAISS, and then use these embeddings for question answering. Below is an explanation of the code components:

---

### **Imports**

```python
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import textwrap
```

- `HuggingFaceEmbeddings`: A class from LangChain that allows embedding documents using a Hugging Face model.
- `PyMuPDFLoader`: Used to load PDF documents in the form of text. It reads PDF files and converts them to a format suitable for processing.
- `RecursiveCharacterTextSplitter`: A text splitting utility that divides documents into smaller chunks. This is especially useful for handling long documents.
- `FAISS`: A library for efficient similarity search, used here to store and search document embeddings.
- `RetrievalQA`: A class that combines retrieval of relevant documents (from vector store) and answers generation using LLMs.
- `textwrap`: A Python standard library module for formatting text to fit within a specified width, used here to prettify the output.

---

### **Function: `load_pdf_data`**

```python
def load_pdf_data(file_path):
    # Creating a PyMuPDFLoader object with file_path
    loader = PyMuPDFLoader(file_path=file_path)

    # loading the PDF file
    docs = loader.load()

    # returning the loaded document
    return docs
```

- **Purpose**: This function loads a PDF document from the specified `file_path`.
- **Process**:
  - It creates an instance of `PyMuPDFLoader` using the provided file path.
  - The `load()` method of the loader reads the content of the PDF file and returns it as a list of documents.
- **Returns**: A list of documents containing the text extracted from the PDF.

---

### **Function: `split_docs`**

```python
def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    # Initializing the RecursiveCharacterTextSplitter with
    # chunk_size and chunk_overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Splitting the documents into chunks
    chunks = text_splitter.split_documents(documents=documents)

    # returning the document chunks
    return chunks
```

- **Purpose**: This function splits a list of documents into smaller chunks for easier processing.
- **Parameters**:
  - `documents`: A list of documents to be split.
  - `chunk_size`: The maximum length of each chunk (default is 1000 characters).
  - `chunk_overlap`: The overlap between consecutive chunks to maintain context (default is 20 characters).
- **Process**:
  - A `RecursiveCharacterTextSplitter` object is initialized with the specified chunk size and overlap.
  - The `split_documents` method splits the documents into chunks based on these parameters.
- **Returns**: A list of document chunks.

---

### **Function: `load_embedding_model`**

```python
def load_embedding_model(model_path, normalize_embedding=True):
    return HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={'device': 'cpu'},  # here we will run the model with CPU only
        encode_kwargs={
            'normalize_embeddings': normalize_embedding  # keep True to compute cosine similarity
        }
    )
```

- **Purpose**: This function loads an embedding model from Hugging Face's model hub.
- **Parameters**:
  - `model_path`: The path or name of the pre-trained model to be used for embeddings.
  - `normalize_embedding`: Whether to normalize the embeddings (default is `True`).
- **Process**:
  - Initializes a `HuggingFaceEmbeddings` object with the model path and configuration.
  - The model will be run on the CPU, and the embeddings will be normalized for cosine similarity calculations.
- **Returns**: A `HuggingFaceEmbeddings` object that can be used to generate embeddings for documents.

---

### **Function: `create_embeddings`**

```python
def create_embeddings(chunks, embedding_model, storing_path="vectorstore"):
    # Creating the embeddings using FAISS
    vectorstore = FAISS.from_documents(chunks, embedding_model)

    # Saving the model in current directory
    vectorstore.save_local(storing_path)

    # returning the vectorstore
    return vectorstore
```

- **Purpose**: This function generates document embeddings using a pre-trained embedding model and stores them in a FAISS vector store.
- **Parameters**:
  - `chunks`: A list of document chunks to be embedded.
  - `embedding_model`: The embedding model (created in the previous function).
  - `storing_path`: The path where the FAISS vector store will be saved (default is `vectorstore`).
- **Process**:
  - The `FAISS.from_documents` method is used to create a vector store by embedding the document chunks.
  - The `save_local()` method saves the vector store locally.
- **Returns**: The FAISS vector store that contains the embedded document chunks.

---

### **Template Strings**

```python
prompt = """
### System:
You are an AI Assistant that follows instructions extremely well. Help as much as you can.

### User:
{prompt}

### Response:

"""

template = """
### System:
You are a respectful and honest assistant. You have to answer the user's questions using only the context provided to you. If you don't know the answer, just say you don't know. Don't try to make up an answer.

### Context:
{context}

### User:
{question}

### Response:
"""
```

- **Purpose**: These are template strings used to structure the interaction between the AI assistant and the user.
- **System**: Describes the assistant's behavior in the conversation. It provides instructions on how to respond.
- **User**: Represents the user’s query or prompt.
- **Context**: The relevant context or documents provided to the assistant for answering the query.

---

### **Function: `load_qa_chain`**

```python
def load_qa_chain(retriever, llm, prompt):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever, # here we are using the vectorstore as a retriever
        chain_type="stuff",
        return_source_documents=True, # including source documents in output
        chain_type_kwargs={'prompt': prompt} # customizing the prompt
    )
```

- **Purpose**: This function initializes a question-answering chain using the retriever (vector store) and the language model (LLM).
- **Parameters**:
  - `retriever`: The retriever, in this case, the FAISS vector store.
  - `llm`: The language model to be used for generating the answer.
  - `prompt`: The prompt template to guide the AI’s responses.
- **Process**:
  - The function creates a `RetrievalQA` chain using the specified retriever and LLM.
  - The `chain_type="stuff"` means the retriever retrieves relevant documents, which are then used as input to the LLM for generating the answer.
- **Returns**: A `RetrievalQA` chain object that can be used to answer questions.

---

### **Function: `get_response`**

```python
def get_response(query, chain):
    # Getting response from chain
    response = chain({'query': query})

    # Wrapping the text for better output in Jupyter Notebook
    wrapped_text = textwrap.fill(response['result'], width=100)
    print(wrapped_text)
```

- **Purpose**: This function takes a user query and gets a response from the `RetrievalQA` chain.
- **Parameters**:
  - `query`: The user’s question.
  - `chain`: The `RetrievalQA` chain object created earlier.
- **Process**:
  - The function queries the chain with the user's question and retrieves the result.
  - It then uses `textwrap.fill()` to wrap the response text for better readability in Jupyter Notebook.
- **Returns**: Prints the wrapped response to the console.

---

### **Summary**

- This code is designed to create an AI-powered document processing and question-answering system.
- It extracts text from PDF files, splits it into chunks, embeds the text using a Hugging Face model, stores embeddings in FAISS for efficient similarity search, and uses the embeddings for answering user queries.
- The system uses the LangChain library for efficient handling of the embedding and retrieval process, ensuring that the AI assistant can provide accurate responses based on the documents provided.

