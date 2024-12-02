from ra_fs import *
from langchain.llms import Ollama
from langchain import PromptTemplate

# Loading orca-mini from Ollama
llm = Ollama(model="phi3", temperature=0)

# Loading the Embedding Model
embed = load_embedding_model(model_path="all-MiniLM-L6-v2")

# loading and splitting the documents
docs = load_pdf_data(file_path="C:/Users/SK/Downloads/financial_report/RFL_2022.pdf")
documents = split_docs(documents=docs)

# creating vectorstore
vectorstore = create_embeddings(documents, embed)

# converting vectorstore to a retriever
retriever = vectorstore.as_retriever()

# Creating the prompt from the template which we created before
prompt = PromptTemplate.from_template(template)

# Creating the chain
chain = load_qa_chain(retriever, llm, prompt)
get_response("generate summary Profit and loss statement of Reliance Financial Limited, 2022", chain)
