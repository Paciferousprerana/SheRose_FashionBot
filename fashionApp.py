# pip install PyPDF2

import os
import chromadb
import PyPDF2
import streamlit as st
import time
import random
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("Could not import sentence_transformers: Please install sentence-transformers package.")
    
try:
    import chromadb
    from chromadb.api.types import EmbeddingFunction
except ImportError:
    raise ImportError("Could not import chromdb: Please install chromadb package.")




class MiniLML6V2EmbeddingFunction(EmbeddingFunction):
    MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    def __call__(self, texts):
        return MiniLML6V2EmbeddingFunction.MODEL.encode(texts).tolist()
emb_func = MiniLML6V2EmbeddingFunction()

class TextSplitter:           
    def create_chunks(self, text: str, chunk_size: int, chunk_overlap: int, separator: str = '\n'):
        """
        Splits the text into equally distributed chunks with specific word overlap.
        Args:
            text (str): Text to be converted into chunks.
            chunk_length (int): Maximum number of words in each chunk.
            chunk_overlap (int): Number of words to overlap between chunks.
            Additional parameters can be passed using kwargs.
        """
        text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size = chunk_size,
                    chunk_overlap  = chunk_overlap,
                    length_function=len, 
                    add_start_index=True)
        chunks = text_splitter.split_documents(text)
        return chunks

class buildChromaDb:

    def __init__(self, chromaClient, model=SentenceTransformer('all-MiniLM-L6-v2')):
    # Load the pre-trained model for generating embeddings
        self.model = model

    # Initialize ChromaDB client
        self.chroma_client = chromaClient

    def get_or_create_collection(self, collection_name, embedding_dim):
        # Check if the collection exists
        collections = self.chroma_client.list_collections()
        if any(col.name == collection_name for col in collections):
            self.chroma_client.delete_collection(collection_name)
        # Create a new collection with the correct embedding dimension
        return self.chroma_client.get_or_create_collection(collection_name, metadata={"embedding_dim": embedding_dim})

    def store_chunks_in_chromadb(self,chunk_lst, collection_name):
        # Get or create the collection with embedding dimension of 768
        collection = self.get_or_create_collection(collection_name, 768)
        
        ids_lst = []
        embeddings_lst = []
        
        for chunk in chunk_lst:

            # Generate embeddings for the summary and questions
            chunk_embedding = self.model.encode(chunk)
                
            # Concatenate embeddings
            final_embedding = chunk_embedding.tolist() 
            # + questions_embedding.tolist()
            
            # Create a unique ID for each chunk
            chunk_id = f"{str(chunk[10:40])+str(chunk[-40:])}"
            
            # Append data to respective lists
            # print(combined_embedding)
            ids_lst.append(chunk_id)
            embeddings_lst.append(final_embedding)

        # Store in ChromaDB
        collection.add(ids=ids_lst, embeddings=embeddings_lst, documents=chunk_lst)


class retrieval:
    def __init__(self, chromaClient, model=SentenceTransformer('all-MiniLM-L6-v2')):
        self.model = model
        self.chroma_client = chromaClient

    # @instrument
    def retrieved_chunks(self, query, collection_name, top_k=5):
        # Generate embedding for the query
        query_embedding = self.model.encode(query)

        # Get the collection
        collection = self.chroma_client.get_collection(collection_name)

        # Retrieve all documents from the collection
        all_documents = collection.get()

        # Compute similarity scores between query and document summaries
        similarities = []
        for document in all_documents['documents']:
            document_embedding = np.array(self.model.encode(document))
            similarity_score = cosine_similarity([query_embedding], [document_embedding])[0][0]
            similarities.append((document, similarity_score))

        # Sort by similarity score (descending) and select top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        relevant_chunks_with_scores = similarities[:top_k]

        return relevant_chunks_with_scores

    # @instrument
    def join_chunks(self, chunks):
        rel_chunks = []
        for chunk in chunks:
            rel_chunks.append(chunk[0])
        context = ' '.join(rel_chunks)

        return context

    # @instrument
    def queryLLM(self,query : str, collection_name : str, model):
        relevant_chunks = self.retrieved_chunks(query, collection_name)
        chunks = self.join_chunks(relevant_chunks)

        prompt=f"""[INST]
        <<SYS>>
        You are a fashion expert and you only answer questions related to fashion. Read the context and answer the question. If it cannot be answered, only say: 'Unanswerable'. Answer should be concise and professional. Make sure response is not cut off, and do not give an empty response. Don't mention that you are answering from the text or context provided, impersonate as if this is coming from your knowledge.
        Guidelines for Answering:
        1. Understand the Context
        2. Base answers solely on the information within the given context; do not rely on external knowledge.
        3. Don't mention that you are answering from the text or context provided, impersonate as if this is coming from your knowledge.
        4. Craft responses in full sentences to enhance clarity.
        5. Be Concise and Relevant. Avoid unnecessary elaboration.
        6. Provide answers without personal opinions or interpretations.
        7. Keep your response format consistent, adapting it to fit the nature of the question
        8. Rely solely on the provided context. Do not introduce external information.
        9. Don't mention that you are answering from the text or context provided, impersonate as if this is coming from your knowledge.
        10. You must not use phrases like- Based on the context provided
        11. Do not answer anything which is not in given in the context.
        12. Answer should be framed from the provided context only.
        <</SYS>>
        Context: {chunks}
        Question: {query}
        Answer 
        [/INST]
        """

        response = model.generate_text(prompt)

        return response



def get_wml_creds():
    load_dotenv(override=True)
    api_key = os.getenv("API_KEY", None)
    ibm_cloud_url = os.getenv("IBM_CLOUD_URL", None)
    project_id = os.getenv("PROJECT_ID", None)
    if api_key is None or ibm_cloud_url is None or project_id is None:
        print("Ensure you copied the .env file that you created earlier into the same directory as this notebook")
    else:
        creds = {
            "url": ibm_cloud_url,
            "apikey": api_key 
        }
    return project_id, creds


project_id, creds = get_wml_creds()

def merge_pdfs(pdf_list, output_path):
    pdf_merger = PyPDF2.PdfMerger()

    for pdf in pdf_list:
        pdf_merger.append(pdf)

    with open(output_path, 'wb') as output_pdf:
        pdf_merger.write(output_pdf)


def pdfToText(pdfPath : str):
    loader = PyPDFLoader(pdfPath)
    pages = loader.load()
    
    return pages

def flashyStatements(model):

        prompt=f"""[INST]
        <<SYS>>
        You are a fashion expert and you generate catchy statements that can attract youngsters to align with the trending fashion. Generate single line catchy phrases to attract youngsters to stay up to date with the fashion.
        Do not include anything mentioned below in your answer.
        Guidelines for Answering:
        1. Understand the Context
        2. Don't mention that you are answering from the text or context provided, impersonate as if this is coming from your knowledge.
        3. Craft responses in full sentences to enhance clarity.
        4. Be Concise and Relevant. Avoid unnecessary elaboration.
        5. Keep your response format consistent, adapting it to fit the nature of the question
        6. Don't mention that you are answering from the text or context provided, impersonate as if this is coming from your knowledge.
        7. You must not use phrases like- Based on the context provided
        <</SYS>>
        Exmple:
        "Bold colors and unique textures are trending!",
        "Sustainable fashion is taking over the industry.",
        "Winter is coming! Buy new Jackets.",
        "The rise of streetwear in high fashion."
        Answer 
        [/INST]
        """

        response = response = model.generate_text(prompt)

        return response

pdf_files = ['fashion_trending_news.pdf', 'Fashion_Analysis_report.pdf', 'socialMedia_trending_report.pdf']  # Replace with your PDF file paths
output_pdf_path = 'final_fashion_text.pdf'
merge_pdfs(pdf_files, output_pdf_path)
print(f"Merged PDF saved as {output_pdf_path}")
doc = pdfToText('final_fashion_text.pdf')
textSplitter = TextSplitter()
chunks = textSplitter.create_chunks(doc, 500, 20)
chunk_list = []
for chunk in chunks:
    chunk_list.append(chunk.page_content)

model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.Client()
collection_name = "fashion_chunks"
bcd = buildChromaDb(chroma_client, model)
bcd.store_chunks_in_chromadb(chunk_list, collection_name)

params = {
        GenParams.DECODING_METHOD: "greedy",
        GenParams.MIN_NEW_TOKENS: 1,
        GenParams.MAX_NEW_TOKENS: 500,
        GenParams.TEMPERATURE: 0,
    }
llm_model = Model(model_id='meta-llama/llama-2-70b-chat', params=params, credentials=creds, project_id=project_id)


statement = flashyStatements(model=llm_model)
statements_lst = statement.split('\n')
trendy_news = statements_lst

# streamlit appplication
def buffer_screen():
    st.title("Welcome to Fashion Trends Chatbot")
    st.write("Loading the latest trends...")
    with st.spinner('Please wait...'):
        # Display a single random trendy news statement
        st.write(random.choice(trendy_news))
        time.sleep(10)



def chat_screen():
    st.title("Fashion Trends Chatbot")
    query = st.text_input("Ask me about the latest trends in fashion:")
    if query:
        ret = retrieval(chroma_client)
        response = ret.queryLLM(query=query, collection_name=collection_name, model=llm_model)
        st.write(response)

# Main application logic
if 'buffer_done' not in st.session_state:
    st.session_state.buffer_done = False

if not st.session_state.buffer_done:
    buffer_screen()
    st.session_state.buffer_done = True
    st.experimental_rerun()
else:
    chat_screen()



