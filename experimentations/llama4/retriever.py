import os

from huggingface_hub import list_models
from dotenv import load_dotenv, find_dotenv

from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.schema import Document


_ = load_dotenv(find_dotenv())

# 1. Set your Hugging Face token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.get_env("HUGGINGFACEHUB_API_TOKEN")

# 2. Retrieve Hugging Face model cards (limited to top N)
def fetch_model_cards(n=5):
    model_infos = list_models(filter="text-classification", sort="downloads", limit=n)
    urls = [f"https://huggingface.co/{model.modelId}" for model in model_infos]
    return urls

# 3. Load model cards from the web
def load_and_split_documents(urls):
    loader = WebBaseLoader(urls)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

# 4. Create a retriever using FAISS
def create_retriever(documents):
    #embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    vectordb = FAISS.from_documents(documents, embeddings)
    return vectordb.as_retriever()

# 5 Rerank selected documents 
def reranker(self):
    from sentence_transformers import CrossEncoder
    # 1. Load a pretrained CrossEncoder model
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
    # The texts for which to predict similarity scores
    query = "How many people live in Berlin?"
    passages = [
        "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
        "Berlin has a yearly total of about 135 million day visitors, making it one of the most-visited cities in the European Union.",
        "In 2013 around 600,000 Berliners were registered in one of the more than 2,300 sport and fitness clubs.",
    ]

    # 2a. predict scores pairs of texts
    scores = model.predict([(query, passage) for passage in passages])
    print(scores)
    # 2b. Rank a list of passages for a query
    ranks = model.rank(query, passages, return_documents=True)

    print("Query:", query)
    for rank in ranks:
        print(f"- #{rank['corpus_id']} ({rank['score']:.2f}): {rank['text']}")
        
        
        
if __name__ == "__main__":
    pass