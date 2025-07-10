import torch
import transformers
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate


class RagWorkflow():
    def _innit_(self, docs_path, embedding_model="hkunlp/instructor-large", llm_model="meta-llama/Llama-2-7b-chat-hf", auth_token=None):        
        self.llm_model = llm_model
        self.auth_token = auth_token
        self.embedding_model = embedding_model    
        self.vectorstore = None     
        self.docs_path = None
        self.qa_chain = None
        
    def load_documents(self, chunk_size=500, chunk_overlap=100):
        # Connect to external documents
        loader = DirectoryLoader(self.docs_path, glob="**/*.md", loader_cls=TextLoader)
        docs = loader.load()
        # Text Splitting
        text_splitter = RecursiveCharacterTextSplitter(chunk_size, chunk_overlap, separators=['\n', '.'])
        splits = text_splitter.split_documents(loader.load())
        return splits

    def embedding_model(self, splits):
        # Downloading embedding model
        embedding_model = HuggingFaceInstructEmbeddings(
            model_name = self.embedding_model,
            embed_instruction = "Represent the model cards for retrieval: ",
            query_instruction = 'Represent the user question for retrieving supporting documents: ',
            model_kwargs = {'device': 'cuda'}
        )      
        # Create the VectorStore
        self.vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)

    def retriever(self):
        retriever = self.vectorstore.as_retriever()
        model = AutoModelForCausalLM.from_pretrained(self.llm_model,
                                                     load_in_4bit=True,
                                                     device_map='auto',
                                                     use_auth_token=self.auth_token)
        tokenizer = AutoTokenizer.from_pretrained(self.llm_model, use_auth_token=self.auth_token)
        text_generation_pipeline = transformers.pipeline(
            model=model,
            tokenizer=tokenizer,
            task='text-generation',
            do_sample=True,
            temperature=0.2,
            repetition_penalty=1.1,
            return_full_text=True,
            max_new_tokens=1000
        )
        llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
        # Prompt Template
        qa_template = """### Instruction: You are a helpful assistant.
        Use the following context to answer the question below.
        If you don't know the answer or the context does not help you answer the question, please say "I don't know".
        
        {context}
        
        {question}
        
        ### Answer: """
        
        # Create a prompt instance
        QA_prompt = PromptTemplate.from_template(qa_template)
        
        # Instantiate the QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=retriever,
            chain_type_kwargs={"prompt": QA_prompt},
            return_source_documents=True
        )

    def query(self, question):
        response = self.qa_chain({'query': question})
        # Return the result
        return response['result']
        
        
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


def main():
    pass

if __name__ == "__main__":
    main()
        