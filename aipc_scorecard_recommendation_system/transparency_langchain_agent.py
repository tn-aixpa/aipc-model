import os
from typing import List, Dict, Optional
from pydantic import BaseModel
import numpy as np
from langchain_community.document_loaders import DirectoryLoader, JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Configuration
MODEL_CARDS_DIR = "model_cards"
TRANSPARENCY_ASPECTS = [
    "model architecture details",
    "training data description",
    "data preprocessing",
    "evaluation methodology",
    "limitations",
    "intended use",
    "ethical considerations",
    "bias analysis",
    "environmental impact",
    "model parameters",
    "hyperparameters",
    "computational requirements",
    "license information",
    "contact information"
]
os.environ["OPENAI_API_KEY"] = ""  # For OpenAI/LangChain

# Pydantic models for type safety
class ModelCard(BaseModel):
    id: str
    content: str
    metadata: Dict[str, str]
    file_path: str

class TransparencyScore(BaseModel):
    model_card_id: str
    overall_score: float
    aspect_scores: Dict[str, float]
    explanation: str

# Initialize components
embedding_model = OpenAIEmbeddings()  # Can replace with other embeddings
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

# Custom JSON loader for model cards
def load_model_cards(directory: str) -> List[ModelCard]:
    """Load model cards from JSON files using LangChain's JSONLoader."""
    loader = DirectoryLoader(
        directory,
        glob="**/*.json",
        loader_cls=JSONLoader,
        loader_kwargs={"jq_schema": ".", "text_content": False}
    )
    docs = loader.load()
    
    model_cards = []
    for doc in docs:
        content = doc.page_content
        if isinstance(content, dict):
            content = json.dumps(content)
        model_cards.append(ModelCard(
            id=doc.metadata.get("source", "").split("/")[-1],
            content=content,
            metadata=doc.metadata,
            file_path=doc.metadata["source"]
        ))
    return model_cards

# Text splitting
def split_documents(model_cards: List[ModelCard]) -> List[Dict]:
    """Split model card content into chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    split_cards = []
    for card in model_cards:
        chunks = text_splitter.split_text(card.content)
        for i, chunk in enumerate(chunks):
            split_cards.append({
                "id": f"{card.id}_chunk_{i}",
                "content": chunk,
                "metadata": {**card.metadata, "chunk": i},
                "source_card": card.id
            })
    return split_cards

# Vector store creation
def create_vector_store(model_cards: List[Dict], embeddings: Embeddings):
    """Create a FAISS vector store from model card chunks."""
    contents = [card["content"] for card in model_cards]
    metadatas = [card["metadata"] for card in model_cards]
    ids = [card["id"] for card in model_cards]
    
    return FAISS.from_texts(
        texts=contents,
        embedding=embeddings,
        metadatas=metadatas,
        ids=ids
    )

# Transparency evaluation prompt
transparency_eval_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an AI model transparency evaluator. Analyze the provided model card content and evaluate its transparency based on the following aspects:
     
     Transparency Aspects:
     {aspects}
     
     For each aspect, provide a score from 0-1 where:
     0 = Not mentioned at all
     0.3 = Briefly mentioned
     0.6 = Adequately described
     0.8 = Well described with details
     1 = Thoroughly explained with comprehensive information
     
     Provide your evaluation in JSON format with the aspect scores and a brief explanation for each."""),
    ("human", "Model Card Content:\n{content}")
])

# Ranking prompt
ranking_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a model card ranking system. Given the transparency evaluations for multiple model cards, rank them from most to least transparent.
     
     For each model card, you have the following information:
     - Overall transparency score
     - Scores for each transparency aspect
     - Explanations for the scores
     
     Provide your final ranking with justification for each position."""),
    ("human", "Transparency Evaluations:\n{evaluations}")
])

# Tools for the agent
@tool
def evaluate_model_card_transparency(model_card_content: str) -> Dict:
    """Evaluate the transparency of a model card based on standard transparency aspects."""
    chain = transparency_eval_prompt | llm | StrOutputParser()
    aspects_str = "\n".join(f"- {aspect}" for aspect in TRANSPARENCY_ASPECTS)
    result = chain.invoke({
        "aspects": aspects_str,
        "content": model_card_content
    })
    return {"evaluation": result}

@tool
def retrieve_similar_model_cards(query: str, k: int = 3) -> List[Dict]:
    """Retrieve similar model cards based on a query."""
    if not hasattr(retrieve_similar_model_cards, "vector_store"):
        model_cards = load_model_cards(MODEL_CARDS_DIR)
        split_cards = split_documents(model_cards)
        retrieve_similar_model_cards.vector_store = create_vector_store(split_cards, embedding_model)
    
    docs = retrieve_similar_model_cards.vector_store.similarity_search(query, k=k)
    return [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]

@tool
def rank_model_cards_by_transparency(evaluations: List[Dict]) -> List[Dict]:
    """Rank model cards based on their transparency evaluations."""
    chain = ranking_prompt | llm | StrOutputParser()
    evaluations_str = "\n\n".join(
        f"Model Card {i}:\n{json.dumps(eval, indent=2)}" 
        for i, eval in enumerate(evaluations, 1)
    )
    result = chain.invoke({"evaluations": evaluations_str})
    return {"ranking": result}

# Agent setup
tools = [evaluate_model_card_transparency, retrieve_similar_model_cards, rank_model_cards_by_transparency]

agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Model Transparency Ranking Agent. Your job is to help users evaluate and rank model cards based on their transparency.
     
     You have access to the following tools:
     - evaluate_model_card_transparency: Evaluate a model card's transparency
     - retrieve_similar_model_cards: Find similar model cards
     - rank_model_cards_by_transparency: Rank model cards by transparency
     
     Always be thorough in your analysis and provide clear explanations for your rankings."""),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad")
])

agent = create_tool_calling_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Main workflow
def main():
    # Example usage of the agent
    chat_history = []
    
    # Load some model cards to evaluate
    model_cards = load_model_cards(MODEL_CARDS_DIR)
    
    # Start conversation with the agent
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        response = agent_executor.invoke({
            "input": user_input,
            "chat_history": chat_history
        })
        
        print(f"\nAgent: {response['output']}")
        chat_history.extend([
            HumanMessage(content=user_input),
            AIMessage(content=response['output'])
        ])

if __name__ == "__main__":
    main()