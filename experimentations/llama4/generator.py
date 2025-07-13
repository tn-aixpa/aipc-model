import transformers
from dotenv import load_dotenv, find_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

_ = load_dotenv(find_dotenv())

# 1. Set your Hugging Face token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.get_env("HUGGINGFACEHUB_API_TOKEN")

# 2. Use LLama 4 from Hugging Face
def load_llama4():
    access_token = os.environ["HUGGINGFACEHUB_API_TOKEN"]
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-4-Scout-17B-16E-Instruct", token=access_token)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-4-Scout-17B-16E-Instruct", token=access_token)    
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
    return HuggingFacePipeline(pipeline=text_generation_pipeline)


# 3 Prompt optimiszation technique
def prompt_opt():
    pass