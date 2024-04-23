import os
import mlrun
import torch
from minio import Minio
from huggingface_hub import login
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

class ChatBot(mlrun.serving.V2ModelServer):
    def load(self):
        """load and initialize the model and/or other elements"""
        model_name_or_path = self.get_param("model_name")
        adapter_path = self.get_param("adapter_path")
        print(adapter_path)
        print(model_name_or_path)
        model, tokenizer = self.load_model(model_name_or_path, adapter_path)
        self.model = model
        self.tokenizer = tokenizer

    def predict(self, body: dict):
        """Generate model predictions from sample."""
        print(body)
        response = []
        for el in body["inputs"]:
            row = el["row"]
            skip_special_tokens = el["skip_special_tokens"]
            max_new_tokens = el["max_new_tokens"]
            do_sample = el["do_sample"]
            response.append(self.generate(row, skip_special_tokens, max_new_tokens, do_sample))            
        return response
    
    
    def load_model(self, model_name_or_path, adapter_path):
        """
        Load the model and tokenizer
        """        
        login('')

        secrets = {"MINIO_URL": "", "MINIO_AK": "", "MINIO_SK": ""}
        client = Minio(
            "minio-api.digitalhub-dev.smartcommunitylab.it",
            access_key = secrets["MINIO_AK"],
            secret_key = secrets["MINIO_SK"],
        )

        for item in client.list_objects("adapter",recursive=True):
            client.fget_object("adapter",item.object_name,item.object_name)

        model_id = "checkpoint-400"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
        )
        return model, tokenizer
    
    def generate(self, row, skip_special_tokens=True, max_new_tokens=250, do_sample=False):
        """
        Generate text using the model
        """  
        print("inside generate function")
        inputs = self.tokenizer(row, return_tensors="pt").to(0)
        outputs = self.model.generate(inputs.input_ids, max_new_tokens, do_sample)
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens)
        return generated

    


