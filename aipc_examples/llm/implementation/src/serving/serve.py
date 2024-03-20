import mlrun
import torch
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
        model, tokenizer = self.load_model(model_name_or_path, adapter_path)
        self.model = model
        self.tokenizer = tokenizer

    def predict(self, body: dict):
        """Generate model predictions from sample."""
        row = body["row"]
        skip_special_tokens = body["skip_special_tokens"]
        max_length = body["max_length"]
        adapter_papth = body["adapter_path"]
        return self.generate(row, skip_special_tokens, max_length, adapter_papth)
    
    
    def load_model(self, model_name_or_path, adapter_path):
        """
        Load the model and tokenizer
        """
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        # Fixing some of the early LLaMA HF conversion issues.
        tokenizer.bos_token_id = 1

        # Load the model (use bf16 for faster inference)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            #load_in_4bit=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
            )
        )
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
        return model, tokenizer
    
    def generate(self, row, skip_special_tokens=True, max_length=64):
        """
        Generate text using the model
        """
        input_ids = self.tokenizer(row, return_tensors='pt').input_ids.to(0)        
        generations = self.model.generate(
            input_ids=input_ids, 
            max_new_tokens=max_length,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            top_p=0.90
        )
        gen = self.tokenizer.batch_decode(generations.detach().cpu().numpy()[:, input_ids.shape[1]:], skip_special_tokens=skip_special_tokens)[0]
        return gen

    


