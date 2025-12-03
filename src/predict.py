import torch
import os
import re
import gc
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# --- CONFIGURATION ---
HF_REPO_ID = "Delstarford/uploader"

class DualBot:
    def __init__(self):
        # Use CPU to save memory
        self.device = "cpu"
        print(f"⚙️  AI Running on: {self.device}")
        
        # We DO NOT load models here anymore to save startup time and memory
        self.models = {}
        self.tokenizers = {}
        self.current_mode = None

    def _load_specific_model(self, mode):
        """
        Smart Loader: Unloads old models to free up RAM before loading the new one.
        This ensures we stay under the 512MB Render limit.
        """
        # If we already have this model loaded, do nothing
        if self.current_mode == mode and mode in self.models:
            return

        print(f"🔄 Switching brain to: {mode.upper()}...")

        # 1. CLEAR MEMORY (Unload previous model)
        if self.current_mode is not None:
            print(f"🗑️ Unloading {self.current_mode}...")
            if self.current_mode in self.models:
                del self.models[self.current_mode]
            if self.current_mode in self.tokenizers:
                del self.tokenizers[self.current_mode]
            
            # Force Python to release memory
            self.models.clear()
            self.tokenizers.clear()
            gc.collect()

        # 2. LOAD NEW MODEL
        try:
            print(f"☁️  Downloading {mode} from Hugging Face...")
            
            # Decide folder name based on mode
            # 'friend' and 'therapy' use the fallback base model
            if mode in ['roast', 'relationship']:
                folder = f"{mode}_model"
                self.models[mode] = GPT2LMHeadModel.from_pretrained(HF_REPO_ID, subfolder=folder).to(self.device)
                self.tokenizers[mode] = GPT2Tokenizer.from_pretrained(HF_REPO_ID, subfolder=folder)
            else:
                # Fallback for friend/therapy (Base DistilGPT2)
                self.models[mode] = GPT2LMHeadModel.from_pretrained('distilgpt2').to(self.device)
                self.tokenizers[mode] = GPT2Tokenizer.from_pretrained('distilgpt2')

            self.tokenizers[mode].pad_token = self.tokenizers[mode].eos_token
            self.current_mode = mode
            print(f"✅ {mode.upper()} Loaded Successfully!")

        except Exception as e:
            print(f"⚠️ Load failed: {e}. Using backup.")
            # Emergency Backup
            self.models[mode] = GPT2LMHeadModel.from_pretrained('distilgpt2').to(self.device)
            self.tokenizers[mode] = GPT2Tokenizer.from_pretrained('distilgpt2')
            self.current_mode = mode

    def generate(self, text, mode="roast", user_data=None):
        if not user_data: user_data = {"name": "User", "gender": "male", "age": 18}
        name = user_data.get('name', 'User')
        
        # 1. Load ONLY the model we need right now
        target_mode = mode if mode in ['roast', 'relationship'] else 'friend'
        self._load_specific_model(target_mode)
        
        tokenizer = self.tokenizers[target_mode]
        model = self.models[target_mode]
        
        # 2. Prompt Logic
        gender = user_data.get('gender', 'male').lower()
        if mode == "relationship":
            role = "Girlfriend" if gender == 'male' else "Boyfriend"
            tone = "flirty and sweet"
            input_text = f"Instruction: Act as {name}'s {tone} {role}.\n{name}: {text}\n{role}:"
        elif mode == "roast":
            input_text = f"Input: {text}\nRoast:"
        else:
            input_text = f"Context: Best friends chatting.\n{name}: {text}\nBestie:"

        # 3. Generate
        inputs = tokenizer(input_text, return_tensors='pt', padding=True).to(self.device)
        output = model.generate(
            inputs.input_ids, 
            attention_mask=inputs.attention_mask, 
            max_length=150, 
            do_sample=True, 
            temperature=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

        response = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # 4. Cleanup
        response = response.replace(input_text, "").strip()
        response = response.split(f"{name}:")[0]
        response = re.sub(r'[_\*]{2,}', '', response)
        
        return response.strip()