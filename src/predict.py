import torch
import os
import re
import gc
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# --- CONFIGURATION ---
# ✅ CORRECT REPO ID (Where your models actually are)
HF_REPO_ID = "Delstarford/uploader"

# Limit threads to prevent CPU spikes crashing the free tier
torch.set_num_threads(1)

class DualBot:
    def __init__(self):
        self.device = "cpu"
        print(f"⚙️  AI Initialized on: {self.device}")
        
        self.models = {}
        self.tokenizers = {}
        self.current_mode = None

    def _load_specific_model(self, mode):
        # If already loaded, do nothing (Save time)
        if self.current_mode == mode and mode in self.models:
            return

        print(f"🔄 Request to switch brain to: {mode.upper()}...")

        # 1. AGGRESSIVE MEMORY CLEANUP
        # We MUST clear the old model before loading the new one
        self.models.clear()
        self.tokenizers.clear()
        self.current_mode = None
        gc.collect() # Force Python to release RAM immediately

        try:
            print(f"☁️  Downloading {mode} from Hugging Face ({HF_REPO_ID})...")
            
            # Determine folder
            if mode in ['roast', 'relationship']:
                folder = f"{mode}_model"
                
                # Load Tokenizer first (It's small)
                tokenizer = GPT2Tokenizer.from_pretrained(HF_REPO_ID, subfolder=folder)
                
                # Load Model with low_cpu_mem_usage (Critical for Render)
                model = GPT2LMHeadModel.from_pretrained(
                    HF_REPO_ID, 
                    subfolder=folder, 
                    low_cpu_mem_usage=True
                ).to(self.device)
            else:
                # Fallback base model
                print("Using generic backup model...")
                tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
                model = GPT2LMHeadModel.from_pretrained('distilgpt2', low_cpu_mem_usage=True).to(self.device)

            # Assign to class
            self.models[mode] = model
            self.tokenizers[mode] = tokenizer
            self.tokenizers[mode].pad_token = self.tokenizers[mode].eos_token
            self.current_mode = mode
            print(f"✅ {mode.upper()} Loaded Successfully!")

        except Exception as e:
            print(f"⚠️ MODEL LOAD FAILED: {e}")
            # IMPORTANT: If loading fails, do NOT crash. Just leave current_mode as None.
            # The generate function will handle this safely.
            self.current_mode = None

    def generate(self, text, mode="roast", user_data=None):
        if not user_data: user_data = {"name": "User", "gender": "male", "age": 18}
        name = user_data.get('name', 'User')
        
        # 1. Try to load the model
        target_mode = mode if mode in ['roast', 'relationship'] else 'friend'
        
        try:
            self._load_specific_model(target_mode)
        except Exception as e:
            print(f"❌ CRITICAL ERROR LOADING MODEL: {e}")
            return "My brain is rebooting. Please use 'Smart Mode' (Gemini) for now!"

        # 2. Check if load succeeded
        if target_mode not in self.models:
            return "I'm dizzy (Memory Full or Model Missing). Please use '✨ Smart' Mode!"

        tokenizer = self.tokenizers[target_mode]
        model = self.models[target_mode]
        
        # 3. Prompt Logic
        gender = user_data.get('gender', 'male').lower()
        if mode == "relationship":
            role = "Girlfriend" if gender == 'male' else "Boyfriend"
            tone = "flirty and sweet"
            input_text = f"Instruction: Act as {name}'s {tone} {role}.\n{name}: {text}\n{role}:"
        elif mode == "roast":
            input_text = f"Input: {text}\nRoast:"
        else:
            input_text = f"Context: Best friends chatting.\n{name}: {text}\nBestie:"

        # 4. Generate
        try:
            inputs = tokenizer(input_text, return_tensors='pt', padding=True).to(self.device)
            
            # Limit tokens to prevent long processing times
            output = model.generate(
                inputs.input_ids, 
                attention_mask=inputs.attention_mask, 
                max_length=100,  # Shorter length = Faster & Less Crashy
                do_sample=True, 
                temperature=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

            response = tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Cleanup
            response = response.replace(input_text, "").strip()
            response = response.split(f"{name}:")[0]
            response = re.sub(r'[_\*]{2,}', '', response)
            
            return response.strip()
            
        except Exception as e:
            print(f"❌ Generation Error: {e}")
            return "I lost my train of thought."