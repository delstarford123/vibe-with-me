import os
import re
import gc

# Note: We do NOT import torch/transformers here. 
# We import them inside the class to save memory during startup.

# --- CONFIGURATION ---
HF_REPO_ID = "Delstarford/uploader"

class DualBot:
    def __init__(self):
        # 1. LAZY IMPORT: Only load heavy libraries now
        print("⚙️  Initializing AI Libraries...")
        global torch, GPT2LMHeadModel, GPT2Tokenizer
        import torch
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        
        # Limit threads to prevent CPU spikes
        torch.set_num_threads(1)
        
        self.device = "cpu"
        print(f"⚙️  AI Running on: {self.device}")
        
        self.models = {}
        self.tokenizers = {}
        self.current_mode = None

    def _load_specific_model(self, mode):
        if self.current_mode == mode and mode in self.models:
            return

        print(f"🔄 Switching brain to: {mode.upper()}...")

        # Memory Cleanup
        self.models.clear()
        self.tokenizers.clear()
        self.current_mode = None
        gc.collect()

        try:
            print(f"☁️  Downloading {mode} from Hugging Face ({HF_REPO_ID})...")
            
            if mode in ['roast', 'relationship']:
                folder = f"{mode}_model"
                tokenizer = GPT2Tokenizer.from_pretrained(HF_REPO_ID, subfolder=folder)
                model = GPT2LMHeadModel.from_pretrained(
                    HF_REPO_ID, 
                    subfolder=folder, 
                    low_cpu_mem_usage=True
                ).to(self.device)
            else:
                print("Using generic backup model...")
                tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
                model = GPT2LMHeadModel.from_pretrained('distilgpt2', low_cpu_mem_usage=True).to(self.device)

            self.models[mode] = model
            self.tokenizers[mode] = tokenizer
            self.tokenizers[mode].pad_token = self.tokenizers[mode].eos_token
            self.current_mode = mode
            print(f"✅ {mode.upper()} Loaded Successfully!")

        except Exception as e:
            print(f"⚠️ MODEL LOAD FAILED: {e}")
            self.current_mode = None

    def generate(self, text, mode="roast", user_data=None):
        if not user_data: user_data = {"name": "User", "gender": "male", "age": 18}
        name = user_data.get('name', 'User')
        
        target_mode = mode if mode in ['roast', 'relationship'] else 'friend'
        
        try:
            self._load_specific_model(target_mode)
        except Exception as e:
            print(f"❌ CRITICAL ERROR: {e}")
            return "My brain is rebooting. Try 'Smart Mode'!"

        if target_mode not in self.models:
            return "I'm dizzy (Memory Full). Please use '✨ Smart' Mode!"

        tokenizer = self.tokenizers[target_mode]
        model = self.models[target_mode]
        
        # Prompt Logic
        gender = user_data.get('gender', 'male').lower()
        if mode == "relationship":
            role = "Girlfriend" if gender == 'male' else "Boyfriend"
            tone = "flirty and sweet"
            input_text = f"Instruction: Act as {name}'s {tone} {role}.\n{name}: {text}\n{role}:"
        elif mode == "roast":
            input_text = f"Input: {text}\nRoast:"
        else:
            input_text = f"Context: Best friends chatting.\n{name}: {text}\nBestie:"

        try:
            inputs = tokenizer(input_text, return_tensors='pt', padding=True).to(self.device)
            output = model.generate(
                inputs.input_ids, 
                attention_mask=inputs.attention_mask, 
                max_length=100, 
                do_sample=True, 
                temperature=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

            response = tokenizer.decode(output[0], skip_special_tokens=True)
            response = response.replace(input_text, "").strip()
            response = response.split(f"{name}:")[0]
            response = re.sub(r'[_\*]{2,}', '', response)
            return response.strip()
            
        except Exception as e:
            print(f"❌ Generation Error: {e}")
            return "I lost my train of thought."