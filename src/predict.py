import torch
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# --- PATH SETUP ---
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_SCRIPT_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

class DualBot:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"⚙️  AI Running on: {self.device}")
        
        self.models = {}
        self.tokenizers = {}
        
        # Load models
        self._load_model('roast')
        self._load_model('therapy')
        # We use the default DistilGPT2 for the 'friend' mode initially
        self._load_fallback('friend') 

    def _load_model(self, mode):
        model_path = os.path.join(MODELS_DIR, f"{mode}_model")
        print(f"🔍 Checking for {mode} model...")

        if os.path.exists(model_path):
            try:
                self.models[mode] = GPT2LMHeadModel.from_pretrained(model_path).to(self.device)
                self.tokenizers[mode] = GPT2Tokenizer.from_pretrained(model_path)
                self.tokenizers[mode].pad_token = self.tokenizers[mode].eos_token
                print(f"✅ Loaded {mode.upper()} model!")
            except Exception as e:
                print(f"❌ Error loading {mode}: {e}")
                self._load_fallback(mode)
        else:
            print(f"⚠️  {mode} model not trained yet. Using backup.")
            self._load_fallback(mode)

    def _load_fallback(self, mode):
        if mode not in self.models:
            print(f"🔄 Loading base brain for {mode}...")
            self.models[mode] = GPT2LMHeadModel.from_pretrained('distilgpt2').to(self.device)
            self.tokenizers[mode] = GPT2Tokenizer.from_pretrained('distilgpt2')
            self.tokenizers[mode].pad_token = self.tokenizers[mode].eos_token

    def generate(self, text, mode="roast"):
        # Select the personality
        if mode not in self.models:
            mode = 'friend' # Default to friend if mode is unknown
            
        tokenizer = self.tokenizers[mode]
        model = self.models[mode]
        
        # --- PROMPT ENGINEERING (The Vibe Check) ---
        if mode == "roast":
            input_text = f"Input: {text}\nRoast:"
        elif mode == "therapy":
            input_text = f"User: {text}\nTherapist:"
        else: # FRIEND / VIBE MODE
            # We give it a context to be a storyteller/best friend
            input_text = f"Context: Two friends vibing and telling stories.\nMe: {text}\nBestie:"

        inputs = tokenizer(input_text, return_tensors='pt', padding=True).to(self.device)

        # Generate output
        output = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=150,           # Allow longer responses for stories
            num_return_sequences=1, 
            no_repeat_ngram_size=2,
            do_sample=True,
            temperature=0.9,          # High temperature = More creative/random
            top_p=0.92,
            pad_token_id=tokenizer.eos_token_id
        )

        response = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Clean up the response
        response = response.replace(input_text, "").strip()
        
        # Remove any extra "Me:" or "Bestie:" hallucinations
        response = response.split("Me:")[0]
        response = response.split("Bestie:")[0]
        
        return response