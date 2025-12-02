import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from preprocess import load_and_clean_data

# Automatically determine paths so you don't have to type them
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "../models")

def train_model(mode):
    print(f"\n==========================================")
    print(f"   STARTING TRAINING: {mode.upper()} MODE")
    print(f"==========================================")
    
    # 1. Load Data (Using your smart preprocess.py)
    try:
        texts, tokenizer = load_and_clean_data(mode)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    if not texts:
        print(f"⚠️ No data found for {mode}. Skipping training.")
        return

    print(f"📚 Loaded {len(texts)} examples. Preparing to train...")

    # Save texts to a temporary file for the TextDataset loader
    # (TextDataset requires a file path, not a list of strings)
    temp_file = os.path.join(BASE_DIR, f"temp_{mode}.txt")
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write("\n".join(texts))

    # 2. Prepare Dataset
    # Block size is the max length of a sequence. 128 is good for speed/memory.
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=temp_file,
        block_size=128
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # 3. Initialize Model
    # We use DistilGPT2 because it is small and fast for laptops
    model = GPT2LMHeadModel.from_pretrained('distilgpt2')

    # 4. Training Arguments
    output_dir = os.path.join(MODEL_DIR, f"{mode}_model")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,              # 3 loops over the data
        per_device_train_batch_size=4,   # Keep low (4 or 8) to save RAM
        save_steps=1000,                 # Save model every 1000 steps
        warmup_steps=100,
        logging_steps=50,
        prediction_loss_only=True,
    )

    # 5. Train
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    print(f"🏃 Training started... (This might take a while)")
    trainer.train()
    
    # 6. Save the final model
    print(f"💾 Saving model to: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Cleanup temp file
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    print(f"✅ {mode.upper()} Model successfully saved!")

if __name__ == "__main__":
    # Create models folder if it doesn't exist
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # Train Roast Model (Sarcasm/Humor)
    train_model("roast")
    
    # Train Therapy Model (Mental Health) - Optional if you have the data
    train_model("therapy")