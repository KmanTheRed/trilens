#!/usr/bin/env python
import os
import torch
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from sacrebleu.metrics import BLEU
from tqdm.auto import tqdm

def main():
    # Check CUDA status and set device.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load Mafand dataset for fine-tuning.
    # The Mafand dataset does not have a "train" split;
    # we concatenate "validation" and "test" splits as our training data.
    print("Loading Mafand training dataset...")
    mafand_valid = load_dataset("masakhane/mafand", "en-amh", split="validation")
    mafand_test = load_dataset("masakhane/mafand", "en-amh", split="test")
    mafand_dataset = concatenate_datasets([mafand_valid, mafand_test])
    print(f"Loaded Mafand dataset with {len(mafand_dataset)} examples.")

    # 2. Load IrokoBench dev split for evaluation.
    # Use the correct dataset ID and configuration for afrimmlu.
    print("Loading afrimmlu dev dataset...")
    afrimmlu_dev = load_dataset("masakhane/irokobench", "afrimmlu", split="dev")

    # 3. Remove duplicates from Mafand based on IrokoBench dev Amharic texts.
    afrimmlu_amh_texts = {entry["amh_text"] for entry in afrimmlu_dev}
    mafand_filtered = [entry for entry in mafand_dataset if entry["amh_text"] not in afrimmlu_amh_texts]
    print(f"Filtered Mafand training dataset size: {len(mafand_filtered)} examples.")

    # Create a Dataset from the filtered examples.
    train_dataset = Dataset.from_list(mafand_filtered)

    # 4. Load pre-trained model and tokenizer; move model to CUDA.
    model_name = "facebook/mbart-large-50"
    print(f"Loading model and tokenizer: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)

    # Define a preprocessing function to tokenize input (Amharic) and target (English) texts.
    def preprocess_function(example):
        # Format input to resemble the prompt "Translate [input] into English."
        # Adjust max_length and padding as needed.
        inputs = tokenizer(example["amh_text"], max_length=512, truncation=True, padding="max_length")
        targets = tokenizer(example["en_text"], max_length=512, truncation=True, padding="max_length")
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": targets["input_ids"],
        }
    
    print("Tokenizing training dataset...")
    tokenized_dataset = train_dataset.map(preprocess_function, batched=False)

    # 5. Set up training arguments for fine-tuning.
    training_args = TrainingArguments(
        output_dir="./finetuned_translation_model",
        num_train_epochs=3,                       # Adjust the number of epochs as needed.
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        evaluation_strategy="no",
        save_total_limit=2,
        fp16=True if device == "cuda" else False
    )

    # Initialize the Trainer.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer
    )

    # 6. Fine-tune the model.
    print("Starting fine-tuning on GPU...")
    trainer.train()
    print("Fine-tuning complete.")

    # 7. Evaluate using the first 10 examples from the afrimmlu dev split.
    print("Evaluating the finetuned model on the dev set...")
    test_examples = afrimmlu_dev.select(range(10))
    test_sentences = test_examples["amh_text"]
    ground_truths = test_examples["en_text"]

    translations = []
    for sentence in tqdm(test_sentences, desc="Translating examples", leave=True):
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: val.to(device) for key, val in inputs.items()}  # Move tensor inputs to CUDA.
        # Use beam search (num_beams set to 5) to improve translation quality.
        output_ids = model.generate(
            **inputs,
            max_length=512,
            num_beams=5,
            early_stopping=True
        )
        translated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        translations.append(translated_text)

    # 8. Print the translation pairs.
    print("\nTranslation pairs:")
    for src, pred in zip(test_sentences, translations):
        print("Amharic: ", src)
        print("English: ", pred)
        print("-" * 50)

    # 9. Compute the overall BLEU score using sacreBLEU.
    bleu = BLEU()
    bleu_score = bleu.corpus_score(translations, [ground_truths])
    print(f"\nAverage BLEU Score on the test set: {bleu_score.score:.2f}")

if __name__ == "__main__":
    main()