# train.py
import os, argparse, math, random
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Optional but nice: ROUGE for simple eval
try:
    import evaluate
    rouge = evaluate.load("rouge")
except Exception:
    rouge = None


def parse_args():
    p = argparse.ArgumentParser(description="Minimal HF fine-tuning for resume-ready project")
    p.add_argument("--model_name", type=str, default="sshleifer/distilbart-cnn-12-6",
                   help="Small, fast summarization model")
    p.add_argument("--train_size", type=int, default=1000, help="Subset size for training")
    p.add_argument("--eval_size", type=int, default=200, help="Subset size for eval")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--train_bs", type=int, default=2)
    p.add_argument("--eval_bs", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_input_len", type=int, default=512)
    p.add_argument("--max_target_len", type=int, default=128)
    p.add_argument("--output_dir", type=str, default="results")
    p.add_argument("--logs_dir", type=str, default="logs")
    p.add_argument("--artifacts_dir", type=str, default="artifacts/model")
    p.add_argument("--samples_out", type=str, default="artifacts/samples.txt")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # --- 1) Data ---
    print("[INFO] Loading SAMSum dataset (dialogue summarization)…")
    ds_all = load_dataset("samsum")

    # small, quick subset for a minimal real project
    train_ds = ds_all["train"].select(range(min(args.train_size, len(ds_all["train"]))))
    eval_ds  = ds_all["test"].select(range(min(args.eval_size, len(ds_all["test"]))))

    # --- 2) Model / Tokenizer ---
    print(f"[INFO] Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    model.to(device)

    # --- 3) Preprocess / Tokenize ---
    def preprocess(batch):
        inputs = batch["dialogue"]
        targets = batch["summary"]

        model_inputs = tokenizer(
            inputs,
            max_length=args.max_input_len,
            truncation=True,
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=args.max_target_len,
                truncation=True,
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    cols_to_remove = list(set(train_ds.column_names) - {"dialogue", "summary"})
    tokenized_train = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    tokenized_eval  = eval_ds.map(preprocess,  batched=True, remove_columns=eval_ds.column_names)

    # --- 4) Training Setup ---
    os.makedirs(args.logs_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(Path(args.samples_out).parent, exist_ok=True)

    fp16 = torch.cuda.is_available()
    args_hf = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="no",
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_dir=args.logs_dir,
        logging_steps=20,
        report_to=["tensorboard"],
        fp16=fp16,
        dataloader_pin_memory=torch.cuda.is_available(),
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    def compute_metrics(eval_pred):
        if rouge is None:
            return {}
        preds, labels = eval_pred
        # Replace -100 with pad token id for decoding
        labels = [[(l if l != -100 else tokenizer.pad_token_id) for l in label] for label in labels]
        pred_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)
        label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
        scores = rouge.compute(predictions=pred_texts, references=label_texts, use_stemmer=True)
        # Round for cleaner logs
        return {k: round(v, 4) for k, v in scores.items()}

    trainer = Trainer(
        model=model,
        args=args_hf,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # --- 5) Train / Evaluate ---
    print("[INFO] Training…")
    trainer.train()
    print("[INFO] Evaluating…")
    metrics = trainer.evaluate()
    print("[METRICS]", metrics)

    # --- 6) Save model + tokenizer for reproducibility ---
    Path(args.artifacts_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.artifacts_dir)
    tokenizer.save_pretrained(args.artifacts_dir)
    print(f"[INFO] Saved model + tokenizer to: {args.artifacts_dir}")

    # --- 7) Generate a few sample summaries ---
    print("[INFO] Generating sample summaries…")
    test_raw = ds_all["test"].select(range(5))
    with open(args.samples_out, "w", encoding="utf-8") as f:
        for i in range(len(test_raw)):
            dialogue = test_raw[i]["dialogue"]
            inputs = tokenizer(dialogue, return_tensors="pt", truncation=True, max_length=args.max_input_len).to(device)
            with torch.no_grad():
                out_ids = model.generate(**inputs, max_length=args.max_target_len, num_beams=4)
            summary = tokenizer.decode(out_ids[0], skip_special_tokens=True)
            f.write("===" + str(i) + "===\n")
            f.write("DIALOGUE:\n" + dialogue + "\n\n")
            f.write("SUMMARY:\n" + summary + "\n\n")
    print(f"[INFO] Wrote sample outputs to: {args.samples_out}")

    print("\n[DONE] Quick project complete.")
    print("Open TensorBoard (optional): tensorboard --logdir logs")
    print(f"Model dir: {args.artifacts_dir}")
    print(f"Samples:   {args.samples_out}")


if __name__ == "__main__":
    main()
