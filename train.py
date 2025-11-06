# train.py
import os, argparse
from pathlib import Path
import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

try:
    import evaluate

    rouge = evaluate.load("rouge")
except Exception:
    rouge = None


def parse_args():
    p = argparse.ArgumentParser(
        description="Minimal HF fine-tuning for resume-ready project"
    )
    p.add_argument("--model_name", type=str, default="sshleifer/distilbart-cnn-12-6")
    p.add_argument("--train_size", type=int, default=1000)
    p.add_argument("--eval_size", type=int, default=200)
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


def load_any_summarization_dataset():
    """
    Try multiple public datasets; if the Hub is blocked, fall back to a local toy dataset.
    Returns (dataset_dict, source_name) with columns 'dialogue' and 'summary'.
    """
    try:
        print("[INFO] Loading SAMSum…")
        ds = load_dataset("samsum")
        return ds, "samsum"
    except Exception as e1:
        print(
            f"[WARN] Could not load 'samsum' ({type(e1).__name__}). Trying dialogsum…"
        )

    try:
        print("[INFO] Loading knkarthick/dialogsum…")
        ds = load_dataset("knkarthick/dialogsum")
        return ds, "knkarthick/dialogsum"
    except Exception as e2:
        print(
            f"[WARN] Could not load 'knkarthick/dialogsum' ({type(e2).__name__}). Trying cnn_dailymail…"
        )

    try:
        print("[INFO] Loading cnn_dailymail 3.0.0…")
        ds = load_dataset("cnn_dailymail", "3.0.0")

        def rename(batch):
            return {"dialogue": batch["article"], "summary": batch["highlights"]}

        ds = DatasetDict(
            {
                "train": ds["train"].map(
                    rename, remove_columns=ds["train"].column_names
                ),
                "validation": ds["validation"].map(
                    rename, remove_columns=ds["validation"].column_names
                ),
                "test": ds["test"].map(rename, remove_columns=ds["test"].column_names),
            }
        )
        return ds, "cnn_dailymail"
    except Exception as e3:
        print(
            f"[WARN] Could not load 'cnn_dailymail' ({type(e3).__name__}). Falling back to local toy dataset…"
        )

    data = {
        "dialogue": [
            "A: Did you send the report?\nB: Not yet, I will by 5pm.\nA: Ok, thanks.",
            "A: Morning! Server logs show errors.\nB: I'll restart the API.\nA: Ping me when done.",
            "A: Coffee chat at 2?\nB: Make it 2:30, I have a standup.\nA: Works.",
            "A: Can you review PR #42?\nB: After lunch.\nA: Cool.",
        ],
        "summary": [
            "Teammate commits to send the report by 5pm.",
            "Team discusses API errors; restart planned with follow-up.",
            "Meeting moved to 2:30 due to standup.",
            "Code review scheduled for after lunch.",
        ],
    }
    ds = DatasetDict(
        {
            "train": Dataset.from_dict(data),
            "test": Dataset.from_dict(data),
        }
    )
    return ds, "local-toy"


def main():
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    ds_all, source = load_any_summarization_dataset()
    print(f"[INFO] Dataset source: {source}")

    train_split = "train"
    eval_split = "validation" if "validation" in ds_all else "test"

    train_ds = ds_all[train_split]
    eval_ds = ds_all[eval_split]

    train_ds = train_ds.select(range(min(args.train_size, len(train_ds))))
    eval_ds = eval_ds.select(range(min(args.eval_size, len(eval_ds))))

    print(f"[INFO] Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name).to(device)

    def preprocess(batch):
        model_inputs = tokenizer(
            batch["dialogue"], max_length=args.max_input_len, truncation=True
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["summary"], max_length=args.max_target_len, truncation=True
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_train = train_ds.map(
        preprocess, batched=True, remove_columns=train_ds.column_names
    )
    tokenized_eval = eval_ds.map(
        preprocess, batched=True, remove_columns=eval_ds.column_names
    )

    os.makedirs(args.logs_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    Path(args.samples_out).parent.mkdir(parents=True, exist_ok=True)

    use_fp16 = torch.cuda.is_available()
    args_hf = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_dir=args.logs_dir,
        logging_steps=20,
        report_to=["tensorboard"],
        fp16=use_fp16,
        dataloader_pin_memory=torch.cuda.is_available(),
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    def compute_metrics(eval_pred):
        if rouge is None:
            return {}
        preds, labels = eval_pred
        labels = [
            [(l if l != -100 else tokenizer.pad_token_id) for l in label]
            for label in labels
        ]
        pred_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)
        label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
        scores = rouge.compute(
            predictions=pred_texts, references=label_texts, use_stemmer=True
        )
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

    print("[INFO] Training…")
    trainer.train()
    print("[INFO] Evaluating…")
    metrics = trainer.evaluate()
    print("[METRICS]", metrics)

    Path(args.artifacts_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.artifacts_dir)
    tokenizer.save_pretrained(args.artifacts_dir)
    print(f"[INFO] Saved model + tokenizer to: {args.artifacts_dir}")

    print("[INFO] Generating sample summaries…")
    sample_split = "test" if "test" in ds_all else eval_split
    test_raw = ds_all[sample_split].select(range(min(5, len(ds_all[sample_split]))))
    with open(args.samples_out, "w", encoding="utf-8") as f:
        for i in range(len(test_raw)):
            dialogue = test_raw[i]["dialogue"]
            inputs = tokenizer(
                dialogue,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_input_len,
            ).to(device)
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs, max_length=args.max_target_len, num_beams=4
                )
            summary = tokenizer.decode(out_ids[0], skip_special_tokens=True)
            f.write(f"==={i}===\nDIALOGUE:\n{dialogue}\n\nSUMMARY:\n{summary}\n\n")
    print(f"[INFO] Wrote sample outputs to: {args.samples_out}")

    print("\n[DONE] Quick project complete.")
    print("Optional: tensorboard --logdir logs")
    print(f"Model dir: {args.artifacts_dir}")
    print(f"Samples:   {args.samples_out}")


if __name__ == "__main__":
    main()
