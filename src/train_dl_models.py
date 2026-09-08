"""
Deep Learning Fine-Tuning Script for Transformer Models
======================================================
Optimized for NVIDIA GeForce RTX 3050 (4GB VRAM) & Google Colab GPU.

Supported Tasks:
  --task fakenews    (Binary classification: Real News vs Fake News)
  --task hatespeech  (Binary classification: Safe vs Hate Speech/Hostile)

Supported Architectures:
  --model_name microsoft/deberta-v3-small (Recommended for 4GB VRAM)
  --model_name roberta-base
  --model_name bert-base-uncased
  --model_name ai4bharat/indic-bert

Memory Optimization Techniques for 4GB VRAM:
  - Mixed Precision Training (FP16 via torch.cuda.amp)
  - Per-device batch size = 8 with Gradient Accumulation = 2 (effective batch size 16)
  - Max Sequence Length capped at 256 tokens
  - Evaluation after every epoch with best model checkpoint saved
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def get_device_info():
    """Detects available hardware and prints GPU diagnostics."""
    print("=" * 65)
    print(" HARDWARE & RUNTIME DIAGNOSTICS")
    print("=" * 65)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f" [✓] NVIDIA CUDA GPU Detected: {gpu_name}")
        print(f" [✓] Available VRAM: {total_vram_gb:.2f} GB")
        print(f" [✓] CUDA Version: {torch.version.cuda}")
        print(f" [✓] PyTorch Version: {torch.__version__}")
        if total_vram_gb <= 4.5:
            print(" [!] Detected 4GB VRAM environment: Enabling FP16 + Gradient Accumulation.")
        device = "cuda"
        fp16_enabled = True
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print(" [✓] Apple Silicon (MPS) Detected.")
        device = "mps"
        fp16_enabled = False
    else:
        print(" [!] No GPU detected. Running on CPU (training will be slow).")
        device = "cpu"
        fp16_enabled = False
    print("=" * 65)
    return device, fp16_enabled

def load_data(task: str):
    """Loads stratified train, val, and test splits."""
    splits_dir = os.path.join(BASE_DIR, 'data', 'splits')

    if task == 'fakenews':
        train_path = os.path.join(splits_dir, 'train.csv')
        val_path = os.path.join(splits_dir, 'val.csv')
        test_path = os.path.join(splits_dir, 'test.csv')
    else:
        train_path = os.path.join(splits_dir, 'hate_train.csv')
        val_path = os.path.join(splits_dir, 'hate_val.csv')
        test_path = os.path.join(splits_dir, 'hate_test.csv')

    print(f"\n[Loading Data] Task: {task.upper()}")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # Standardize column mapping
    for df in [df_train, df_val, df_test]:
        # If headline exists, prepend to article_text for full contextual signal
        if 'headline' in df.columns:
            df['text'] = df['headline'].fillna('') + " " + df['article_text'].fillna('')
        else:
            df['text'] = df['article_text'].fillna('')
        df['text'] = df['text'].astype(str).str.strip()
        df['label'] = df['label'].astype(int)

    print(f"  Train samples: {len(df_train):,} | Real/Safe: {(df_train['label']==0).sum():,}, Fake/Hostile: {(df_train['label']==1).sum():,}")
    print(f"  Val samples:   {len(df_val):,} | Real/Safe: {(df_val['label']==0).sum():,}, Fake/Hostile: {(df_val['label']==1).sum():,}")
    print(f"  Test samples:  {len(df_test):,} | Real/Safe: {(df_test['label']==0).sum():,}, Fake/Hostile: {(df_test['label']==1).sum():,}")

    ds_train = Dataset.from_pandas(df_train[['text', 'label']])
    ds_val = Dataset.from_pandas(df_val[['text', 'label']])
    ds_test = Dataset.from_pandas(df_test[['text', 'label']])

    return DatasetDict({'train': ds_train, 'val': ds_val, 'test': ds_test})

def compute_metrics(eval_pred):
    """Computes comprehensive metrics: Accuracy, Precision, Recall, F1."""
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, average='macro')

    return {
        'accuracy': round(float(acc), 4),
        'f1': round(float(f1), 4),
        'precision': round(float(precision), 4),
        'recall': round(float(recall), 4),
        'f1_macro': round(float(f1_macro), 4),
        'precision_macro': round(float(precision_macro), 4),
        'recall_macro': round(float(recall_macro), 4)
    }

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Transformers on Fake News & Hate Speech")
    parser.add_argument('--task', type=str, choices=['fakenews', 'hatespeech'], default='fakenews',
                        help="Classification task: 'fakenews' or 'hatespeech'")
    parser.add_argument('--model_name', type=str, default='microsoft/deberta-v3-small',
                        help="Hugging Face model repository ID")
    parser.add_argument('--epochs', type=int, default=3, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=8, help="Per-device batch size (8 for 4GB VRAM)")
    parser.add_argument('--grad_accum', type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument('--lr', type=float, default=2e-5, help="Peak learning rate")
    parser.add_argument('--max_length', type=int, default=256, help="Maximum sequence length in tokens")
    parser.add_argument('--output_dir', type=str, default=None, help="Directory to save final model checkpoints")

    args = parser.parse_args()

    # Hardware Check
    device, fp16 = get_device_info()

    # Determine Output Directory
    clean_model_tag = args.model_name.replace('/', '_')
    if args.output_dir is None:
        output_dir = os.path.join(BASE_DIR, 'models', 'deep_learning', args.task, clean_model_tag)
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load Data
    raw_datasets = load_data(args.task)

    # Load Tokenizer
    print(f"\n[Tokenizer] Loading tokenizer for '{args.model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize_fn(batch):
        return tokenizer(batch['text'], truncation=True, max_length=args.max_length)

    print("[Tokenizer] Tokenizing datasets...")
    tokenized_datasets = raw_datasets.map(tokenize_fn, batched=True, remove_columns=['text'])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Load Pretrained Model
    print(f"\n[Model] Initializing {args.model_name} for sequence classification (num_labels=2)...")
    id2label = {0: "Real News" if args.task == 'fakenews' else "Safe",
                1: "Fake News" if args.task == 'fakenews' else "Hate Speech / Hostile"}
    label2id = {v: k for k, v in id2label.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label=id2label,
        label2id=label2id
    )

    # Training Arguments (Tuned for 4GB VRAM)
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_steps=0.1,
        fp16=fp16,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["val"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\n" + "=" * 65)
    print(f" STARTING FINE-TUNING ({args.epochs} Epochs)")
    print(f" Effective Batch Size: {args.batch_size * args.grad_accum}")
    print(f" Mixed Precision FP16: {fp16}")
    print("=" * 65)

    trainer.train()

    # Final Evaluation on Unseen Test Split
    print("\n" + "=" * 65)
    print(" RUNNING FINAL EVALUATION ON UNSEEN TEST SET")
    print("=" * 65)
    test_results = trainer.evaluate(tokenized_datasets["test"])
    print("\nTest Set Metrics:")
    for k, v in test_results.items():
        if not k.startswith("eval_runtime") and not k.startswith("eval_samples"):
            print(f"  {k}: {v}")

    # Generate Confusion Matrix on Test Split
    predictions_output = trainer.predict(tokenized_datasets["test"])
    preds = np.argmax(predictions_output.predictions, axis=1)
    true_labels = tokenized_datasets["test"]["label"]
    cm = confusion_matrix(true_labels, preds).tolist()

    # Save Model & Tokenizer
    final_model_dir = os.path.join(output_dir, "best_model")
    print(f"\n[Saving] Saving best model and tokenizer to: {final_model_dir}")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    # Save Benchmark Metrics to JSON
    benchmark_payload = {
        "task": args.task,
        "model_architecture": args.model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "max_length": args.max_length,
        "test_metrics": test_results,
        "confusion_matrix": cm,
        "labels": id2label
    }

    metrics_file = os.path.join(output_dir, "dl_benchmark_results.json")
    with open(metrics_file, "w") as f:
        json.dump(benchmark_payload, f, indent=2)
    print(f"[Saving] Metrics benchmark saved to: {metrics_file}")

    # Generate and Save Confusion Matrix Plot
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=[id2label[0], id2label[1]],
                    yticklabels=[id2label[0], id2label[1]])
        plt.title(f"{args.model_name} - Confusion Matrix ({args.task.upper()})")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        cm_plot_path = os.path.join(output_dir, "test_confusion_matrix.png")
        plt.savefig(cm_plot_path, dpi=300)
        plt.close()
        print(f"[Saving] Confusion matrix plot saved to: {cm_plot_path}")
    except Exception as e:
        print(f"[Warning] Could not generate confusion matrix plot: {e}")

    print("\n[Done] Training and evaluation complete!\n")

if __name__ == "__main__":
    main()
