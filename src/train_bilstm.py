"""
Bidirectional LSTM (BiLSTM) Training & Evaluation Pipeline
==========================================================
Standard PyTorch Deep Learning training pipeline for:
  - Fake News Detection (train.csv / val.csv / test.csv)
  - Hate Speech Detection (hate_train.csv / hate_val.csv / hate_test.csv)

Outputs:
  - Best model checkpoint: models/deep_learning/{task}/bilstm/best_model.pt
  - Evaluation metrics: models/deep_learning/{task}/bilstm/dl_benchmark_results.json
  - Confusion Matrix plot: models/deep_learning/{task}/bilstm/test_confusion_matrix.png
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure src/ is in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.models.bilstm import BiLSTMClassifier


def get_device_info():
    """Detects available hardware and prints GPU diagnostics."""
    print("=" * 65)
    print(" HARDWARE & RUNTIME DIAGNOSTICS (BiLSTM)")
    print("=" * 65)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f" [✓] NVIDIA CUDA GPU Detected: {gpu_name}")
        print(f" [✓] Available VRAM: {total_vram_gb:.2f} GB")
        print(f" [✓] CUDA Version: {torch.version.cuda}")
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print(" [✓] Apple Silicon (MPS) Detected.")
        device = torch.device("mps")
    else:
        print(" [!] Running on CPU.")
        device = torch.device("cpu")
    print("=" * 65)
    return device


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

    for df in [df_train, df_val, df_test]:
        if 'headline' in df.columns:
            df['text'] = df['headline'].fillna('') + " " + df['article_text'].fillna('')
        else:
            df['text'] = df['article_text'].fillna('')
        df['text'] = df['text'].astype(str).str.strip()
        df['label'] = df['label'].astype(int)

    print(f"  Train samples: {len(df_train):,} | Class 0: {(df_train['label']==0).sum():,}, Class 1: {(df_train['label']==1).sum():,}")
    print(f"  Val samples:   {len(df_val):,} | Class 0: {(df_val['label']==0).sum():,}, Class 1: {(df_val['label']==1).sum():,}")
    print(f"  Test samples:  {len(df_test):,} | Class 0: {(df_test['label']==0).sum():,}, Class 1: {(df_test['label']==1).sum():,}")

    return df_train, df_val, df_test


def evaluate_model(model, dataloader, device, criterion):
    """Evaluates the model on a given dataloader."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    avg_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)

    metrics = {
        'loss': round(float(avg_loss), 4),
        'accuracy': round(float(acc), 4),
        'f1': round(float(f1), 4),
        'precision': round(float(prec), 4),
        'recall': round(float(rec), 4),
        'f1_macro': round(float(f1_macro), 4),
        'precision_macro': round(float(prec_macro), 4),
        'recall_macro': round(float(rec_macro), 4)
    }
    return metrics, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description="Train BiLSTM on Fake News or Hate Speech")
    parser.add_argument('--task', type=str, choices=['fakenews', 'hatespeech'], default='fakenews',
                        help="Classification task")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training and eval")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--embed_dim', type=int, default=128, help="Embedding dimension")
    parser.add_argument('--hidden_dim', type=int, default=128, help="LSTM hidden dimension")
    parser.add_argument('--num_layers', type=int, default=2, help="Number of stacked BiLSTM layers")
    parser.add_argument('--max_length', type=int, default=200, help="Maximum sequence length")
    parser.add_argument('--output_dir', type=str, default=None, help="Directory to save artifacts")

    args = parser.parse_args()

    device = get_device_info()

    if args.output_dir is None:
        output_dir = os.path.join(BASE_DIR, 'models', 'deep_learning', args.task, 'bilstm')
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Data
    df_train, df_val, df_test = load_data(args.task)

    # 2. Tokenizer (Industry Standard Subword Tokenizer)
    print("\n[Tokenizer] Initializing standard fast tokenizer ('bert-base-uncased')...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    def tokenize_df(df):
        dataset = Dataset.from_pandas(df[['text', 'label']])
        def tok_fn(examples):
            return tokenizer(examples['text'], truncation=True, max_length=args.max_length)
        tokenized = dataset.map(tok_fn, batched=True, remove_columns=['text'])
        # Rename label -> labels for PyTorch collation
        tokenized = tokenized.rename_column("label", "labels")
        return tokenized

    print("[Tokenizer] Tokenizing datasets...")
    train_ds = tokenize_df(df_train)
    val_ds = tokenize_df(df_val)
    test_ds = tokenize_df(df_test)

    # 3. DataLoaders with Dynamic Batch Padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, collate_fn=data_collator)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, collate_fn=data_collator)

    # 4. Initialize BiLSTM
    print(f"\n[Model] Initializing BiLSTM (vocab={tokenizer.vocab_size}, embed_dim={args.embed_dim}, hidden_dim={args.hidden_dim}, layers={args.num_layers})...")
    model = BiLSTMClassifier(
        vocab_size=tokenizer.vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=2,
        dropout=0.25,
        pad_idx=tokenizer.pad_token_id
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)

    # 5. Training Loop
    print("\n" + "=" * 65)
    print(f" STARTING BiLSTM TRAINING ({args.epochs} Epochs) ON {args.task.upper()}")
    print("=" * 65)

    best_val_f1 = -1.0
    best_model_path = os.path.join(output_dir, "best_model.pt")
    start_train_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        train_samples = 0
        epoch_start = time.time()

        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * len(labels)
            train_samples += len(labels)

        train_loss = running_loss / train_samples
        val_metrics, _, _ = evaluate_model(model, val_loader, device, criterion)
        scheduler.step(val_metrics['f1'])

        epoch_duration = time.time() - epoch_start
        print(f"Epoch {epoch:02d}/{args.epochs:02d} [{epoch_duration:.1f}s] - Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']*100:.2f}% | Val F1: {val_metrics['f1']*100:.2f}%")

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save(model.state_dict(), best_model_path)
            print(f"  ↳ [✓] New best validation F1 ({best_val_f1*100:.2f}%). Checkpoint saved.")

    total_train_time = round(time.time() - start_train_time, 2)
    print(f"\n[Training Complete] Total time: {total_train_time} seconds.")

    # 6. Evaluation on Unseen Test Split using Best Checkpoint
    print("\n" + "=" * 65)
    print(" RUNNING FINAL EVALUATION ON UNSEEN TEST SPLIT")
    print("=" * 65)
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    test_metrics, test_preds, test_labels = evaluate_model(model, test_loader, device, criterion)

    print(f"Test Accuracy:    {test_metrics['accuracy']*100:.2f}%")
    print(f"Test F1-Score:    {test_metrics['f1']*100:.2f}%")
    print(f"Test Precision:   {test_metrics['precision']*100:.2f}%")
    print(f"Test Recall:      {test_metrics['recall']*100:.2f}%")
    print(f"Test Macro F1:    {test_metrics['f1_macro']*100:.2f}%")

    cm = confusion_matrix(test_labels, test_preds).tolist()

    id2label = {0: "Real News" if args.task == 'fakenews' else "Safe",
                1: "Fake News" if args.task == 'fakenews' else "Hate Speech / Hostile"}

    # 7. Save Benchmark Results JSON
    benchmark_payload = {
        "task": args.task,
        "model_architecture": "BiLSTM",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "train_time_sec": total_train_time,
        "test_metrics": test_metrics,
        "confusion_matrix": cm,
        "labels": id2label
    }

    metrics_file = os.path.join(output_dir, "dl_benchmark_results.json")
    with open(metrics_file, "w") as f:
        json.dump(benchmark_payload, f, indent=2)
    print(f"\n[Saving] Metrics benchmark saved to: {metrics_file}")

    # 8. Generate & Save Confusion Matrix Plot
    try:
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=[id2label[0], id2label[1]],
                    yticklabels=[id2label[0], id2label[1]])
        plt.title(f"BiLSTM - Confusion Matrix ({args.task.upper()})")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        cm_plot_path = os.path.join(output_dir, "test_confusion_matrix.png")
        plt.savefig(cm_plot_path, dpi=300)
        plt.close()
        print(f"[Saving] Confusion matrix plot saved to: {cm_plot_path}")
    except Exception as e:
        print(f"[Warning] Could not generate confusion matrix plot: {e}")

    # Save tokenizer configuration for inference
    tokenizer.save_pretrained(output_dir)
    print(f"[Saving] Tokenizer config saved to: {output_dir}")
    print("\n[Done] BiLSTM execution complete!\n")


if __name__ == "__main__":
    main()
