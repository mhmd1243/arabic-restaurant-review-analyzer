# finetune.py
# Fine-tunes the sentiment model on our restaurant reviews (2 classes: negative / positive)
# and compares it with the original model on the same test reviews.
#
# Run:
#   python finetune.py --quick   # 2 minutes test run on a few reviews (checks everything works)
#   python finetune.py           # full training (GPU: ~10 min, CPU: 1-2 hours)
#
# Output: models/sentiment-finetuned/  (model + eval_results.json)
# After it finishes: run process_dataset.py again, then restart the app.

import warnings
warnings.filterwarnings('ignore')
# Core imports
import os
import json
import argparse
import torch
import numpy as np
import evaluate
from datasets import Dataset
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from core import BASE_SENTIMENT_MODEL, FINETUNED_DIR, load_reviews

# Device configuration
device = 0 if torch.cuda.is_available() else -1  # 0 = GPU, -1 = CPU
device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✅ PyTorch version: {torch.__version__}")
print(f"✅ Device: {device_name.upper()}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--max-len", type=int, default=256, help="tokens per review")
parser.add_argument("--quick", action="store_true", help="tiny run to check everything works")
args = parser.parse_args()

model_name = BASE_SENTIMENT_MODEL
id2label = {0: "negative", 1: "positive"}
label2id = {"negative": 0, "positive": 1}

# ------------------------------------------------------------------
# 1. Data: train / validation / test
# ------------------------------------------------------------------
# test (20%)       = same split_of() rule as the app -> never seen in training
# validation (10% of train) = used during training to pick the best epoch
df = load_reviews()
test_df = df[df["split"] == "test"].reset_index(drop=True)
train_df, val_df = train_test_split(
    df[df["split"] == "train"],
    test_size=0.1,
    stratify=df.loc[df["split"] == "train", "polarity"],
    random_state=42
)

if args.quick:
    train_df = train_df.sample(200, random_state=42)
    val_df = val_df.sample(50, random_state=42)
    test_df = test_df.sample(100, random_state=42).reset_index(drop=True)
    args.epochs = 1
    print("⚡ Quick mode: 200 train / 50 validation / 100 test, 1 epoch")

print(f"✅ Train: {len(train_df):,} | Validation: {len(val_df):,} | Test: {len(test_df):,}")
majority = test_df["true_sentiment"].value_counts(normalize=True).iloc[0]
print(f"✅ Majority baseline on test: {majority:.1%}")

# ------------------------------------------------------------------
# 2. Before training: the original model on the test reviews
# ------------------------------------------------------------------
print("\n📏 Evaluating the original model on the test set...")
base_pipe = pipeline(
    "text-classification",
    model=model_name,
    device=device
)
base_pipe.tokenizer.model_max_length = 512
base_preds = base_pipe(test_df["text"].tolist(), truncation=True, max_length=512, batch_size=32)
test_df["base_pred"] = [p["label"].lower() for p in base_preds]

base_correct = test_df["base_pred"] == test_df["true_sentiment"]
base_committed = test_df["base_pred"] != "neutral"
base_results = {
    "accuracy": float(base_correct.mean()),
    "f1_macro": float(f1_score(test_df["true_sentiment"], test_df["base_pred"],
                               labels=["negative", "positive"], average="macro")),
    "neutral_rate": float((~base_committed).mean()),
    "accuracy_committed": float(base_correct[base_committed].mean()),
}
print(f"✅ Original model: accuracy {base_results['accuracy']:.1%}, "
      f"macro F1 {base_results['f1_macro']:.3f}, said neutral {base_results['neutral_rate']:.1%}")
del base_pipe

# ------------------------------------------------------------------
# 3. Tokenization
# ------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name)

lengths = [len(ids) for ids in tokenizer(train_df["text"].tolist())["input_ids"]]
print(f"\n✅ Review length: median {int(np.median(lengths))} tokens, "
      f"{np.mean(np.array(lengths) > args.max_len):.1%} longer than {args.max_len} (cut)")


def to_dataset(frame):
    frame = frame[["text", "polarity"]].rename(columns={"polarity": "label"})
    return Dataset.from_pandas(frame, preserve_index=False)


def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=args.max_len)


train_ds = to_dataset(train_df).map(tokenize, batched=True)
val_ds = to_dataset(val_df).map(tokenize, batched=True)
test_ds = to_dataset(test_df).map(tokenize, batched=True)

# ------------------------------------------------------------------
# 4. Model: same encoder, new 2-class head
# ------------------------------------------------------------------
# the original head has 3 outputs, we need 2 -> ignore_mismatched_sizes replaces only the head
# (the "newly initialized" warning is expected)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

# class weights: ~70% of reviews are positive, so mistakes on negative reviews cost more
counts = np.bincount(train_df["polarity"], minlength=2)
class_weights = torch.tensor(len(train_df) / (2 * counts), dtype=torch.float)
print(f"✅ Class counts {counts.tolist()} -> weights {class_weights.numpy().round(2).tolist()}")


class WeightedTrainer(Trainer):
    # same as Trainer, but the loss uses our class weights
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(outputs.logits.device))
        loss = loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_metric.compute(predictions=preds, references=labels)["accuracy"],
        "f1_macro": f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"],
    }


# ------------------------------------------------------------------
# 5. Training
# ------------------------------------------------------------------
batch_size = 16
steps_per_epoch = int(np.ceil(len(train_ds) / batch_size))
warmup_steps = int(0.1 * steps_per_epoch * args.epochs)  # first 10% of steps: small learning rate

training_args = TrainingArguments(
    output_dir="checkpoints",
    num_train_epochs=args.epochs,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=64,
    warmup_steps=warmup_steps,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    save_total_limit=1,
    fp16=torch.cuda.is_available(),
    logging_steps=50,
    report_to="none",
    seed=42
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics
)

print("\n🏋️ Training...")
trainer.train()

print("\nValidation results per epoch:")
for log in trainer.state.log_history:
    if "eval_f1_macro" in log:
        print(f"   epoch {log['epoch']:.0f}: accuracy {log['eval_accuracy']:.1%}, "
              f"macro F1 {log['eval_f1_macro']:.3f}, loss {log['eval_loss']:.3f}")

# ------------------------------------------------------------------
# 6. After training: the same test reviews
# ------------------------------------------------------------------
test_output = trainer.predict(test_ds)
probs = torch.softmax(torch.tensor(test_output.predictions), dim=-1).numpy()
test_df["ft_pred"] = [id2label[i] for i in probs.argmax(axis=-1)]
test_df["ft_confidence"] = probs.max(axis=-1)

ft_results = {
    "accuracy": float((test_df["ft_pred"] == test_df["true_sentiment"]).mean()),
    "f1_macro": float(f1_score(test_df["true_sentiment"], test_df["ft_pred"],
                               labels=["negative", "positive"], average="macro")),
}

print("\n" + "-" * 60)
print(f"{'':28}{'Accuracy':>12}{'Macro F1':>12}")
print(f"{'Majority baseline':28}{majority:>12.1%}{'-':>12}")
print(f"{'Original model (3 classes)':28}{base_results['accuracy']:>12.1%}{base_results['f1_macro']:>12.3f}")
print(f"{'Fine-tuned (2 classes)':28}{ft_results['accuracy']:>12.1%}{ft_results['f1_macro']:>12.3f}")
print("-" * 60)
print(classification_report(test_df["true_sentiment"], test_df["ft_pred"], digits=3))

print("Most confident mistakes (often the dataset label is the wrong one):")
errors = test_df[test_df["ft_pred"] != test_df["true_sentiment"]]
for _, row in errors.sort_values("ft_confidence", ascending=False).head(5).iterrows():
    print(f"   label={row['true_sentiment']}, model={row['ft_pred']} ({row['ft_confidence']:.2f})")
    print(f"   {row['text'][:150]}")
    print("   " + "-" * 57)

# ------------------------------------------------------------------
# 7. Save (the app reads eval_results.json for the before/after chart)
# ------------------------------------------------------------------
if args.quick:
    print("\n⚡ Quick mode: model not saved. Run without --quick for the real training.")
else:
    trainer.save_model(FINETUNED_DIR)
    tokenizer.save_pretrained(FINETUNED_DIR)
    eval_results = {
        "base_model": model_name,
        "test_size": int(len(test_df)),
        "majority_baseline": float(majority),
        "max_length": args.max_len,
        "epochs": args.epochs,
        "base": base_results,
        "finetuned": ft_results,
    }
    with open(os.path.join(FINETUNED_DIR, "eval_results.json"), "w", encoding="utf-8") as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved to {FINETUNED_DIR}")
    print("Next: python process_dataset.py   then   streamlit run streamlit_app.py")
