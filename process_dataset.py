# process_dataset.py
# Analyzes all the reviews once and saves the results to reviews_analyzed.parquet
# The Streamlit app reads that file, so it opens instantly instead of re-running the models.
#
# Run:
#   python process_dataset.py --limit 200   # quick test first
#   python process_dataset.py               # all reviews
#
# Progress is saved every 256 reviews: if it stops, run it again and it continues.
# If models/sentiment-finetuned exists (after finetune.py), it is used automatically.

import warnings
warnings.filterwarnings('ignore')
# Core imports
import os
import argparse
import pandas as pd
from core import SENTIMENT_MODEL, load_models, load_reviews, analyze_batch

OUTPUT = "reviews_analyzed.parquet"
CHECKPOINT = "reviews_checkpoint.parquet"

parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=None, help="number of reviews for a quick test")
parser.add_argument("--chunk", type=int, default=256, help="save progress every N reviews")
parser.add_argument("--batch-size", type=int, default=16)
args = parser.parse_args()

# 1. Data
print("📥 Loading the dataset...")
df = load_reviews()
if args.limit:
    df = df.head(args.limit)
print(f"✅ {len(df):,} reviews from {df['restaurant_id'].nunique():,} restaurants")
print(f"✅ Sentiment model: {SENTIMENT_MODEL}")

# 2. Continue from the checkpoint (if there is one)
if os.path.exists(CHECKPOINT):
    done = pd.read_parquet(CHECKPOINT)
else:
    done = pd.DataFrame()

# a checkpoint made with another model (e.g. before fine-tuning) is useless
if len(done) > 0 and done["sentiment_model"].iloc[0] != SENTIMENT_MODEL:
    print("↩️  Old checkpoint was made with a different model, starting over")
    done = pd.DataFrame()

start = len(done)
if start > 0:
    print(f"↩️  Continuing from review {start:,}")

# 3. Analyze in chunks
if start < len(df):
    print("🤖 Loading the models...")
    models = load_models(with_summarizer=False)  # summaries are made on demand in the app

    for i in range(start, len(df), args.chunk):
        chunk = df.iloc[i:i + args.chunk]
        results = analyze_batch(chunk["text"], models, args.batch_size)
        results = pd.DataFrame(results, index=chunk.index)
        results["sentiment_model"] = SENTIMENT_MODEL

        done = pd.concat([done, pd.concat([chunk, results], axis=1)])
        done.to_parquet(CHECKPOINT, index=False)
        print(f"   ✅ {min(i + args.chunk, len(df)):,} / {len(df):,}")
else:
    print("✅ Already analyzed")

# 4. Save
done = done.reset_index(drop=True)
done.to_parquet(OUTPUT, index=False)
if os.path.exists(CHECKPOINT):
    os.remove(CHECKPOINT)

# 5. Accuracy on the test split only
# (the fine-tuned model was trained on the train split, so measuring it there would be cheating)
test = done[done["split"] == "test"]
accuracy = (test["pred_sentiment"] == test["true_sentiment"]).mean()
print(f"\n🎉 Done! Results saved to {OUTPUT}")
print(f"✅ Sentiment accuracy on the test split ({len(test):,} reviews): {accuracy:.1%}")
