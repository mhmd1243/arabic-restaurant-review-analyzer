# core.py
# All the project logic: models, data, rules (no UI here)
# For every review:
#   1. Sentiment Analysis -> positive / negative
#   2. NER                -> places, people, organizations
#   3. Summarization      -> short summary of long reviews (on demand)
#   + aspects  -> what the review talks about (food, service, price...)
#   + priority -> does the restaurant need to reply quickly?

import warnings
warnings.filterwarnings('ignore')
# Core imports
import os
import re
import hashlib
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset

# Device configuration
device = 0 if torch.cuda.is_available() else -1  # 0 = GPU, -1 = CPU
device_name = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model names
BASE_SENTIMENT_MODEL = "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"  # 3 classes
FINETUNED_DIR = os.path.join("models", "sentiment-finetuned")              # 2 classes, after finetune.py
ner_model_name = "CAMeL-Lab/bert-base-arabic-camelbert-msa-ner"
summary_model_name = "csebuetnlp/mT5_multilingual_XLSum"

# لو الموديل المتدرب موجود بنستخدمه، غير كده الموديل الأصلي
if os.path.exists(os.path.join(FINETUNED_DIR, "config.json")):
    SENTIMENT_MODEL = FINETUNED_DIR
else:
    SENTIMENT_MODEL = BASE_SENTIMENT_MODEL

# Dataset
DATASET_ID = "hadyelsahar/ar_res_reviews"
POLARITY_MAP = {0: "negative", 1: "positive"}
TEST_EVERY = 5        # 1 of every 5 reviews goes to test (20%)
NER_MAX_CHARS = 1000  # NER reads the first 1000 characters (stays under 512 tokens)

# Aspects: keywords in MSA + dialect, because reviews mix both
ASPECTS = {
    "food": ["طعم", "الطعم", "لذيذ", "بارد", "محروق", "مالح", "بايت", "طازج", "جودة",
             "الأكل", "الاكل", "اللحم", "الدجاج", "الفراخ", "الرز", "الصوص", "زيت",
             "نيء", "مستوي", "الوجبة", "الشاورما", "البرجر", "البيتزا"],
    "service": ["خدمة", "الخدمة", "الموظف", "الموظفين", "العامل", "العمال", "الجرسون",
                "النادل", "تعامل", "أسلوب", "اسلوب", "استقبال", "المحاسب", "الكاشير"],
    "price": ["سعر", "السعر", "أسعار", "اسعار", "الأسعار", "الاسعار", "غالي", "غالية",
              "مبالغ", "رخيص", "ريال", "جنيه", "الفاتورة", "الحساب"],
    "place": ["نظيف", "نظافة", "النظافة", "وسخ", "وساخة", "المكان", "الجلسات", "جلسات",
              "ديكور", "زحمة", "زحام", "مزدحم", "الحمام", "دورة المياه", "صراصير", "ذباب"],
    "waiting": ["انتظار", "الانتظار", "انتظرت", "انتظرنا", "تأخر", "تاخر", "تأخير",
                "التوصيل", "توصيل", "دليفري", "ساعة"],
}

URGENT_KEYWORDS = ["تسمم", "مستشفى", "المستشفى", "صراصير", "حشرة", "حشرات", "شعرة",
                   "البلدية", "ابلاغ", "إبلاغ", "حماية المستهلك", "لن أعود", "لن اعود",
                   "آخر مرة", "اخر مرة", "مش هرجع", "نصب", "احتيال", "فضيحة", "خطر"]

# Arabic labels for the UI
ASPECT_AR = {
    "food": "الأكل والجودة",
    "service": "الخدمة والتعامل",
    "price": "الأسعار",
    "place": "المكان والنظافة",
    "waiting": "الانتظار والتوصيل",
    "general": "عام",
}
PRIORITY_AR = {"high": "عالية", "medium": "متوسطة", "low": "منخفضة"}
SENTIMENT_AR = {"negative": "سلبي", "neutral": "محايد", "positive": "إيجابي"}
ENTITY_AR = {"LOC": "مكان", "ORG": "شركة/جهة", "PERS": "شخص", "MISC": "أخرى"}
ENTITY_COLORS = {"مكان": "#bfdbfe", "شركة/جهة": "#fde68a", "شخص": "#bbf7d0", "أخرى": "#e9d5ff"}


def load_models(with_summarizer=True):
    sentiment_pipe = pipeline(
        "text-classification",
        model=SENTIMENT_MODEL,
        device=device
    )
    ner_pipe = pipeline(
        "token-classification",
        model=ner_model_name,
        aggregation_strategy="simple",
        device=device
    )
    # CAMeL tokenizers don't save their max length, so truncation didn't cut anything
    # and a review with 804 tokens crashed the model. We set BERT's limit (512) ourselves.
    sentiment_pipe.tokenizer.model_max_length = 512
    ner_pipe.tokenizer.model_max_length = 512

    models = {"sentiment": sentiment_pipe, "ner": ner_pipe}

    if with_summarizer:
        models["sum_tokenizer"] = AutoTokenizer.from_pretrained(summary_model_name)
        models["sum_model"] = AutoModelForSeq2SeqLM.from_pretrained(summary_model_name)

    print(f"✅ Models loaded on {device_name.upper()} (sentiment: {SENTIMENT_MODEL})")
    return models


def split_of(text):
    # train or test? md5 of the text gives the same answer on any computer, every time
    # (finetune.py uses this same function, so the test reviews are never trained on)
    h = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)
    if h % TEST_EVERY == 0:
        return "test"
    return "train"


def load_reviews():
    dataset = load_dataset(DATASET_ID, split="train")
    df = dataset.to_pandas()

    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(subset="text").reset_index(drop=True)  # the dataset has repeated reviews

    df["true_sentiment"] = df["polarity"].map(POLARITY_MAP)
    df["restaurant_id"] = df["restaurant_id"].astype(str)
    df["split"] = df["text"].map(split_of)
    return df


def detect_aspects(text):
    # count the keywords of each aspect found in the review
    scores = {}
    for aspect, keywords in ASPECTS.items():
        count = 0
        for keyword in keywords:
            if keyword in text:
                count += 1
        if count > 0:
            scores[aspect] = count

    if not scores:
        return ["general"]
    # strongest aspect first
    return sorted(scores, key=scores.get, reverse=True)


def compute_priority(text, sentiment, confidence):
    score = 0
    reasons = []

    if sentiment == "negative":
        score += 2
        reasons.append(f"الريفيو سلبي (ثقة الموديل {confidence:.0%})")
    elif sentiment == "neutral":
        score += 1

    urgent_found = []
    for keyword in URGENT_KEYWORDS:
        if keyword in text:
            urgent_found.append(keyword)
    if urgent_found:
        score += 2 * min(len(urgent_found), 2)  # max +4
        reasons.append("كلمات خطيرة أو تدل على خسارة العميل: " + "، ".join(urgent_found))

    if text.count("!") >= 3:
        score += 1
        reasons.append("أسلوب كتابة منفعل (علامات تعجب كتير)")

    if score >= 4:
        level = "high"
    elif score >= 2:
        level = "medium"
    else:
        level = "low"

    if not reasons:
        reasons = ["مفيش مؤشرات خطر"]
    return level, score, reasons


def clean_entities(text, entities):
    result = []
    for e in entities:
        result.append({
            "label": ENTITY_AR.get(e["entity_group"], e["entity_group"]),
            "word": text[e["start"]:e["end"]],
            "start": int(e["start"]),
            "end": int(e["end"]),
        })
    return result


def summarize(text, models):
    text = re.sub(r"\s+", " ", text.strip())
    if len(text.split()) < 30:  # a short review is already a summary
        return text

    tokenizer = models["sum_tokenizer"]
    model = models["sum_model"]

    inputs = tokenizer(
        text,
        max_length=512,
        truncation=True,
        return_tensors="pt"
    )
    summary_ids = model.generate(
        **inputs,
        max_length=64,
        num_beams=4,
        no_repeat_ngram_size=2
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def analyze(text, models, do_summary=True):
    # full analysis of one review (used by the app)
    result = models["sentiment"](text, truncation=True, max_length=512)[0]
    sentiment = result["label"].lower()
    confidence = float(result["score"])

    ner_text = text[:NER_MAX_CHARS]
    entities = clean_entities(ner_text, models["ner"](ner_text))
    level, score, reasons = compute_priority(text, sentiment, confidence)

    summary = ""
    if do_summary and "sum_model" in models:
        summary = summarize(text, models)

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "entities": entities,
        "aspects": detect_aspects(text),
        "summary": summary,
        "priority": level,
        "priority_score": score,
        "reasons": reasons,
    }


def analyze_batch(texts, models, batch_size=16):
    # many reviews at once: the models take a batch per call (much faster)
    texts = list(texts)
    ner_texts = [t[:NER_MAX_CHARS] for t in texts]

    sentiments = models["sentiment"](texts, truncation=True, max_length=512, batch_size=batch_size)
    ner_results = models["ner"](ner_texts, batch_size=batch_size)

    rows = []
    for text, ner_text, result, ents in zip(texts, ner_texts, sentiments, ner_results):
        sentiment = result["label"].lower()
        confidence = float(result["score"])
        level, score, reasons = compute_priority(text, sentiment, confidence)
        entities = clean_entities(ner_text, ents)
        aspects = detect_aspects(text)

        entity_text = " ، ".join(f"{e['word']} ({e['label']})" for e in entities)
        rows.append({
            "pred_sentiment": sentiment,
            "confidence": round(confidence, 4),
            "aspects": "|".join(aspects),
            "main_aspect": aspects[0],
            "priority": level,
            "priority_score": score,
            "reasons": " • ".join(reasons),
            "entities": entity_text,
        })
    return rows
