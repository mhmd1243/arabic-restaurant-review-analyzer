# Arabic Restaurant Review Analyzer

Sentiment Analysis, NER and Summarization on Arabic restaurant reviews using Hugging Face Transformers.

## Models
- Sentiment: `CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment` (fine-tuned)
- NER: `CAMeL-Lab/bert-base-arabic-camelbert-msa-ner`
- Summarization: `csebuetnlp/mT5_multilingual_XLSum`

## Dataset
[hadyelsahar/ar_res_reviews](https://huggingface.co/datasets/hadyelsahar/ar_res_reviews)

## Results
| Model | Accuracy | Macro F1 |
|---|---|---|
| Baseline | 71.0% | - |
| Original | 69.4% | 0.739 |
| Fine-tuned | 88.1% | 0.854 |

## Files
- `core.py`
- `process_dataset.py`
- `finetune.py`
- `streamlit_app.py`

## Run
```bash
pip install -r requirements.txt
python finetune.py
python process_dataset.py
streamlit run streamlit_app.py
```
