"""
Fake News Controller
====================
Handles:
  1. Input validation (raw text or link/URL)
  2. URL scraping and headline extraction
  3. Preprocessing and TF-IDF transformation
  4. Model inference
  5. Returning verdict + confidence score
"""

import os
import time
import joblib
from typing import Optional, Dict, Any
from fastapi import HTTPException
from pydantic import BaseModel, Field
from controllers.scraper_utils import is_valid_url, scrape_url_content

# Model Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
VECTORIZER_PATH = os.path.join(MODELS_DIR, 'fake_news_tfidf_vectorizer.joblib')
MODEL_PATH = os.path.join(MODELS_DIR, 'fake_news_best_model.joblib')

# In-memory cached models and vectorizer
_vectorizer = None
_model = None
_bert_tokenizer = None
_bert_model = None
_bert_device = None

def get_fake_news_bert():
    """Loads and caches fine-tuned BERT model (Highest F1: 97.02%)."""
    global _bert_tokenizer, _bert_model, _bert_device
    if _bert_model is None:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        _bert_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        bert_path = os.path.join(MODELS_DIR, 'deep_learning', 'fakenews', 'bert-base-uncased', 'best_model')
        if not os.path.exists(bert_path):
            raise FileNotFoundError(f"BERT model path not found at: {bert_path}")
        _bert_tokenizer = AutoTokenizer.from_pretrained(bert_path)
        _bert_model = AutoModelForSequenceClassification.from_pretrained(bert_path).to(_bert_device)
        _bert_model.eval()
    return _bert_tokenizer, _bert_model, _bert_device

def get_fake_news_model():
    """Loads and caches baseline SVM model and vectorizer in memory."""
    global _vectorizer, _model
    if _vectorizer is None:
        if not os.path.exists(VECTORIZER_PATH):
            raise RuntimeError(f"Vectorizer file not found at: {VECTORIZER_PATH}")
        _vectorizer = joblib.load(VECTORIZER_PATH)
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"Model file not found at: {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _vectorizer, _model

class FakeNewsInput(BaseModel):
    text: Optional[str] = Field(None, description="Raw news article text or claim headline")
    url: Optional[str] = Field(None, description="Web link/URL of the news article to scrape")

def handle_fake_news_prediction(payload: FakeNewsInput) -> Dict[str, Any]:
    """
    Validates input, processes text/link, executes inference,
    and returns verdict + confidence score.
    """
    t_start = time.time()
    raw_text = (payload.text or "").strip()
    raw_url = (payload.url or "").strip()

    # Step 1: Input Validation
    if not raw_text and not raw_url:
        raise HTTPException(
            status_code=422,
            detail="Validation Error: Please provide either 'text' or 'url' to analyze."
        )

    # Check if text is actually a URL
    if raw_text and is_valid_url(raw_text) and not raw_url:
        raw_url = raw_text
        raw_text = ""

    input_type = "url" if raw_url else "text"
    headline = ""
    article_text = ""

    # Step 2: Handle Link vs. Raw Text
    if raw_url:
        headline, article_text = scrape_url_content(raw_url)
    else:
        # Validate minimum text length
        if len(raw_text) < 10:
            raise HTTPException(
                status_code=422,
                detail="Validation Error: Input text is too short. Please provide at least 10 characters."
            )
        # Parse headline from first sentence or line
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
        if len(lines) > 1:
            headline = lines[0]
            article_text = " ".join(lines[1:])
        else:
            parts = raw_text.split('. ')
            headline = parts[0] if len(parts) > 1 else raw_text[:80]
            article_text = raw_text

    # Combined input for model inference
    combined_input = f"{headline} {article_text}".strip()

    # Step 3: Run Model Inference using Best F1 Model (BERT: 97.02% F1)
    model_name = "BERT (bert-base-uncased | Best F1: 97.02%)"
    try:
        tok, bert_model, device = get_fake_news_bert()
        import torch
        with torch.no_grad():
            inputs = tok(combined_input, return_tensors='pt', truncation=True, max_length=512).to(device)
            logits = bert_model(**inputs).logits
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
            prob_real = float(probabilities[0])
            prob_fake = float(probabilities[1])
    except Exception as err:
        # Graceful fallback to Linear SVM baseline if PyTorch / BERT is unavailable
        vectorizer, model = get_fake_news_model()
        vec = vectorizer.transform([combined_input])
        probabilities = model.predict_proba(vec)[0]
        prob_real = float(probabilities[0])
        prob_fake = float(probabilities[1])
        model_name = f"Linear SVM Baseline (Fallback | F1: 93.00%)"

    # Determine Verdict and Confidence Score
    if prob_fake >= 0.5:
        verdict = "Fake News"
        confidence_score = round(prob_fake * 100, 2)
    else:
        verdict = "Real News"
        confidence_score = round(prob_real * 100, 2)

    inference_time = round((time.time() - t_start) * 1000, 2)

    # Step 4: Return Formatted Result
    return {
        "status": "success",
        "verdict": verdict,
        "confidence_score": confidence_score,
        "model_used": model_name,
        "input_type": input_type,
        "headline_analyzed": headline[:200],
        "preview_text": article_text[:300] + ("..." if len(article_text) > 300 else ""),
        "inference_time_ms": inference_time
    }
