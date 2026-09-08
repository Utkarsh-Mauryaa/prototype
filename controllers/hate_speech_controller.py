"""
Hate Speech Controller
======================
Handles:
  1. Input validation (raw text or link/URL)
  2. URL scraping if web link received
  3. Preprocessing and TF-IDF transformation
  4. Model inference
  5. Returning verdict + confidence score
"""

import os
import time
import tempfile
import joblib
from typing import Optional, Dict, Any
from fastapi import HTTPException, UploadFile
from pydantic import BaseModel, Field
from controllers.scraper_utils import is_valid_url, scrape_url_content
from controllers.media_utils import get_media_type, transcribe_media_file

# Model Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
VECTORIZER_PATH = os.path.join(MODELS_DIR, 'hate_speech_tfidf_vectorizer.joblib')
MODEL_PATH = os.path.join(MODELS_DIR, 'hate_speech_best_model.joblib')

# In-memory cached models and vectorizer
_vectorizer = None
_model = None
_bert_tokenizer = None
_bert_model = None
_bert_device = None

def get_hate_speech_bert():
    """Loads and caches fine-tuned BERT model (Highest F1: 78.08%)."""
    global _bert_tokenizer, _bert_model, _bert_device
    if _bert_model is None:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        _bert_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        bert_path = os.path.join(MODELS_DIR, 'deep_learning', 'hatespeech', 'bert-base-uncased', 'best_model')
        if not os.path.exists(bert_path):
            raise FileNotFoundError(f"BERT model path not found at: {bert_path}")
        _bert_tokenizer = AutoTokenizer.from_pretrained(bert_path)
        _bert_model = AutoModelForSequenceClassification.from_pretrained(bert_path).to(_bert_device)
        _bert_model.eval()
    return _bert_tokenizer, _bert_model, _bert_device

def get_hate_speech_model():
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

class HateSpeechInput(BaseModel):
    text: Optional[str] = Field(None, description="Raw social media comment, post, or tweet text")
    url: Optional[str] = Field(None, description="Web link/URL of post or comment thread to scrape")

def handle_hate_speech_prediction(payload: HateSpeechInput) -> Dict[str, Any]:
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
    text_to_analyze = ""

    # Step 2: Handle Link vs. Raw Text
    if raw_url:
        headline, body = scrape_url_content(raw_url)
        text_to_analyze = f"{headline} {body}".strip()
    else:
        # Validate minimum length
        if len(raw_text) < 5:
            raise HTTPException(
                status_code=422,
                detail="Validation Error: Input text is too short. Please provide at least 5 characters."
            )
        text_to_analyze = raw_text

    # Step 3: Run Model Inference using Best F1 Model (BERT: 78.08% F1)
    model_name = "BERT (bert-base-uncased | Best F1: 78.08%)"
    try:
        tok, bert_model, device = get_hate_speech_bert()
        import torch
        with torch.no_grad():
            inputs = tok(text_to_analyze, return_tensors='pt', truncation=True, max_length=256).to(device)
            logits = bert_model(**inputs).logits
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
            prob_safe = float(probabilities[0])
            prob_hostile = float(probabilities[1])
    except Exception as err:
        # Graceful fallback to Linear SVM baseline if PyTorch / BERT is unavailable
        vectorizer, model = get_hate_speech_model()
        vec = vectorizer.transform([text_to_analyze])
        probabilities = model.predict_proba(vec)[0]
        prob_safe = float(probabilities[0])
        prob_hostile = float(probabilities[1])
        model_name = f"Linear SVM Baseline (Fallback | F1: 72.59%)"

    # Determine Verdict and Confidence Score
    if prob_hostile >= 0.5:
        verdict = "Hate Speech / Hostile"
        confidence_score = round(prob_hostile * 100, 2)
    else:
        verdict = "Safe / Non-Hate"
        confidence_score = round(prob_safe * 100, 2)

    inference_time = round((time.time() - t_start) * 1000, 2)

    # Step 4: Return Formatted Result
    return {
        "status": "success",
        "verdict": verdict,
        "confidence_score": confidence_score,
        "model_used": model_name,
        "input_type": input_type,
        "preview_text": text_to_analyze[:300] + ("..." if len(text_to_analyze) > 300 else ""),
        "inference_time_ms": inference_time
    }

async def handle_hate_speech_media(file: UploadFile) -> Dict[str, Any]:
    """
    Handles uploaded audio or video files:
      1. Validates media format (.mp3, .wav, .mp4, .mov, etc.)
      2. Extracts audio from video if needed using ffmpeg
      3. Transcribes speech using OpenAI Whisper
      4. Executes Hate Speech model inference on transcribed text
      5. Returns verdict + confidence score + transcript
    """
    t_start = time.time()
    filename = file.filename or ""
    media_type = get_media_type(filename)

    if not media_type:
        ext = os.path.splitext(filename.lower())[1]
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file format '{ext}'. Allowed audio: mp3, wav, m4a, ogg, flac; video: mp4, mov, mkv, webm."
        )

    # Save uploaded file to temp path
    ext = os.path.splitext(filename.lower())[1]
    temp_file = None

    try:
        temp_fd, temp_file = tempfile.mkstemp(suffix=ext)
        with os.fdopen(temp_fd, 'wb') as f:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                f.write(chunk)

        # Transcribe speech using Whisper
        is_video = (media_type == 'video')
        transcribed_text = transcribe_media_file(temp_file, is_video=is_video)

        if not transcribed_text or len(transcribed_text.strip()) < 3:
            raise HTTPException(
                status_code=422,
                detail="No recognizable speech was detected in the uploaded audio/video file."
            )

        # Run Model Inference using Best F1 Model (BERT: 78.08% F1)
        model_name = "BERT (bert-base-uncased | Best F1: 78.08%)"
        try:
            tok, bert_model, device = get_hate_speech_bert()
            import torch
            with torch.no_grad():
                inputs = tok(transcribed_text, return_tensors='pt', truncation=True, max_length=256).to(device)
                logits = bert_model(**inputs).logits
                probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
                prob_safe = float(probabilities[0])
                prob_hostile = float(probabilities[1])
        except Exception as err:
            vectorizer, model = get_hate_speech_model()
            vec = vectorizer.transform([transcribed_text])
            probabilities = model.predict_proba(vec)[0]
            prob_safe = float(probabilities[0])
            prob_hostile = float(probabilities[1])
            model_name = f"Linear SVM Baseline (Fallback | F1: 72.59%)"

        if prob_hostile >= 0.5:
            verdict = "Hate Speech / Hostile"
            confidence_score = round(prob_hostile * 100, 2)
        else:
            verdict = "Safe / Non-Hate"
            confidence_score = round(prob_safe * 100, 2)

        inference_time = round((time.time() - t_start) * 1000, 2)

        return {
            "status": "success",
            "verdict": verdict,
            "confidence_score": confidence_score,
            "model_used": model_name,
            "input_type": media_type,
            "transcribed_text": transcribed_text,
            "preview_text": transcribed_text[:300] + ("..." if len(transcribed_text) > 300 else ""),
            "inference_time_ms": inference_time
        }

    finally:
        # Clean up temporary uploaded file
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception:
                pass

