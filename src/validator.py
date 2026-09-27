"""
TMG 2nd-Line Validation Harness
Loads a pre-trained Hugging Face checkpoint and treats it as a vendor black-box
NLP triage engine for insurance claims.
"""

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Tuple, Dict
import warnings

warnings.filterwarnings("ignore")


class VendorNLPTriageValidator:
    """
    Simulates a 2nd-line validation harness against a vendor-supplied
    NLP triage model.

    The model is treated as a black box: we only interact with it through
    its tokenizer and forward pass, mimicking how a 2nd-line validator
    would audit a vendor checkpoint without access to training data or
    model internals.
    """

    # Class 0 = Negative sentiment (treated as high-risk / adverse action)
    # Class 1 = Positive sentiment (treated as low-risk / favorable action)
    MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

    def __init__(self, model_name: str = None):
        self.model_name = model_name or self.MODEL_NAME
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the pre-trained checkpoint. Treat as vendor black-box."""
        print(f"[Validator] Loading vendor checkpoint: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        )
        self.model.eval()
        print("[Validator] Model loaded and set to evaluation mode.")

    def get_risk_scores(self, texts: List[str]) -> np.ndarray:
        """
        Returns the probability of the 'adverse' class (Class 0) for each text.

        In our simulated triage pipeline:
        - High score (>0.70) = Escalate to SIU / Delay investigation
        - Low score (<0.50) = Fast-track approval
        """
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            # Class 0 = Negative (High Friction), Class 1 = Positive (Low Friction)
            adverse_probs = probs[:, 0].numpy()
        return adverse_probs

    def get_binary_predictions(
        self, texts: List[str], threshold: float = 0.50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (risk_scores, binary_predictions) where:
        - 1 = Adverse action (flag for investigation)
        - 0 = Favorable action (fast-track)
        """
        scores = self.get_risk_scores(texts)
        preds = (scores >= threshold).astype(int)
        return scores, preds


def get_risk_score(texts: List[str]) -> np.ndarray:
    """
    Convenience function for quick one-off scoring.
    Creates a validator instance if one doesn't exist.
    """
    global _default_validator
    if "_default_validator" not in globals() or _default_validator is None:
        _default_validator = VendorNLPTriageValidator()
    return _default_validator.get_risk_scores(texts)