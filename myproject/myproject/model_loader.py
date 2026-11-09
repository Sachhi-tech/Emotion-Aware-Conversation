import os
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Disable the Hugging Face symlinks warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

print(f"[Model Loader] Initializing CPU PyTorch models (PyTorch version: {torch.__version__})...")

EMOTION_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

emotion_classifier = pipeline(
    "text-classification",
    model=EMOTION_MODEL_NAME,
    device=-1
)

GENERATOR_MODEL_NAME = "facebook/blenderbot-400M-distill"

gen_tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL_NAME)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(GENERATOR_MODEL_NAME)

print("[Model Loader] All models loaded successfully into memory.")


def predict_and_generate(user_text: str):
    """
    Takes raw user text as input, classifies the emotion using Model 1,
    and generates an empathetic response using Model 2 on CPU.
    """
    if not user_text.strip():
        return "neutral", 0.0, "Please provide valid text."

    # Step A: Predict Emotion (Model 1)
    classification_output = emotion_classifier(user_text)[0]
    predicted_emotion = classification_output['label']
    confidence_score = round(classification_output['score'], 2)

    # Step B: Construct Prompt for Generator (Model 2)
    prompt = (
        f"User text: '{user_text}' | "
        f"Detected emotion: {predicted_emotion} | "
        f"Instruction: Write a supportive and empathetic response matching the user's emotion:"
    )

    # Step C: Tokenize and Generate Response on CPU
    inputs = gen_tokenizer(prompt, return_tensors="pt")

    outputs = gen_model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    bot_response = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)

    return predicted_emotion, confidence_score, bot_response