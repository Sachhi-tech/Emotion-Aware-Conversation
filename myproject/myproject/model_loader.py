import os
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Limit PyTorch CPU threads so it never maxes out your processor or heats up
torch.set_num_threads(2)

# 1. Emotion Classifier (Lightweight DistilRoBERTa - ~320MB)
EMOTION_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
emotion_classifier = pipeline(
    "text-classification",
    model=EMOTION_MODEL_NAME,
    device=-1
)

# 2. Ultra-Lightweight Conversational LLM (~720MB, fast on CPU)
# Alternative if you want slightly smarter replies: "Qwen/Qwen2.5-0.5B-Instruct"
GENERATOR_MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"

gen_tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL_NAME)
gen_model = AutoModelForCausalLM.from_pretrained(
    GENERATOR_MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="cpu"
)


def predict_and_generate(user_text: str):
    if not user_text.strip():
        return "neutral", 0.0, "Please provide valid text."

    # Step A: Classify Emotion
    classification_output = emotion_classifier(user_text)[0]
    predicted_emotion = classification_output['label']
    confidence_score = round(classification_output['score'], 2)

    # Step B: Short, clear system prompt (fewer input tokens = faster CPU response)
    messages = [
        {
            "role": "system",
            "content": (
                f"You are a kind, supportive friend. The user feels {predicted_emotion}. "
                "Tell them not to overthink, assure them you will solve it slowly together, "
                "and ask what specific help they need. Reply in 2 short sentences."
            )
        },
        {"role": "user", "content": user_text}
    ]

    prompt = gen_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = gen_tokenizer([prompt], return_tensors="pt")

    # Step C: Fast CPU Generation (capped at 60 tokens to prevent CPU spikes)
    with torch.no_grad():
        outputs = gen_model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=gen_tokenizer.eos_token_id
        )

    input_length = inputs.input_ids.shape[1]
    bot_response = gen_tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()

    return predicted_emotion, confidence_score, bot_response