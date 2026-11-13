from django.shortcuts import render
from .model_loader import predict_and_generate

def chat_view(request):
    context = {}

    if request.method == "POST":
        user_text = request.POST.get("user_text", "").strip()
        

        if user_text:
            # Run model pipeline
            emotion, confidence, bot_response = predict_and_generate(user_text)

            # Pass variables to template context
            context = {
                "user_text": user_text,
                "emotion": emotion,
                "confidence": confidence,
                "bot_response": bot_response,
            }

    return render(request, "chat.html", context)