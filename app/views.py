

# chatbot/views.py
from django.shortcuts import render

from .utils import *

def chatbot(request):
    if request.method == 'POST':
        user_input = request.POST.get('user_input', '')
        response, audio_path, lip_sync_animation = get_chatbot_response(user_input)
        return render(request, 'chatbot.html', {
            'user_input': user_input,
            'response': response,
            'audio_path': audio_path,
            'lip_sync_animation': lip_sync_animation,
        })
    else:
        return render(request, 'chatbot.html', {'user_input': ''})
