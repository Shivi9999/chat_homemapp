# chatbot/utils.py
import torchaudio
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pyttsx3
import os


import random
from .models import QA

from .models import QA  # Import your QA model

def get_chatbot_response(user_input):
    # Check if the user input matches any data in the QA database
    qa_entry = QA.objects.filter(question__iexact=user_input).first()

    if qa_entry:
        # If there's a match in the database, return the corresponding answer
        response = qa_entry.answer
    else:
        # Load pre-trained GPT-2 model and tokenizer
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        # Tokenize user input
        input_ids = tokenizer.encode(user_input, return_tensors='pt')

        if input_ids.size(1) == 0:
            # Handle the case where the input tensor is empty
            response = "Sorry, I couldn't understand your input."
            audio_path = save_audio(response)
            lip_sync_animation = generate_lip_sync_animation(response)
            return response, audio_path, lip_sync_animation

        # Generate response with attention mask and pad token id settings
        pad_token_id = tokenizer.eos_token_id  # Set pad_token_id to eos_token_id
        attention_mask = (input_ids != pad_token_id).long()  # Create attention mask

        output = model.generate(
            input_ids,
            max_length=50,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.95,
            pad_token_id=pad_token_id,  # Set pad_token_id to eos_token_id
            attention_mask=attention_mask  # Use the created attention mask
        )

        # Decode and return the generated response
        response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Generate lip sync animation data
    lip_sync_animation = generate_lip_sync_animation(response)

    # Integrate text-to-speech and save audio
    audio_path = save_audio(response)

    return response, audio_path, lip_sync_animation
def save_audio(text):
    # Save the audio to the same file each time
    audio_path = "static/bot_response.mp3"  # Ensure the "media" folder exists
    engine = pyttsx3.init()
    engine.save_to_file(text, audio_path)
    engine.runAndWait()
    return audio_path
def generate_lip_sync_animation(text):
    # Placeholder for lip sync animation (replace with your lip sync model)
    # This function should return lip sync animation data

    # For simplicity, generate a random sequence of lip sync data
    lip_sync_data = []
    for char in text:
        frame = {
            'mouth_open': random.uniform(0.3, 0.7),  # Simulate different degrees of mouth openness
            'head_nod': random.uniform(0.1, 0.5)     # Simulate different degrees of head nod
        }
        lip_sync_data.append(frame)

    print(f"Lip Sync Data: {lip_sync_data}")
    return lip_sync_data