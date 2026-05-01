import os
import time
import asyncio
import edge_tts

def generate_free_audio(text, gender="male"):
    """
    Generates high-quality neural speech for free using Microsoft Edge TTS.
    """
    # Pick highly realistic neural voices
    voice = "en-US-ChristopherNeural" if gender == "male" else "en-US-JennyNeural"
    
    static_path = os.path.join(os.getcwd(), 'static')
    os.makedirs(static_path, exist_ok=True)
    
    timestamp = int(time.time())
    filename = f"response_{timestamp}.mp3"
    output_path = os.path.join(static_path, filename)
    
    async def _generate():
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)
        
    try:
        # Run the async generation
        asyncio.run(_generate())
        return f"/static/{filename}"
    except Exception as e:
        print(f"[TTS Error] Failed to generate audio: {e}")
        return None