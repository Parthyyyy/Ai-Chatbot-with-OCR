import boto3
import os

def generate_polly_audio(text, filename="response.mp3", voice="Matthew"):
    static_path = os.path.join('Ai_chatbot', 'static')
    os.makedirs(static_path, exist_ok=True)
    output_path = os.path.join(static_path, filename)

    try:
        polly = boto3.client('polly', region_name='us-east-1')

        response = polly.synthesize_speech(
            Text=text,
            OutputFormat='mp3',
            VoiceId=voice
        )

        print("[Polly RAW RESPONSE]", response)

        if 'AudioStream' not in response:
            print("[Polly ERROR] AudioStream not in response")
            return None

        with open(output_path, 'wb') as f:
            f.write(response['AudioStream'].read())

        print("[✅] Audio saved to:", output_path)
        return f"/static/{filename}"

    except Exception as e:
        print("[Polly ERROR]", str(e))
        return None

# Run the test
generate_polly_audio("Hello! I'm your assistant.", voice="Matthew")
