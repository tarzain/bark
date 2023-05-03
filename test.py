from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav

# download and load all models
preload_models(cache_dir="/Users/zain/workspace/.cache/models")

# generate audio from text
text_prompt = """
     Hey there! I'm Bark, a text-to-speech model trained to sound very realistic.
"""
audio_array = generate_audio(text_prompt, history_prompt="v2/en_speaker_9")

# save audio to disk
write_wav("bark_generation.wav", SAMPLE_RATE, audio_array)