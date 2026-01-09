#!/usr/bin/env python3
"""
Download additional Piper TTS voices for HardTalks
Supports English and Chinese voices
"""

import os
import subprocess
import tempfile
import requests
from pathlib import Path

# Piper voice repository
PIPER_REPO = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

# Available voices to download (with correct file naming)
VOICES = {
    # English voices - better quality than amy-medium
    "en_US-lessac-medium": {
        "description": "Male, clear, natural (better than amy)",
        "base_url": f"{PIPER_REPO}/en/en_US/lessac/medium",
        "files": {
            "onnx": "en_US-lessac-medium.onnx",
            "json": "en_US-lessac-medium.onnx.json"
        }
    },
    "en_US-ryan-medium": {
        "description": "Male, warm, natural",
        "base_url": f"{PIPER_REPO}/en/en_US/ryan/medium",
        "files": {
            "onnx": "en_US-ryan-medium.onnx",
            "json": "en_US-ryan-medium.onnx.json"
        }
    },
    # Chinese voices
    "zh-huanyu-medium": {
        "description": "Chinese male (Mandarin)",
        "base_url": f"{PIPER_REPO}/zh/zh-huanyu/medium",
        "files": {
            "onnx": "zh-huanyu-medium.onnx",
            "json": "zh-huanyu-medium.onnx.json"
        }
    },
    "zh-xiaoyan": {
        "description": "Chinese female (Mandarin)",
        "base_url": f"{PIPER_REPO}/zh/zh-xiaoyan/medium",
        "files": {
            "onnx": "zh-xiaoyan-medium.onnx",
            "json": "zh-xiaoyan-medium.onnx.json"
        }
    }
}

def download_file(url, filepath):
    """Download a file with progress"""
    print(f"Downloading {filepath.name}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(filepath, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"✓ Downloaded {filepath.name} ({filepath.stat().st_size / 1024 / 1024:.1f} MB)")

def download_voice(voice_name, voice_info, models_dir):
    """Download a specific voice"""
    print(f"\n=== Downloading {voice_name} ===")
    print(f"Description: {voice_info['description']}")
    
    voice_dir = models_dir
    voice_dir.mkdir(exist_ok=True)
    
    # Download each file
    for file_type, filename in voice_info['files'].items():
        target_path = voice_dir / filename
        
        if target_path.exists():
            print(f"✓ Already exists: {filename}")
            continue
        
        # Construct URL
        url = f"{voice_info['base_url']}/{filename}"
        
        try:
            download_file(url, target_path)
        except Exception as e:
            print(f"✗ Failed to download {filename}: {e}")
            return False
    
    print(f"✓ Voice {voice_name} downloaded successfully!")
    return True

def test_voice(voice_name, models_dir):
    """Test a voice with a sample phrase"""
    print(f"\n=== Testing {voice_name} ===")
    
    # Get the correct filenames
    if voice_name == "en_US-lessac-medium":
        model_name = "en_US-lessac-medium.onnx"
        config_name = "en_US-lessac-medium.onnx.json"
    elif voice_name == "en_US-ryan-medium":
        model_name = "en_US-ryan-medium.onnx"
        config_name = "en_US-ryan-medium.onnx.json"
    elif voice_name == "zh-huanyu-medium":
        model_name = "zh-huanyu-medium.onnx"
        config_name = "zh-huanyu-medium.onnx.json"
    elif voice_name == "zh-xiaoyan":
        model_name = "zh-xiaoyan-medium.onnx"
        config_name = "zh-xiaoyan-medium.onnx.json"
    else:
        model_name = f"{voice_name}.onnx"
        config_name = f"{voice_name}.onnx.json"
    
    model_path = models_dir / model_name
    config_path = models_dir / config_name
    
    if not model_path.exists() or not config_path.exists():
        print(f"✗ Voice files not found: {model_path}, {config_path}")
        return False
    
    # Test phrase
    test_phrases = {
        "en_US": "Hello, this is a test of the new voice model.",
        "zh": "你好，这是一个新语音模型的测试。"
    }
    
    # Determine language
    if voice_name.startswith("zh"):
        test_text = test_phrases["zh"]
    else:
        test_text = test_phrases["en_US"]
    
    print(f"Test text: {test_text}")
    
    # Generate audio
    try:
        import tempfile
        import wave
        from piper.voice import PiperVoice
        from piper.config import PiperConfig
        import onnxruntime as ort
        import json
        
        # Load config
        with open(config_path) as f:
            config_dict = json.load(f)
        config = PiperConfig.from_dict(config_dict)
        
        # Create session
        session = ort.InferenceSession(str(model_path))
        voice = PiperVoice(session, config)
        
        # Generate audio
        audio = voice.synthesize(test_text)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
        
        with wave.open(temp_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(config.sample_rate)
            
            for chunk in audio:
                wav_file.writeframes(chunk.audio_int16_bytes)
        
        # Get file size
        file_size = os.path.getsize(temp_path)
        print(f"✓ Generated audio: {file_size} bytes")
        
        # Clean up
        os.unlink(temp_path)
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def main():
    """Main function"""
    models_dir = Path("/home/bing/code/HardTalks/models/piper")
    models_dir.mkdir(exist_ok=True)
    
    print("=== Piper Voice Downloader ===")
    print("This will download additional voices for better TTS quality")
    print("and Chinese language support.\n")
    
    # Show current voices
    print("Current files in models/piper:")
    current_files = [f for f in models_dir.iterdir() if f.is_file()]
    for f in current_files:
        print(f"  - {f.name}")
    
    # Download voices
    print("\nDownloading voices...")
    for voice_name, voice_info in VOICES.items():
        download_voice(voice_name, voice_info, models_dir)
    
    # Test all downloaded voices
    print("\n" + "="*50)
    print("TESTING VOICES")
    print("="*50)
    
    for voice_name in VOICES.keys():
        test_voice(voice_name, models_dir)
    
    print("\n=== Summary ===")
    print("Voices downloaded to: /home/bing/code/HardTalks/models/piper/")
    print("\nTo use a different voice, update your .env file:")
    print("REALTIMETTS_VOICE=en_US-lessac-medium  # Better male voice")
    print("REALTIMETTS_VOICE=zh-huanyu-medium     # Chinese male")
    print("REALTIMETTS_VOICE=zh-xiaoyan           # Chinese female")
    print("\nFor mixed English/Chinese support, you can:")
    print("1. Use the Chinese voices for Chinese text")
    print("2. Keep English voices for English text")
    print("3. The system will automatically detect and use the appropriate voice")

if __name__ == "__main__":
    main()