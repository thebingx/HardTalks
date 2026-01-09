# HardTalks - AI Chat Assistant with Full Audio Interaction

A complete audio chat bot application with local speech recognition (Faster Whisper), cloud AI responses (Xiaomi MiMO), and local text-to-speech (Piper TTS). Built with FastAPI for high-performance real-time interaction.

## Features

- 🎤 **Full Audio Interaction**: Record your voice and get audio responses
- 🤖 **AI Chat Bot**: Powered by Xiaomi MiMO-V2-Flash API
- 🎯 **Local STT**: Faster Whisper for speech-to-text (GPU accelerated)
- 🔊 **Local TTS**: Piper TTS with espeak-ng fallback for text-to-speech
- 🚀 **FastAPI Backend**: Modern, high-performance Python web framework
- 💬 **Real-time Communication**: Supports both HTTP and WebSocket protocols
- 🎨 **Beautiful UI**: Clean, responsive chat interface
- 🔒 **Privacy-Focused**: STT and TTS remain local, only LLM uses cloud API

## Project Structure

```
HardTalks/
├── main.py              # FastAPI backend server (single file)
├── .env                 # Configuration and API keys
├── requirements.txt     # Python dependencies
├── .gitignore          # Git ignore rules
├── README.md           # This file
├── run.sh              # Quick start script
├── statics/
│   └── index.html      # Frontend chat interface
└── models/             # Downloaded AI models
    ├── faster-whisper/ # STT models
    └── piper/          # TTS models
```

## Quick Start

### 1. Install Dependencies

```bash
# Install basic dependencies
pip install -r requirements.txt

# For production, install AI models:
# pip install faster-whisper TTS torch sounddevice
```

### 2. Configure Environment

Edit the `.env` file with your configuration:

```bash
# API Keys and Configuration
XIAOMI_API_KEY=your_xiaomi_mimo_api_key_here
XIAOMI_API_BASE=https://api.xiaomi.com/v1

# Application Configuration
DEBUG=True
HOST=0.0.0.0
PORT=8000

# Model Paths (for local models)
FASTER_WHISPER_MODEL=base
TTS_MODEL_PATH=/path/to/tts/model
```

### 3. Run the Application

```bash
# Development mode (with auto-reload)
python main.py

# Or using uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Access the App

Open your browser and navigate to:
```
http://localhost:8000
```

## Architecture Overview

### Backend (main.py)
- **FastAPI** web server
- **Mock STT**: Faster Whisper interface (ready for real implementation)
- **Mock LLM**: Xiaomi MiMO-V2-Flash interface (ready for real API)
- **Mock TTS**: CoquiEngine interface (ready for real implementation)
- **HTTP API**: REST endpoints for chat and audio processing
- **WebSocket**: Real-time bidirectional communication

### Frontend (index.html)
- **Responsive Design**: Works on desktop and mobile
- **Audio Recording**: Browser-based microphone access
- **Audio Playback**: Plays AI responses as audio
- **Visual Feedback**: Typing indicators, loading states, audio visualizer
- **Dual Mode**: Supports both HTTP and WebSocket communication

## API Endpoints

### HTTP Endpoints
- `GET /` - Serves the chat interface
- `GET /health` - Health check endpoint
- `POST /api/chat` - Send chat message (text or audio)
- `POST /api/speech-to-text` - Convert audio to text
- `GET /api/session/new` - Create new session

### WebSocket Endpoint
- `ws://localhost:8000/ws/chat` - Real-time chat

## Usage Examples

### Text Chat
1. Type your message in the input field
2. Press Enter or click "Send"
3. Get AI response with text and audio

### Audio Chat
1. Click "🎤 Record Audio"
2. Speak into your microphone
3. Click "⏹️ Stop Recording"
4. Get AI response with text and audio

## Production Deployment

### For Real AI Models

1. **Install Real Models**:
   ```bash
   pip install faster-whisper TTS torch sounddevice
   ```

2. **Update main.py**:
   - Replace `mock_stt()` with real Faster Whisper implementation
   - Replace `mock_llm()` with real Xiaomi MiMO API call
   - Replace `mock_tts()` with real CoquiEngine implementation

3. **Configure GPU** (optional):
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

### Environment Variables for Production

```bash
# Real API keys
XIAOMI_API_KEY=your_production_key

# Model settings
FASTER_WHISPER_MODEL=large-v2  # Better quality, needs more resources
TTS_MODEL_PATH=/models/tts/coqui

# Performance
WORKERS=4
MAX_WORKERS=8
```

## Mock vs Production

This implementation uses **mock functions** for demonstration:

- **STT**: Returns "This is a mock transcription..."
- **LLM**: Returns keyword-based responses
- **TTS**: Returns mock audio data

To make it production-ready, replace the mock functions in `main.py` with real implementations.

## Browser Compatibility

- ✅ Chrome/Edge (full support)
- ✅ Firefox (full support)
- ✅ Safari (full support)
- ⚠️ Mobile browsers may require HTTPS for microphone access

## Security Notes

- Currently configured for development (CORS allows all origins)
- Add proper authentication for production
- Use HTTPS in production for microphone access
- Validate and sanitize all inputs
- Implement rate limiting for API endpoints

## Troubleshooting

### Microphone not working
- Check browser permissions
- Ensure HTTPS in production
- Try different browser

### Audio playback issues
- Check browser audio policies
- Ensure audio context is resumed on user interaction

### WebSocket connection fails
- Falls back to HTTP automatically
- Check firewall settings
- Verify WebSocket endpoint is accessible

## Future Enhancements

- [ ] Real-time voice streaming
- [ ] Conversation history persistence
- [ ] Multiple AI model support
- [ ] File upload for audio processing
- [ ] Voice activity detection
- [ ] Custom wake words
- [ ] Multi-language support

## License

MIT License - Feel free to use and modify for your projects.

## Support

For issues or questions, please check the troubleshooting section or refer to the FastAPI and browser API documentation.
## System Requirements

- **Python**: 3.8+
- **GPU**: NVIDIA GPU recommended (CUDA for Faster Whisper)
- **Storage**: ~2GB for models (faster-whisper-large-v2 + piper)
- **Internet**: Required for Xiaomi MiMO API calls

## Quick Start

### 1. Setup Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Edit the `.env` file with your Xiaomi MiMO API key:

```bash
# API Keys and Configuration
XIAOMI_API_KEY=your_xiaomi_mimo_api_key_here
XIAOMI_API_BASE=https://api.xiaomimimo.com/v1

# Application Configuration
DEBUG=True
HOST=0.0.0.0
PORT=8000

# Faster Whisper Configuration
FASTER_WHISPER_MODEL=large-v2
FASTER_WHISPER_DEVICE=cuda
FASTER_WHISPER_COMPUTE_TYPE=float16
FASTER_WHISPER_DOWNLOAD_DIR=/home/bing/code/HardTalks/models
FASTER_WHISPER_LOCAL_FILES_ONLY=True

# RealtimeTTS Configuration
REALTIMETTS_ENGINE=piper
REALTIMETTS_MODEL_PATH=/home/bing/code/HardTalks/models/piper
REALTIMETTS_VOICE=en_US-amy-medium
REALTIMETTS_GPU=True

# MiMO Model Configuration
MIMO_MODEL=mimo-v2-flash
MIMO_TEMPERATURE=0.7
MIMO_MAX_TOKENS=512
```

### 4. Run the Application

```bash
# Method 1: Using the run script
./run.sh

# Method 2: Direct Python
python main.py

# Method 3: Using uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Access the Application

Open your browser and navigate to:
```
http://localhost:8000
```

## API Endpoints

### HTTP Endpoints

- `GET /` - Main chat interface
- `GET /health` - Health check with system status
- `POST /api/chat` - Chat with text or audio input
- `POST /api/speech-to-text` - Convert audio to text
- `GET /api/session/new` - Create new session
- `GET /api/test/stt` - Test STT functionality
- `GET /api/offline/status` - Full system status

### WebSocket Endpoint

- `ws://localhost:8000/ws/chat` - Real-time chat with audio support

## Architecture

### Component Overview

1. **Faster Whisper STT**: Local speech recognition using NVIDIA GPU
2. **Xiaomi MiMO LLM**: Cloud-based AI responses via API
3. **Piper TTS**: Local text-to-speech with espeak-ng fallback
4. **FastAPI**: High-performance async web framework
5. **Web Interface**: Responsive chat UI with audio recording

### Data Flow

```
User Audio → Faster Whisper (STT) → Xiaomi MiMO (LLM) → Piper TTS → User Audio Response
```

## Troubleshooting

### "Faster Whisper not available"

```bash
pip install faster-whisper
```

### "Piper TTS not working"

```bash
pip install piper-tts
# Ensure models/piper/ contains .onnx and .json files
```

### "GPU not detected"

```bash
# Check CUDA installation
nvidia-smi

# Install PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### "API key not found"

Make sure your `.env` file contains:
```bash
XIAOMI_API_KEY=your_actual_key_here
```

## Privacy & Security

- ✅ **STT**: Runs locally, no audio sent to cloud
- ✅ **TTS**: Runs locally, no text sent to cloud
- ⚠️ **LLM**: Requires cloud API (Xiaomi MiMO)
- 🔒 **Data**: No persistent storage of user data

---

**Built with ❤️ using FastAPI, Faster Whisper, and Xiaomi MiMO**

## Enhanced TTS Features

### Multiple English Voices
The system now supports multiple high-quality English voices:

- **en_US-amy-medium** (default): Female, clear, natural
- **en_US-lessac-medium**: Male, clear, natural (recommended)
- **en_US-ryan-medium**: Male, warm, natural

### Chinese Language Support
The system now supports Chinese text-to-speech:

- **Automatic language detection**: Detects Chinese characters in text
- **Mixed language support**: Handles English/Chinese mixed text
- **espeak-ng fallback**: Uses system espeak-ng for Chinese when Piper models aren't available

### Voice Configuration

Update your `.env` file to use different voices:

```bash
# English voice (recommended for better quality)
REALTIMETTS_VOICE_EN=en_US-lessac-medium

# Chinese voice (uses English voice as fallback)
REALTIMETTS_VOICE_ZH=en_US-amy-medium

# Default voice (fallback)
REALTIMETTS_VOICE=en_US-amy-medium
```

### Language Detection
The system automatically:
1. Detects Chinese characters using Unicode ranges
2. Selects appropriate voice for the language
3. Falls back to espeak-ng for Chinese if needed
4. Maintains high quality for English with Piper

### Example Usage
```
English: "Hello, how are you today?"
Chinese: "你好，今天怎么样？"
Mixed: "Hello, 你好！Welcome to HardTalks."
```

## Performance Improvements

- **GPU Acceleration**: Enabled for both STT and TTS
- **Better Voice Quality**: New male voices with improved clarity
- **Language Detection**: Automatic switching between English/Chinese
- **Fallback System**: Multiple layers of fallback for reliability

