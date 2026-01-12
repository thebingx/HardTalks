# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**HardTalks** is an AI-powered job interview practice application with full audio interaction. It combines local speech processing (STT/TTS) with cloud AI responses (Xiaomi MiMO API) to provide a conversational interview coaching experience.

### Architecture

Single-file FastAPI application (`main.py`, ~1448 lines) with hybrid local/cloud processing:

```
Browser (statics/*.html) → FastAPI (main.py) → [STT: Faster Whisper (local)]
                                                    → [LLM: Xiaomi MiMO (cloud)]
                                                    → [TTS: Piper (local)]
```

**Key Design**: Privacy-focused architecture where speech-to-text and text-to-speech run locally, while LLM responses use cloud API.

## Development Commands

### Start Application
```bash
# Quick start (activates venv and starts server)
./run.sh

# Direct Python execution
python main.py

# With uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Setup
```bash
# Virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For production AI models (optional, for full functionality)
pip install faster-whisper TTS torch sounddevice
```

### Environment Configuration
Copy `.env.example` to `.env` and configure:
- `XIAOMI_API_KEY` - Required for LLM responses
- `FASTER_WHISPER_*` - STT model settings
- `REALTIMETTS_*` - TTS engine and voice settings
- `REALTIMETTS_VOICE_EN` / `REALTIMETTS_VOICE_ZH` - Language-specific voices

## Codebase Structure

### Core Files
- `main.py` - Complete FastAPI application (all backend logic in one file)
- `.env` - Environment configuration (API keys, model paths, debug settings)
- `requirements.txt` - Python dependencies
- `run.sh` - Quick start script

### Frontend (`statics/`)
- `chat.html` - Main audio/text chat interface
- `config.html` - Configuration and settings UI
- `review.html` - Interview review and feedback dashboard

### Data Storage
- `conversations/` - JSON files with saved conversation history (auto-saved)
- `scenarios/` - Job interview scenario configurations (e.g., `job_interview_scenario.json`)

### AI Models
- `models/faster-whisper/` - STT models (auto-downloaded or manually placed)
- `models/piper/` - TTS models (`.onnx` + `.json` files for each voice)

## Important Implementation Details

### Single-File Architecture
The entire backend is in `main.py`. When making changes:
- All FastAPI endpoints, WebSocket handlers, and business logic are in one file
- Configuration is via the `Config` class (reads from environment variables)
- Model initialization happens at import time with lazy loading where possible

### Multi-Language Support
- **Language Detection**: Automatic Chinese character detection (Unicode ranges)
- **Voice Selection**: Different Piper voices for English (`REALTIMETTS_VOICE_EN`) and Chinese (`REALTIMETTS_VOICE_ZH`)
- **Fallback TTS**: Piper → espeak-ng → Web Speech API (browser)

### GPU Acceleration
- Faster Whisper: `FASTER_WHISPER_DEVICE=cuda` (requires PyTorch with CUDA)
- Piper TTS: `REALTIMETTS_GPU=True` (if supported)
- Graceful fallback to CPU if GPU unavailable

### Session Management
- Unique session IDs generated via `/api/session/new` or WebSocket connection
- In-memory conversation history during session
- Auto-save to `conversations/{timestamp}.json` on session end

### API Endpoints
- `GET /` - Serves `chat.html`
- `GET /health` - Health check with component availability status
- `POST /api/chat` - Main chat endpoint (accepts text or base64 audio)
- `POST /api/speech-to-text` - Audio-to-text conversion
- `GET /api/session/new` - Create new chat session
- `GET /api/test/stt` - Test STT functionality
- `GET /api/offline/status` - Full system component status
- `WS /ws/chat` - Real-time bidirectional chat with audio support

## Common Patterns

### Adding New API Endpoints
1. Define route in `main.py` with FastAPI decorator
2. Use `ChatMessage` or create new Pydantic model for request/response
3. Handle missing dependencies gracefully (check `FASTER_WHISPER_AVAILABLE`, `REALTIMETTS_AVAILABLE`)
4. Log errors appropriately

### Modifying AI Components
- **STT**: Modify `transcribe_audio()` function
- **LLM**: Modify Xiaomi MiMO API call in chat endpoints
- **TTS**: Modify `text_to_speech()` - handles voice selection and fallback logic

### Frontend Changes
- All frontend files are standalone HTML with embedded CSS/JS
- No build process required - direct file edits
- WebSocket fallback to HTTP is handled client-side

## Testing

Check component availability:
```bash
curl http://localhost:8000/health
curl http://localhost:8000/api/offline/status
```

Test STT independently:
```bash
curl http://localhost:8000/api/test/stt
```

## Deployment Notes

- **Development**: CORS enabled for all origins, debug mode on
- **Production**: Disable debug, configure proper CORS origins, add authentication
- **HTTPS**: Required for microphone access on mobile browsers
- **GPU**: Optional but recommended for faster STT/TTS processing
