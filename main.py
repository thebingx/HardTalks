#!/usr/bin/env python3
"""
HardTalks - Chat Bot with Audio Interaction
FastAPI backend with local Whisper STT, Xiaomi MiMO LLM, and pyttsx3 TTS
"""

import os
import asyncio
import json
import logging
import base64
import io
import wave
import tempfile
import httpx
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import offline components
FASTER_WHISPER_AVAILABLE = False
REALTIMETTS_AVAILABLE = False
WEB_SPEECH_AVAILABLE = False

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
    logger.info("✅ Faster Whisper imported successfully")
except ImportError as e:
    logger.warning(f"❌ Faster Whisper not available: {e}")

try:
    from RealtimeTTS import TextToAudioStream
    from RealtimeTTS.engines import PiperEngine
    REALTIMETTS_AVAILABLE = True
    logger.info("✅ RealtimeTTS with Piper engine imported successfully")
except ImportError as e:
    logger.warning(f"❌ RealtimeTTS not available: {e}")

try:
    # Check for Web Speech API availability (for fallback)
    import webbrowser
    WEB_SPEECH_AVAILABLE = True
    logger.info("✅ Web Speech API fallback available")
except ImportError as e:
    logger.warning(f"❌ Web Speech API fallback not available: {e}")

# Configuration - Mix of local and cloud services
class Config:
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
    
    # Faster Whisper Configuration
    FASTER_WHISPER_MODEL = os.getenv("FASTER_WHISPER_MODEL", "base")
    FASTER_WHISPER_DEVICE = os.getenv("FASTER_WHISPER_DEVICE", "cuda")
    FASTER_WHISPER_COMPUTE_TYPE = os.getenv("FASTER_WHISPER_COMPUTE_TYPE", "float16")
    FASTER_WHISPER_DOWNLOAD_DIR = os.getenv("FASTER_WHISPER_DOWNLOAD_DIR", "/home/bing/code/HardTalks/models")
    FASTER_WHISPER_LOCAL_FILES_ONLY = os.getenv("FASTER_WHISPER_LOCAL_FILES_ONLY", "True").lower() == "true"
    
    # RealtimeTTS Configuration
    REALTIMETTS_ENGINE = os.getenv("REALTIMETTS_ENGINE", "piper")  # piper, coqui, system
    REALTIMETTS_MODEL_PATH = os.getenv("REALTIMETTS_MODEL_PATH", "/home/bing/code/HardTalks/models/piper")
    REALTIMETTS_VOICE = os.getenv("REALTIMETTS_VOICE", "en_US-amy-medium")
    REALTIMETTS_GPU = os.getenv("REALTIMETTS_GPU", "False").lower() == "true"  # Default to False for compatibility
    
    # Multi-voice Configuration for mixed language support
    REALTIMETTS_VOICE_EN = os.getenv("REALTIMETTS_VOICE_EN", "en_US-ryan-high")      # English voice (high quality)
    REALTIMETTS_VOICE_ZH = os.getenv("REALTIMETTS_VOICE_ZH", "espeak-ng-zh")         # Chinese voice (uses espeak-ng)
    
    # MiMO Model Configuration
    MIMO_MODEL = os.getenv("MIMO_MODEL", "mimo-v2-flash")  # Use lowercase as per API
    MIMO_TEMPERATURE = float(os.getenv("MIMO_TEMPERATURE", "0.7"))
    MIMO_MAX_TOKENS = int(os.getenv("MIMO_MAX_TOKENS", "512"))

# Data structures
class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[str] = None

class ChatRequest(BaseModel):
    message: str
    audio_data: Optional[str] = None  # Base64 encoded audio
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    audio_data: Optional[str] = None  # Base64 encoded audio
    session_id: Optional[str] = None
    success: bool = True
    transcribed_text: Optional[str] = None  # The transcribed user input

# Global state - all local, no external dependencies
active_connections: Dict[str, WebSocket] = {}
whisper_model = None
realtime_tts_stream = None
tts_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting HardTalks Application...")
    
    # Initialize Faster Whisper model
    global whisper_model, FASTER_WHISPER_AVAILABLE, realtime_tts_stream, tts_engine, REALTIMETTS_AVAILABLE, WEB_SPEECH_AVAILABLE
    if FASTER_WHISPER_AVAILABLE:
        try:
            model_size = Config.FASTER_WHISPER_MODEL
            device = Config.FASTER_WHISPER_DEVICE
            compute_type = Config.FASTER_WHISPER_COMPUTE_TYPE
            
            logger.info(f"📥 Loading Faster Whisper model: {model_size}")
            logger.info(f"🖥️  Device: {device}")
            logger.info(f"🔢 Compute type: {compute_type}")
            logger.info(f"📁 Model directory: {Config.FASTER_WHISPER_DOWNLOAD_DIR}")
            logger.info(f"🔒 Local files only: {Config.FASTER_WHISPER_LOCAL_FILES_ONLY}")
            
            # Set environment for model caching
            os.environ["HF_HOME"] = Config.FASTER_WHISPER_DOWNLOAD_DIR
            os.environ["HF_HUB_CACHE"] = Config.FASTER_WHISPER_DOWNLOAD_DIR
            
            whisper_model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                download_root=Config.FASTER_WHISPER_DOWNLOAD_DIR,
                local_files_only=Config.FASTER_WHISPER_LOCAL_FILES_ONLY
            )
            
            logger.info("✅ Real Faster Whisper STT initialized successfully")
            logger.info(f"✅ Using {device.upper()} acceleration")
        except Exception as e:
            logger.error(f"❌ Failed to load Faster Whisper: {e}")
            logger.warning("⚠️  Falling back to mock STT")
            FASTER_WHISPER_AVAILABLE = False
    else:
        logger.info("ℹ️  Using mock STT (Faster Whisper not available)")
    
    # Initialize RealtimeTTS availability check
    if REALTIMETTS_AVAILABLE:
        logger.info("🎤 RealtimeTTS available, using system TTS with Piper/espeak-ng")
        logger.info(f"📁 Model path: {Config.REALTIMETTS_MODEL_PATH}")
        logger.info(f"🗣️  Default voice: {Config.REALTIMETTS_VOICE}")
        logger.info(f"🗣️  English voice: {Config.REALTIMETTS_VOICE_EN}")
        logger.info(f"🗣️  Chinese voice: {Config.REALTIMETTS_VOICE_ZH}")
        logger.info(f"🖥️  GPU enabled: {Config.REALTIMETTS_GPU}")
        logger.info("ℹ️  Note: RealtimeTTS provides the framework, system TTS handles synthesis")
        logger.info("ℹ️  Mixed language support: Auto-detects English/Chinese")
    else:
        logger.info("ℹ️  RealtimeTTS not available, will use system TTS or mock")
    
    # Check Xiaomi MiMO API configuration
    if Config.XIAOMI_API_KEY:
        logger.info(f"✅ Xiaomi MiMO API configured: {Config.MIMO_MODEL}")
        logger.info(f"📊 API Base: {Config.XIAOMI_API_BASE}")
        logger.info("🌐 MiMO LLM will use real API calls")
    else:
        logger.warning("⚠️  Xiaomi MiMO API key not found - using mock LLM")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down HardTalks application...")
    active_connections.clear()
    whisper_model = None
    realtime_tts_stream = None
    tts_engine = None

# Create FastAPI app
app = FastAPI(
    title="HardTalks",
    description="Simple chat bot with audio interaction",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="statics"), name="static")

# Real STT (Speech-to-Text) function using Faster Whisper
async def real_stt(audio_data: str) -> str:
    """Real Faster Whisper STT - converts audio to text"""
    if not FASTER_WHISPER_AVAILABLE or whisper_model is None:
        logger.warning("Faster Whisper not available, using mock")
        await asyncio.sleep(0.3)
        return "This is a mock transcription of the audio input."
    
    try:
        # Decode base64 audio data
        audio_bytes = base64.b64decode(audio_data)
        
        # Create a temporary file-like object
        audio_buffer = io.BytesIO(audio_bytes)
        
        # Run Faster Whisper transcription
        logger.info("🎤 Starting Faster Whisper transcription...")
        
        # Use the run method for transcription
        segments, info = whisper_model.transcribe(
            audio_buffer,
            language="en",  # Default to English, can be made dynamic
            beam_size=5,
            best_of=5
        )
        
        # Collect all segments
        transcription = ""
        for segment in segments:
            transcription += segment.text + " "
        
        transcription = transcription.strip()
        
        logger.info(f"✅ Transcription completed: {transcription}")
        
        if not transcription:
            return "I couldn't understand the audio. Please try again."
            
        return transcription
        
    except Exception as e:
        logger.error(f"❌ Faster Whisper error: {e}")
        return f"Speech recognition error: {str(e)}"

# Mock STT function (fallback)
async def mock_stt(audio_data: str) -> str:
    """Mock Faster Whisper STT - converts audio to text"""
    await asyncio.sleep(0.5)  # Simulate processing time
    logger.info("ℹ️  Mock STT: Processing audio data")
    return "This is a mock transcription of the audio input."

# Real LLM (Language Model) function using Xiaomi MiMO API
async def real_llm(prompt: str, session_id: Optional[str] = None) -> str:
    """Real Xiaomi MiMO-V2-Flash LLM - generates responses via API"""
    if not Config.XIAOMI_API_KEY:
        logger.warning("⚠️  MiMO API key not available, using mock LLM")
        return await mock_llm(prompt, session_id)
    
    try:
        logger.info(f"🤖 Calling Xiaomi MiMO API: {prompt[:50]}...")
        
        # Prepare the API request
        api_url = f"{Config.XIAOMI_API_BASE}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {Config.XIAOMI_API_KEY}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": Config.MIMO_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You are MiMO, Xiaomi's AI assistant. You are helpful, friendly, and concise. Respond in the same language as the user."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": Config.MIMO_TEMPERATURE,
            "max_tokens": Config.MIMO_MAX_TOKENS,
        }
        
        # Make the API call
        async with httpx.AsyncClient() as client:
            response = await client.post(
                api_url,
                json=payload,
                headers=headers,
                timeout=30.0  # 30 second timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract the response content
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]
                    logger.info(f"✅ MiMO response received: {content[:50]}...")
                    return content
                else:
                    logger.error(f"❌ Unexpected API response format: {result}")
                    return "I received an unexpected response format from the API."
            else:
                logger.error(f"❌ MiMO API error: {response.status_code} - {response.text}")
                return f"API error: {response.status_code}"
                
    except httpx.TimeoutException:
        logger.error("❌ MiMO API timeout")
        return "Sorry, the MiMO API request timed out."
    except httpx.RequestError as e:
        logger.error(f"❌ MiMO API request error: {e}")
        return f"Request error: {str(e)}"
    except Exception as e:
        logger.error(f"❌ Unexpected error in real_llm: {e}")
        return f"Sorry, I encountered an error: {str(e)}"

# Mock LLM (fallback) - 100% offline
async def mock_llm(prompt: str, session_id: Optional[str] = None) -> str:
    """Mock MiMO-V2-Flash LLM - generates responses offline"""
    await asyncio.sleep(0.3)  # Simulate processing time
    
    # Mock responses based on keywords - all offline
    prompt_lower = prompt.lower()
    
    if "hello" in prompt_lower or "hi" in prompt_lower:
        return "Hello! I'm MiMO, your AI assistant. How can I help you today?"
    elif "how are you" in prompt_lower:
        return "I'm doing great, thank you! Ready to help you with anything you need."
    elif "time" in prompt_lower:
        import datetime
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        return f"The current time is {current_time}."
    elif "weather" in prompt_lower:
        return "I don't have access to real-time weather data, but I can help you with other questions!"
    elif "thank" in prompt_lower:
        return "You're welcome! Is there anything else I can help you with?"
    elif "bye" in prompt_lower or "goodbye" in prompt_lower:
        return "Goodbye! Feel free to come back if you need anything else."
    elif "offline" in prompt_lower or "local" in prompt_lower:
        return "I'm using Xiaomi MiMO API for responses, but Faster Whisper and pyttsx3 run locally!"
    elif "what are you" in prompt_lower or "who are you" in prompt_lower:
        return "I'm MiMO, Xiaomi's AI assistant. I use Faster Whisper for speech recognition, Xiaomi MiMO for responses, and pyttsx3 for text-to-speech."
    else:
        # Generic response
        return f"I understand you said: '{prompt}'. I'm MiMO, Xiaomi's AI assistant. How can I help you?"

# Real TTS (Text-to-Speech) function using RealtimeTTS with Piper
async def real_tts(text: str) -> str:
    """RealtimeTTS with Piper engine - converts text to audio"""
    if not REALTIMETTS_AVAILABLE:
        logger.warning("RealtimeTTS not available, trying system TTS fallback")
        return await system_tts(text)
    
    try:
        logger.info(f"🎤 Converting text to audio with RealtimeTTS: {text[:50]}...")
        
        # Use system TTS which uses piper directly
        # RealtimeTTS is available but we'll use the reliable system approach
        return await system_tts(text)
        
    except Exception as e:
        logger.error(f"❌ RealtimeTTS error: {e}")
        logger.warning("⚠️  Falling back to system TTS")
        return await system_tts(text)

# System TTS fallback using piper directly
async def system_tts(text: str) -> str:
    """System TTS using piper Python API with language detection"""
    try:
        logger.info(f"🎤 Using system TTS with piper for: {text[:50]}...")
        
        import tempfile
        import wave
        import json
        import re
        
        # Detect language and select appropriate voice
        def detect_language(text: str) -> str:
            """Detect if text contains Chinese characters"""
            # Check for Chinese characters (CJK Unified Ideographs)
            if re.search(r'[\u4e00-\u9fff]', text):
                return 'zh'
            return 'en'
        
        language = detect_language(text)
        
        # Select voice based on language
        if language == 'zh':
            # Use espeak-ng for Chinese (Piper Chinese voices not available)
            voice_name = Config.REALTIMETTS_VOICE_ZH
            logger.info(f"Detected Chinese text, using espeak-ng Chinese voice")
            
            # Use espeak-ng directly for Chinese
            try:
                import subprocess
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_path = temp_file.name
                
                # Use Chinese voice for espeak-ng
                subprocess.run([
                    'espeak-ng', '-v', 'cmn', '-w', temp_path, text
                ], check=True, capture_output=True)
                
                with open(temp_path, "rb") as audio_file:
                    audio_data = audio_file.read()
                
                os.unlink(temp_path)
                
                audio_base64 = base64.b64encode(audio_data).decode()
                logger.info(f"✅ espeak-ng Chinese TTS completed: {len(audio_base64)} characters")
                return audio_base64
                
            except Exception as e:
                logger.warning(f"espeak-ng Chinese failed: {e}")
                # Fallback to English voice
                voice_name = Config.REALTIMETTS_VOICE_EN
                logger.info(f"Falling back to English voice: {voice_name}")
        
        else:
            # English text - use Piper
            voice_name = Config.REALTIMETTS_VOICE_EN
            logger.info(f"Detected English text, using voice: {voice_name}")
        
        # Use Piper for English (and fallback for Chinese)
        try:
            # Import piper components
            from piper.voice import PiperVoice
            from piper.config import PiperConfig
            import onnxruntime as ort
            
            # Get model paths
            model_path = f'{Config.REALTIMETTS_MODEL_PATH}/{voice_name}.onnx'
            config_path = f'{Config.REALTIMETTS_MODEL_PATH}/{voice_name}.onnx.json'
            
            # Check if voice exists, fallback to default
            if not os.path.exists(model_path):
                logger.warning(f"Voice {voice_name} not found, falling back to {Config.REALTIMETTS_VOICE}")
                voice_name = Config.REALTIMETTS_VOICE
                model_path = f'{Config.REALTIMETTS_MODEL_PATH}/{voice_name}.onnx'
                config_path = f'{Config.REALTIMETTS_MODEL_PATH}/{voice_name}.onnx.json'
            
            # Load config
            with open(config_path) as f:
                config_dict = json.load(f)
            config = PiperConfig.from_dict(config_dict)
            
            # Create ONNX session
            session = ort.InferenceSession(model_path)
            
            # Load voice
            voice = PiperVoice(session, config)
            
            # Generate audio
            audio = voice.synthesize(text)
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Write audio to WAV file
            with wave.open(temp_path, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(config.sample_rate)
                
                for chunk in audio:
                    wav_file.writeframes(chunk.audio_int16_bytes)
            
            # Read the audio file
            with open(temp_path, "rb") as audio_file:
                audio_data = audio_file.read()
            
            # Clean up
            os.unlink(temp_path)
            
            # Convert to base64
            audio_base64 = base64.b64encode(audio_data).decode()
            logger.info(f"✅ Piper TTS completed: {len(audio_base64)} characters using {voice_name}")
            return audio_base64
                
        except Exception as e:
            logger.warning(f"Piper TTS failed: {e}")
            
        # Try espeak-ng as final fallback
        try:
            import subprocess
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Use appropriate voice for espeak-ng
            voice_flag = 'cmn' if language == 'zh' else 'en-us'
            
            subprocess.run([
                'espeak-ng', '-v', voice_flag, '-w', temp_path, text
            ], check=True, capture_output=True)
            
            with open(temp_path, "rb") as audio_file:
                audio_data = audio_file.read()
            
            os.unlink(temp_path)
            
            audio_base64 = base64.b64encode(audio_data).decode()
            logger.info(f"✅ espeak-ng TTS completed: {len(audio_base64)} characters ({language})")
            return audio_base64
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Final fallback to mock
            await asyncio.sleep(0.3)
            logger.info(f"ℹ️  Mock TTS: {text[:50]}...")
            return "mock_audio_base64_data"
        
    except Exception as e:
        logger.error(f"❌ System TTS error: {e}")
        return "mock_audio_base64_data"

# Web Speech API fallback TTS
async def web_speech_tts(text: str) -> str:
    """Web Speech API fallback - generates audio using browser TTS"""
    try:
        logger.info(f"🎤 Using Web Speech API fallback for: {text[:50]}...")
        
        # Since we're running server-side, we can't use browser Web Speech API directly
        # Use system TTS as fallback
        return await system_tts(text)
        
    except Exception as e:
        logger.error(f"❌ Web Speech API fallback error: {e}")
        return "mock_audio_base64_data"

# Mock TTS (final fallback)
async def mock_tts(text: str) -> str:
    """Mock TTS - final fallback when all else fails"""
    await asyncio.sleep(0.3)
    logger.info(f"ℹ️  Mock TTS: {text[:50]}...")
    return "mock_audio_base64_data"

@app.get("/")
async def get_home():
    """Serve the main chat interface"""
    return FileResponse("statics/index.html")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    import datetime
    stt_status = "real_faster_whisper" if (FASTER_WHISPER_AVAILABLE and whisper_model) else "mock_faster_whisper"
    tts_status = "real_realtimetts_piper" if (REALTIMETTS_AVAILABLE and realtime_tts_stream) else "web_speech_fallback" if WEB_SPEECH_AVAILABLE else "mock_tts"
    llm_status = "real_xiaomi_mimo" if Config.XIAOMI_API_KEY else "mock_offline"
    
    return {
        "status": "healthy",
        "service": "HardTalks",
        "version": "1.0.0",
        "mode": "Hybrid (Local STT/TTS + Cloud LLM)",
        "timestamp": datetime.datetime.now().isoformat(),
        "components": {
            "stt": stt_status,
            "llm": llm_status,
            "tts": tts_status
        },
        "models": {
            "faster_whisper": Config.FASTER_WHISPER_MODEL if FASTER_WHISPER_AVAILABLE else "not_available",
            "realtime_tts": Config.REALTIMETTS_VOICE if REALTIMETTS_AVAILABLE else "not_available",
            "miMO": Config.MIMO_MODEL if Config.XIAOMI_API_KEY else "not_configured"
        },
        "internet_required": bool(Config.XIAOMI_API_KEY),
        "api_keys_required": bool(Config.XIAOMI_API_KEY),
        "gpu_accelerated": Config.REALTIMETTS_GPU if REALTIMETTS_AVAILABLE else False
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """HTTP chat endpoint"""
    try:
        # If audio data is provided, use STT
        transcribed_text = None
        if request.audio_data:
            text_input = await real_stt(request.audio_data)
            transcribed_text = text_input  # Store the transcribed text
        else:
            text_input = request.message
        
        # Get response from LLM - use real MiMO if available
        if Config.XIAOMI_API_KEY:
            llm_response = await real_llm(text_input, request.session_id)
        else:
            llm_response = await mock_llm(text_input, request.session_id)
        
        # Generate audio response - use RealtimeTTS with fallbacks
        if REALTIMETTS_AVAILABLE and realtime_tts_stream:
            audio_data = await real_tts(llm_response)
        else:
            audio_data = await web_speech_tts(llm_response)
        
        return ChatResponse(
            response=llm_response,
            audio_data=audio_data,
            session_id=request.session_id,
            success=True,
            transcribed_text=transcribed_text
        )
    except Exception as e:
        logger.error(f"❌ Error in chat endpoint: {e}")
        return ChatResponse(
            response="Sorry, I encountered an error processing your request.",
            session_id=request.session_id,
            success=False
        )

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket chat endpoint for real-time interaction"""
    await websocket.accept()
    session_id = str(id(websocket))
    active_connections[session_id] = websocket
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Process message
            text_input = message_data.get("message", "")
            audio_data = message_data.get("audio_data")
            transcribed_text = None
            
            # If audio data is provided, use STT
            if audio_data:
                text_input = await real_stt(audio_data)
                transcribed_text = text_input  # Store the transcribed text
            
            # Get response from LLM - use real MiMO if available
            if Config.XIAOMI_API_KEY:
                llm_response = await real_llm(text_input, session_id)
            else:
                llm_response = await mock_llm(text_input, session_id)
            
            # Generate audio response - use RealtimeTTS with fallbacks
            if REALTIMETTS_AVAILABLE and realtime_tts_stream:
                audio_response = await real_tts(llm_response)
            else:
                audio_response = await web_speech_tts(llm_response)
            
            # Send response back to client
            response = {
                "response": llm_response,
                "audio_data": audio_response,
                "session_id": session_id,
                "success": True,
                "transcribed_text": transcribed_text
            }
            
            await websocket.send_json(response)
            
    except WebSocketDisconnect:
        logger.info(f"ℹ️  Client disconnected: {session_id}")
        if session_id in active_connections:
            del active_connections[session_id]
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
        try:
            await websocket.send_json({
                "response": "Sorry, an error occurred.",
                "success": False
            })
        except:
            pass
        if session_id in active_connections:
            del active_connections[session_id]

@app.post("/api/speech-to-text")
async def speech_to_text(file: UploadFile = File(...)):
    """Speech-to-text endpoint"""
    try:
        # Read audio file
        audio_data = await file.read()
        
        if FASTER_WHISPER_AVAILABLE and whisper_model:
            try:
                # Convert to base64 for our real_stt function
                audio_base64 = base64.b64encode(audio_data).decode()
                
                # Use real Faster Whisper
                transcription = await real_stt(audio_base64)
                
                return {
                    "text": transcription,
                    "success": True,
                    "offline": True
                }
            except Exception as e:
                logger.error(f"❌ Faster Whisper error: {e}")
                return {
                    "text": "Error processing audio with Faster Whisper",
                    "success": False,
                    "offline": True
                }
        else:
            # Fallback to mock
            await asyncio.sleep(0.5)
            return {
                "text": "This is a mock transcription of your audio file.",
                "success": True,
                "offline": True
            }
    except Exception as e:
        logger.error(f"❌ STT error: {e}")
        raise HTTPException(status_code=500, detail="Speech-to-text processing failed")

@app.get("/api/session/new")
async def new_session():
    """Create a new session"""
    import uuid
    session_id = str(uuid.uuid4())
    return {"session_id": session_id, "success": True}

@app.get("/api/test/stt")
async def test_stt():
    """Test endpoint to verify Faster Whisper is working"""
    if FASTER_WHISPER_AVAILABLE and whisper_model:
        return {
            "status": "available",
            "model": Config.FASTER_WHISPER_MODEL,
            "message": "Faster Whisper is ready for use",
            "offline": True
        }
    else:
        return {
            "status": "unavailable",
            "message": "Faster Whisper not available, using mock STT",
            "offline": True
        }

@app.get("/api/offline/status")
async def offline_status():
    """Check system status and component availability"""
    return {
        "status": "operational",
        "components": {
            "stt": "real_faster_whisper" if (FASTER_WHISPER_AVAILABLE and whisper_model) else "mock",
            "llm": "real_xiaomi_mimo" if Config.XIAOMI_API_KEY else "mock_offline",
            "tts": "real_realtimetts_piper" if (REALTIMETTS_AVAILABLE and realtime_tts_stream) else "web_speech_fallback" if WEB_SPEECH_AVAILABLE else "mock"
        },
        "configuration": {
            "faster_whisper": {
                "available": FASTER_WHISPER_AVAILABLE,
                "model": Config.FASTER_WHISPER_MODEL,
                "loaded": whisper_model is not None
            },
            "realtime_tts": {
                "available": REALTIMETTS_AVAILABLE,
                "engine": Config.REALTIMETTS_ENGINE,
                "model_path": Config.REALTIMETTS_MODEL_PATH,
                "voice": Config.REALTIMETTS_VOICE,
                "gpu_enabled": Config.REALTIMETTS_GPU,
                "ready": realtime_tts_stream is not None
            },
            "xiaomi_mimo": {
                "configured": bool(Config.XIAOMI_API_KEY),
                "model": Config.MIMO_MODEL,
                "api_base": Config.XIAOMI_API_BASE
            },
            "web_speech_fallback": {
                "available": WEB_SPEECH_AVAILABLE
            }
        },
        "dependencies": {
            "internet_required": bool(Config.XIAOMI_API_KEY),
            "api_keys_required": bool(Config.XIAOMI_API_KEY),
            "local_processing": ["STT", "TTS"] + (["LLM"] if not Config.XIAOMI_API_KEY else []),
            "gpu_accelerated": Config.REALTIMETTS_GPU if REALTIMETTS_AVAILABLE else False
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 70)
    print("🚀 HardTalks - AI Assistant with Xiaomi MiMO")
    print("=" * 70)
    print(f"🌐 Server: http://{Config.HOST}:{Config.PORT}")
    
    if Config.XIAOMI_API_KEY:
        print(f"📊 Mode: Hybrid (Local STT/TTS + Cloud LLM)")
        print(f"🎤 STT: {'Faster Whisper ✅' if FASTER_WHISPER_AVAILABLE else 'Mock ❌'}")
        print(f"🤖 LLM: Xiaomi MiMO ✅ ({Config.MIMO_MODEL})")
        print(f"🔊 TTS: {'RealtimeTTS + System ✅' if REALTIMETTS_AVAILABLE else 'System/Mock ❌'}")
        print(f"🖥️  GPU: {'Enabled ✅' if Config.REALTIMETTS_GPU else 'Disabled ❌'}")
        print(f"🌐 Internet: Required for LLM responses")
        print(f"🔒 Privacy: STT & TTS remain local")
    else:
        print(f"📊 Mode: 100% Offline")
        print(f"🎤 STT: {'Faster Whisper ✅' if FASTER_WHISPER_AVAILABLE else 'Mock ❌'}")
        print(f"🤖 LLM: Mock MiMO (Offline)")
        print(f"🔊 TTS: {'RealtimeTTS + System ✅' if REALTIMETTS_AVAILABLE else 'System/Mock ❌'}")
        print(f"🖥️  GPU: {'Enabled ✅' if Config.REALTIMETTS_GPU else 'Disabled ❌'}")
        print(f"🌐 Internet: Not required")
    
    print(f"📚 TTS Fallbacks: System TTS → espeak-ng → Mock")
    print(f"🔧 Models: Piper (downloaded) + espeak-ng (system)")
    
    print("=" * 70)
    
    uvicorn.run(
        "main:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=Config.DEBUG,
        log_level="info" if Config.DEBUG else "warning"
    )