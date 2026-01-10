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
import uuid
import datetime
from typing import Optional, Dict, Any, List
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
    REALTIMETTS_VOICE_ZH = os.getenv("REALTIMETTS_VOICE_ZH", "zh_CN-huayan-medium")  # Chinese voice (Piper)
    
    # Xiaomi MiMO API Configuration
    XIAOMI_API_KEY = os.getenv("XIAOMI_API_KEY", "")
    XIAOMI_API_BASE = os.getenv("XIAOMI_API_BASE", "https://api.xiaomimimo.com/v1")
    
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
    audio_enabled: Optional[bool] = True  # Whether to generate TTS audio
    job_scenario: Optional[bool] = False  # Whether to use job interview scenario

class ChatResponse(BaseModel):
    response: str
    audio_data: Optional[str] = None  # Base64 encoded audio
    session_id: Optional[str] = None
    success: bool = True
    transcribed_text: Optional[str] = None  # The transcribed user input
    conversation_saved: Optional[str] = None  # Path to saved conversation file

class EndSessionRequest(BaseModel):
    session_id: str

# Global conversation storage
conversation_histories: Dict[str, List[Dict[str, str]]] = {}
job_scenario_config: Optional[Dict[str, Any]] = None
session_file_paths: Dict[str, str] = {}  # Track conversation file paths per session

# Global state - all local, no external dependencies
active_connections: Dict[str, WebSocket] = {}
whisper_model = None
realtime_tts_stream = None
tts_engine = None

# Conversation management functions
def load_job_scenario() -> Optional[Dict[str, Any]]:
    """Load job interview scenario configuration"""
    config_path = "scenarios/job_interview_scenario.json"
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            logger.info(f"✅ Job scenario loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"❌ Failed to load job scenario: {e}")
            return None
    else:
        logger.warning(f"⚠️  Job scenario file not found: {config_path}")
        return None

def create_instruction_prompt(scenario_config: Dict[str, Any]) -> str:
    """Create structured instruction prompt for LLM based on job scenario"""
    if not scenario_config:
        return "You are MiMO, Xiaomi's AI assistant. You are helpful, friendly, and concise. Respond in the same language as the user."
    
    try:
        job_title = scenario_config.get("job_title", "Government Relations Manager")
        company_name = scenario_config.get("company_name", "Dow Chemical")
        job_description = scenario_config.get("job_description", "Responsible for managing government relations and ensuring compliance with regulations.")
        your_resume = scenario_config.get("your_resume", "Experienced in government relations with a strong background in regulatory compliance and stakeholder engagement.")
        intervierer_profile = scenario_config.get("interviewer_profile", "An experienced HR professional with expertise in interviewing candidates for managerial positions.")
        
        instruction = f"""

            You are a licensed interviewer at {company_name}, conducting a interview for the position of {job_title}.

            INTERVIEWER PERSONA: {intervierer_profile}
            
            CANDIDATE'S RESUME: {your_resume}

            JOB DESCRIPTION: {job_description}

            CRITICAL INSTRUCTIONS:
            1. YOU START THE INTERVIEW. Introduce yourself briefly according to your persona and ask the first question.
            2. TAILOR QUESTIONS: Use the candidate's resume to ask specific questions about their past projects or skills as they relate to the job description.
            3. ADOPT PERSONA: If the interviewer profile specifies a tone (e.g., tough, friendly, technical), strictly adhere to it.
            4. BE CONCISE: Keep your questions brief (max 2-3 sentences) to maintain a snappy, real-time pace.
            5. Ask ONLY ONE question at a time.
            6. Listen carefully. Evalute reponses thoughtfully. Dig deeper into their resume if an answer lacks detail.Guide the conversation naturally
            7. Start immediately once the session begins.

            """

        return instruction
    except Exception as e:
        logger.error(f"❌ Error creating instruction prompt: {e}")
        return "You are MiMO, Xiaomi's AI assistant. You are helpful, friendly, and concise. Respond in the same language as the user."

def get_conversation_history(session_id: str) -> List[Dict[str, str]]:
    """Get conversation history for a session"""
    return conversation_histories.get(session_id, [])

def add_to_conversation(session_id: str, role: str, content: str):
    """Add a message to conversation history"""
    if session_id not in conversation_histories:
        conversation_histories[session_id] = []
    
    timestamp = datetime.datetime.now().isoformat()
    conversation_histories[session_id].append({
        "role": role,
        "content": content,
        "timestamp": timestamp
    })

def save_conversation_to_file(session_id: str, job_scenario: bool = False) -> Optional[str]:
    """Save conversation to a JSON file in conversations/ folder"""
    if session_id not in conversation_histories or not conversation_histories[session_id]:
        return None
    
    try:
        # Create conversations directory
        os.makedirs("conversations", exist_ok=True)
        
        # Check if we already have a file for this session
        if session_id in session_file_paths:
            filepath = session_file_paths[session_id]
        else:
            # Generate filename with session start timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{timestamp}_{session_id[:8]}.json"
            filepath = os.path.join("conversations", filename)
            session_file_paths[session_id] = filepath
        
        # Get job data if available
        job_data = None
        if job_scenario and job_scenario_config:
            job_data = job_scenario_config
        
        # Prepare conversation data
        conversation_data = {
            "session_id": session_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "job_scenario": job_scenario,
            "job_data": job_data,
            "conversation": conversation_histories[session_id]
        }
        
        # Save to file (overwrite with complete conversation each time)
        with open(filepath, "w") as f:
            json.dump(conversation_data, f, indent=2)
        
        logger.info(f"✅ Conversation saved to {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"❌ Failed to save conversation: {e}")
        return None

def build_llm_prompt(session_id: str, new_user_input: str, job_scenario: bool = False) -> str:
    """Build the complete LLM prompt with instruction and conversation history"""
    # Get instruction prompt
    instruction = create_instruction_prompt(job_scenario_config) if job_scenario else "You are MiMO, Xiaomi's AI assistant. You are helpful, friendly, and concise. Respond in the same language as the user."
    
    # Get conversation history
    history = get_conversation_history(session_id)
    
    # Build the prompt
    prompt_parts = [f"SYSTEM: {instruction}"]
    
    if history:
        prompt_parts.append("\nCONVERSATION HISTORY:")
        for msg in history:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            prompt_parts.append(f"{role_label}: {msg['content']}")
    
    prompt_parts.append(f"\nUser's new input: {new_user_input}")
    prompt_parts.append("\nPlease provide your response as the interviewer:")
    
    return "\n".join(prompt_parts)

def end_session(session_id: str) -> Optional[str]:
    """End a session and save final conversation"""
    conversation_file = None
    
    # Save final conversation
    if session_id in conversation_histories:
        job_scenario = session_id in conversation_histories and any(
            msg.get("job_scenario", False) for msg in conversation_histories[session_id]
        )
        conversation_file = save_conversation_to_file(session_id, job_scenario)
    
    # Clean up session data
    if session_id in conversation_histories:
        del conversation_histories[session_id]
    if session_id in session_file_paths:
        del session_file_paths[session_id]
    if session_id in active_connections:
        del active_connections[session_id]
    
    logger.info(f"✅ Session {session_id} ended")
    return conversation_file

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
    
    # Load job interview scenario
    global job_scenario_config
    job_scenario_config = load_job_scenario()
    if job_scenario_config:
        logger.info(f"✅ Job interview scenario loaded: {job_scenario_config.get('job_title', 'Unknown')}")
    else:
        logger.info("ℹ️  No job interview scenario loaded")
    
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
        logger.info(f"Prompt: {prompt}")
        
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
                    "content": "You are a experienced hiring professional.Respond in the same language as the user."
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
            # Use Chinese Piper voice
            voice_name = Config.REALTIMETTS_VOICE_ZH
            logger.info(f"Detected Chinese text, using voice: {voice_name}")
        else:
            # English text - use Piper
            voice_name = Config.REALTIMETTS_VOICE_EN
            logger.info(f"Detected English text, using voice: {voice_name}")
        
        # Use Piper for both English and Chinese
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
    """Serve the configuration interface by default"""
    return FileResponse("statics/config.html")

@app.get("/chat")
async def get_chat():
    """Serve the chat interface"""
    return FileResponse("statics/chat.html")

@app.get("/review")
async def get_review():
    """Serve the review interface"""
    return FileResponse("statics/review.html")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    import datetime
    stt_status = "real_faster_whisper" if (FASTER_WHISPER_AVAILABLE and whisper_model) else "mock_faster_whisper"
    tts_status = "real_realtimetts_piper" if (REALTIMETTS_AVAILABLE and realtime_tts_stream) else "web_speech_fallback" if WEB_SPEECH_AVAILABLE else "mock_tts"
    llm_status = "real_xiaomi_mimo" if Config.XIAOMI_API_KEY else "mock_offline"
    
    # Check if config file exists
    config_exists = os.path.exists("scenarios/job_interview_scenario.json")
    
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
        "config": {
            "exists": config_exists,
            "path": "scenarios/job_interview_scenario.json"
        },
        "internet_required": bool(Config.XIAOMI_API_KEY),
        "api_keys_required": bool(Config.XIAOMI_API_KEY),
        "gpu_accelerated": Config.REALTIMETTS_GPU if REALTIMETTS_AVAILABLE else False
    }

@app.get("/api/config/load")
async def load_config():
    """Load the job interview scenario configuration"""
    config_path = "scenarios/job_interview_scenario.json"
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            logger.info(f"ℹ️  Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"❌ Failed to load config: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load configuration: {str(e)}")
    else:
        logger.warning(f"⚠️  Configuration file not found: {config_path}")
        raise HTTPException(status_code=404, detail="Configuration file not found")

@app.post("/api/config/save")
async def save_config(config: Dict[str, Any]):
    """Save the job interview scenario configuration"""
    config_path = "scenarios/job_interview_scenario.json"
    try:
        # Ensure scenarios directory exists
        os.makedirs("scenarios", exist_ok=True)
        
        # Create backup of existing config
        if os.path.exists(config_path):
            backup_path = config_path + ".backup"
            with open(config_path, "r") as src, open(backup_path, "w") as dst:
                dst.write(src.read())
            logger.info(f"ℹ️  Created backup at {backup_path}")
        
        # Save new config
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"ℹ️  Configuration saved to {config_path}")
        return {"status": "success", "message": "Configuration saved successfully", "path": config_path}
    except Exception as e:
        logger.error(f"❌ Failed to save config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save configuration: {str(e)}")

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """HTTP chat endpoint"""
    try:
        # Check if this is a session end request
        if request.message == "__END_SESSION__":
            conversation_file = end_session(request.session_id)
            return ChatResponse(
                response="Session ended. Conversation saved.",
                session_id=request.session_id,
                success=True,
                conversation_saved=conversation_file
            )
        
        # If audio data is provided, use STT
        transcribed_text = None
        if request.audio_data:
            text_input = await real_stt(request.audio_data)
            transcribed_text = text_input  # Store the transcribed text
        else:
            text_input = request.message
        
        # Add user message to conversation
        if request.session_id:
            add_to_conversation(request.session_id, "user", text_input)
        
        # Build LLM prompt with instruction and conversation history
        llm_prompt = build_llm_prompt(request.session_id, text_input, request.job_scenario)
        
        # Get response from LLM - use real MiMO if available
        if Config.XIAOMI_API_KEY:
            llm_response = await real_llm(llm_prompt, request.session_id)
        else:
            llm_response = await mock_llm(llm_prompt, request.session_id)
        
        # Add assistant response to conversation
        if request.session_id:
            add_to_conversation(request.session_id, "assistant", llm_response)
        
        # Save conversation to file
        conversation_saved = None
        if request.session_id:
            conversation_saved = save_conversation_to_file(request.session_id, request.job_scenario)
        
        # Generate audio response only if audio is enabled
        audio_data = None
        if request.audio_enabled:
            # Use RealtimeTTS with fallbacks
            if REALTIMETTS_AVAILABLE and realtime_tts_stream:
                audio_data = await real_tts(llm_response)
            else:
                audio_data = await web_speech_tts(llm_response)
        else:
            logger.info("ℹ️  Audio disabled by client, skipping TTS generation")
        
        return ChatResponse(
            response=llm_response,
            audio_data=audio_data,
            session_id=request.session_id,
            success=True,
            transcribed_text=transcribed_text,
            conversation_saved=conversation_saved
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
            
            # Check if this is a session end request
            if message_data.get("action") == "end_session":
                conversation_file = end_session(session_id)
                await websocket.send_json({
                    "success": True,
                    "action": "session_ended",
                    "conversation_file": conversation_file,
                    "message": "Session ended successfully"
                })
                await websocket.close()
                return
            
            # Process message
            text_input = message_data.get("message", "")
            audio_data = message_data.get("audio_data")
            audio_enabled = message_data.get("audio_enabled", True)  # Default to True for backward compatibility
            job_scenario = message_data.get("job_scenario", False)  # Whether to use job interview scenario
            transcribed_text = None
            
            # If audio data is provided, use STT
            if audio_data:
                text_input = await real_stt(audio_data)
                transcribed_text = text_input  # Store the transcribed text
            
            # Add user message to conversation
            add_to_conversation(session_id, "user", text_input)
            
            # Build LLM prompt with instruction and conversation history
            llm_prompt = build_llm_prompt(session_id, text_input, job_scenario)
            
            # Get response from LLM - use real MiMO if available
            if Config.XIAOMI_API_KEY:
                llm_response = await real_llm(llm_prompt, session_id)
            else:
                llm_response = await mock_llm(llm_prompt, session_id)
            
            # Add assistant response to conversation
            add_to_conversation(session_id, "assistant", llm_response)
            
            # Save conversation to file
            conversation_saved = save_conversation_to_file(session_id, job_scenario)
            
            # Generate audio response only if audio is enabled
            audio_response = None
            if audio_enabled:
                # Use RealtimeTTS with fallbacks
                if REALTIMETTS_AVAILABLE and realtime_tts_stream:
                    audio_response = await real_tts(llm_response)
                else:
                    audio_response = await web_speech_tts(llm_response)
            else:
                logger.info("ℹ️  Audio disabled by client, skipping TTS generation")
            
            # Send response back to client
            response = {
                "response": llm_response,
                "audio_data": audio_response,
                "session_id": session_id,
                "success": True,
                "transcribed_text": transcribed_text,
                "conversation_saved": conversation_saved
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

@app.get("/api/job-scenario")
async def get_job_scenario():
    """Get current job interview scenario configuration"""
    if job_scenario_config:
        return {
            "success": True,
            "scenario": job_scenario_config,
            "enabled": True
        }
    else:
        return {
            "success": False,
            "message": "No job scenario configured",
            "enabled": False
        }

@app.get("/api/review/data")
async def get_review_data():
    """Get all data needed for review page"""
    try:
        # Get job scenario
        config_path = "scenarios/job_interview_scenario.json"
        if not os.path.exists(config_path):
            logger.warning(f"⚠️  Job scenario not found: {config_path}")
            return {"success": False, "error": "No job scenario configured"}
        
        with open(config_path, "r") as f:
            job_data = json.load(f)
        
        # Get latest conversation
        conversations_dir = "conversations"
        if not os.path.exists(conversations_dir):
            logger.warning(f"⚠️  Conversations directory not found: {conversations_dir}")
            return {"success": False, "error": "No conversations found"}
        
        files = [f for f in os.listdir(conversations_dir) if f.endswith('.json')]
        if not files:
            logger.warning(f"⚠️  No conversation files found in {conversations_dir}")
            return {"success": False, "error": "No conversations found"}
        
        files.sort(key=lambda f: os.path.getmtime(os.path.join(conversations_dir, f)), reverse=True)
        latest_file = os.path.join(conversations_dir, files[0])
        
        logger.info(f"✅ Loading latest conversation: {latest_file}")
        
        with open(latest_file, "r") as f:
            conversation_data = json.load(f)
        
        # Validate conversation structure
        if not isinstance(conversation_data, dict):
            raise ValueError("Conversation data is not a valid object")
        
        if "conversation" not in conversation_data:
            raise ValueError("Conversation data missing 'conversation' field")
        
        return {
            "success": True,
            "job_data": job_data,
            "conversation": conversation_data
        }
    except Exception as e:
        logger.error(f"❌ Error getting review data: {e}")
        return {"success": False, "error": str(e)}

@app.get("/api/job/data")
async def get_job_data():
    """Get job scenario data for review"""
    try:
        config_path = "scenarios/job_interview_scenario.json"
        if not os.path.exists(config_path):
            return {"success": False, "error": "No job scenario configured"}
        
        with open(config_path, "r") as f:
            job_data = json.load(f)
        
        return {
            "success": True,
            "job_data": job_data
        }
    except Exception as e:
        logger.error(f"❌ Error loading job data: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/session/end")
async def end_session_endpoint(request: EndSessionRequest):
    """End a session and save conversation"""
    try:
        conversation_file = end_session(request.session_id)
        return {
            "success": True,
            "session_id": request.session_id,
            "conversation_file": conversation_file,
            "message": "Session ended successfully"
        }
    except Exception as e:
        logger.error(f"❌ Error ending session: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/api/conversations/latest")
async def get_latest_conversation():
    """Get the most recent conversation file"""
    try:
        conversations_dir = "conversations"
        if not os.path.exists(conversations_dir):
            return {"success": False, "error": "No conversations found"}
        
        files = [f for f in os.listdir(conversations_dir) if f.endswith('.json')]
        if not files:
            return {"success": False, "error": "No conversations found"}
        
        # Sort by modification time (most recent first)
        files.sort(key=lambda f: os.path.getmtime(os.path.join(conversations_dir, f)), reverse=True)
        
        latest_file = os.path.join(conversations_dir, files[0])
        with open(latest_file, "r") as f:
            data = json.load(f)
        
        return {
            "success": True,
            "filename": files[0],
            "conversation": data
        }
    except Exception as e:
        logger.error(f"❌ Error loading latest conversation: {e}")
        return {"success": False, "error": str(e)}

@app.get("/api/conversations/list")
async def list_conversations():
    """List all saved conversation files"""
    try:
        conversations_dir = "conversations"
        if not os.path.exists(conversations_dir):
            return {"success": True, "sessions": []}
        
        files = [f for f in os.listdir(conversations_dir) if f.endswith('.json')]
        if not files:
            return {"success": True, "sessions": []}
        
        sessions = []
        for filename in files:
            filepath = os.path.join(conversations_dir, filename)
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                
                sessions.append({
                    "filename": filename,
                    "timestamp": data.get("timestamp", ""),
                    "job_title": data.get("job_data", {}).get("job_title", "Interview") if data.get("job_data") else "Interview",
                    "message_count": len(data.get("conversation", [])),
                    "session_id": data.get("session_id", "")
                })
            except Exception as e:
                logger.warning(f"⚠️  Could not read {filename}: {e}")
                continue
        
        # Sort by timestamp (most recent first)
        sessions.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return {
            "success": True,
            "sessions": sessions
        }
    except Exception as e:
        logger.error(f"❌ Error listing conversations: {e}")
        return {"success": False, "error": str(e)}

@app.get("/api/conversations/{filename}")
async def get_conversation(filename: str):
    """Get a specific conversation file and its job data"""
    try:
        conversations_dir = "conversations"
        filepath = os.path.join(conversations_dir, filename)
        
        if not os.path.exists(filepath):
            logger.warning(f"⚠️  Conversation file not found: {filepath}")
            return {"success": False, "error": "Conversation not found"}
        
        # Load conversation
        with open(filepath, "r") as f:
            conversation_data = json.load(f)
        
        # Validate conversation structure
        if not isinstance(conversation_data, dict):
            raise ValueError("Conversation data is not a valid object")
        
        if "conversation" not in conversation_data:
            raise ValueError("Conversation file missing 'conversation' field")
        
        if not isinstance(conversation_data["conversation"], list):
            raise ValueError("'conversation' field is not an array")
        
        # Load job data (either from conversation or current config)
        job_data = conversation_data.get("job_data")
        if not job_data:
            # Try to load from current config
            config_path = "scenarios/job_interview_scenario.json"
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    job_data = json.load(f)
        
        logger.info(f"✅ Loaded conversation: {filename} with {len(conversation_data['conversation'])} messages")
        
        return {
            "success": True,
            "conversation": conversation_data,
            "job_data": job_data
        }
    except Exception as e:
        logger.error(f"❌ Error loading conversation {filename}: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/review/generate")
async def generate_review(request: Dict[str, Any]):
    """Generate interview review feedback"""
    try:
        job_data = request.get("job_data", {})
        conversation = request.get("conversation", [])
        
        logger.info(f"📝 Review generation requested - Job: {job_data.get('job_title', 'N/A')}, Messages: {len(conversation)}")
        
        if not job_data:
            return {"success": False, "error": "Missing job_data"}
        
        if not conversation:
            return {"success": False, "error": "Missing conversation"}
        
        if not isinstance(conversation, list):
            return {"success": False, "error": "Conversation must be an array"}
        
        if len(conversation) == 0:
            return {"success": False, "error": "Conversation is empty"}
        
        # Build conversation transcript
        transcript_parts = []
        for msg in conversation:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                logger.warning(f"⚠️  Invalid message format: {msg}")
                continue
            role = "Candidate" if msg["role"] == "user" else "Interviewer"
            transcript_parts.append(f"{role}: {msg['content']}")
        transcript = "\n".join(transcript_parts)
        
        # Build the feedback prompt
        feedback_prompt = f"""You are an expert Interview Coach. Review the following details of a job interview.
Job: {job_data.get('job_title', 'N/A')} at {job_data.get('company_name', 'N/A')}
Job Description: {job_data.get('job_description', 'N/A')}
Candidate Resume: {job_data.get('your_resume', 'N/A')}
Interviewer Persona: {job_data.get('interviewer_profile', 'N/A')}

Transcript of the Practice Session:
{transcript}

Provide a comprehensive, constructive feedback report in JSON format. 
Include the following:
1. overallScore: (0-100) based on the session performance.
2. summary: A 3-4 sentence overview of performance.
3. strengths: 3-4 specific points of what the candidate did well.
4. weaknesses: 3-4 specific points of what can be improved.
5. detailedAnalysis: Analysis of Communication, Technical Knowledge, Tone, and Confidence.
6. matchScore: (0-100) How well does the RESUME align with the JOB DESCRIPTION?
7. matchJustification: 2-3 sentences explaining the resume-job fit.
8. interviewerNuances: Insights into the interviewer's likely style based on the persona.
9. potentialQuestions: 4-5 tough questions this specific interviewer might ask in a real setting.
10. prepAdvice: 4-5 strategic tips on how to prepare for the REAL interview given the resume, job, and persona.

Return valid JSON matching the InterviewFeedback interface.
IMPORTANT: Return ONLY the JSON object, no additional text or markdown formatting."""

        # Call LLM API
        if Config.XIAOMI_API_KEY:
            review_response = await real_llm(feedback_prompt, "review_session")
        else:
            # Mock review for testing
            review_response = await mock_review_feedback(job_data, conversation)
        
        # Parse JSON response
        try:
            # Clean up response if it contains markdown
            cleaned_response = review_response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:-3].strip()
            elif cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:-3].strip()
            
            review_data = json.loads(cleaned_response)
            
            return {
                "success": True,
                **review_data
            }
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse review JSON: {e}")
            return {
                "success": False,
                "error": "Failed to parse review data",
                "raw_response": review_response
            }
            
    except Exception as e:
        logger.error(f"❌ Error generating review: {e}")
        return {"success": False, "error": str(e)}

async def mock_review_feedback(job_data: Dict[str, Any], conversation: List[Dict[str, str]]) -> str:
    """Mock review feedback for testing without API"""
    await asyncio.sleep(1)
    
    mock_data = {
        "overallScore": 85,
        "summary": "The candidate demonstrated strong communication skills and a good understanding of government affairs. However, there are areas where more specific examples could strengthen responses.",
        "strengths": [
            "Clear and professional communication style",
            "Strong background in government relations and policy advocacy",
            "Demonstrated experience with high-level stakeholder engagement",
            "Good understanding of cross-cultural business environments"
        ],
        "weaknesses": [
            "Could provide more specific metrics and outcomes in examples",
            "Needs to elaborate more on technical policy knowledge",
            "Some responses were too general and could be more targeted",
            "Could demonstrate more confidence in discussing challenges"
        ],
        "detailedAnalysis": {
            "communication": "Excellent - Clear, professional, and well-structured responses",
            "technicalKnowledge": "Good - Solid foundation but could dive deeper into policy specifics",
            "tone": "Professional and appropriate - Maintained business-appropriate tone",
            "confidence": "Moderate - Could be more assertive in highlighting achievements"
        },
        "matchScore": 90,
        "matchJustification": "The candidate's 16 years of government affairs experience, policy analysis skills, and stakeholder engagement expertise align very well with Dow's requirements for a Government Affairs Manager.",
        "interviewerNuances": "The interviewers are senior executives who value concrete results and strategic thinking. They likely prefer candidates who can demonstrate measurable impact and navigate complex regulatory environments.",
        "potentialQuestions": [
            "Can you provide a specific example where you successfully influenced a policy change?",
            "How would you handle a situation where business interests conflict with regulatory requirements?",
            "What strategies would you use to build relationships with key government officials?",
            "How do you stay updated on rapidly changing environmental regulations?",
            "Can you describe a time when you had to manage a crisis involving government relations?"
        ],
        "prepAdvice": [
            "Quantify your achievements with specific metrics (e.g., 'influenced policy affecting $X billion in business')",
            "Prepare detailed examples of navigating complex regulatory challenges",
            "Research Dow's current policy priorities and recent regulatory challenges",
            "Practice discussing circular economy and sustainability initiatives in detail",
            "Prepare questions that show strategic thinking about Dow's business environment"
        ]
    }
    
    return json.dumps(mock_data, ensure_ascii=False)

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