from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from sqlmodel import Session, select
from datetime import timedelta, datetime
from jose import JWTError, jwt
import re
import subprocess
import whisper
import yt_dlp
import tempfile
import os
import gc
from pathlib import Path
from typing import Optional, List, Dict
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs

# Load environment variables from .env file
load_dotenv()

import crud
from auth import (
    create_access_token,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    SECRET_KEY,
    ALGORITHM,
    verify_password,
)
from database import init_db, get_session
from models import User, Video

# ============================
# PROGRESS TRACKING (ADDED FOR POLLING)
# ============================
progress_tracker: Dict[str, Dict] = {}

def update_progress(video_id: str, stage: str, progress: int, detail: str):
    """Update progress for a video being processed"""
    progress_tracker[video_id] = {
        "stage": stage,
        "progress": progress,
        "detail": detail,
        "timestamp": datetime.now().isoformat()
    }
    print(f"📊 PROGRESS: {video_id} - {stage} - {progress}% - {detail}")

def get_progress(video_id: str) -> Dict:
    """Get progress for a video"""
    return progress_tracker.get(video_id, {
        "stage": "starting",
        "progress": 0,
        "detail": "Starting video analysis..."
    })

# ============================
# GEMINI SUMMARIZER WITH AUTO-ROTATE
# ============================
try:
    from gemini_summarizer import GeminiSummarizer
    from api_key_rotator import key_rotator
    
    # Initialize with auto-rotation (loads all keys from environment)
    gemini_summarizer = GeminiSummarizer()
    USE_GEMINI = True
    print("✅" + "="*50)
    print("✅ Gemini 2.0 (NEW SDK) with AUTO-ROTATE initialized!")
    print(f"✅ {len(key_rotator.keys)} API keys loaded")
    print("✅ BART fallback available if all keys fail")
    print("✅" + "="*50)
    
except Exception as e:
    print(f"⚠️ Gemini initialization failed: {e}")
    USE_GEMINI = False
    gemini_summarizer = None
# ============================
# HIERARCHICAL SUMMARIZER (FALLBACK)
# ============================
try:
    from hierarchical_summarizer import HierarchicalSummarizer
    from smart_chunker import SmartChunker
    import traceback
    
    HIERARCHICAL_AVAILABLE = True
    
    # Initialize the hierarchical summarizer as fallback
    hierarchical_summarizer = HierarchicalSummarizer()
    chunker = SmartChunker(max_chunk_size=800, min_chunk_size=250)
    
    if not USE_GEMINI:
        print("✅ Using HIERARCHICAL summarizer as primary (Gemini not available)")
    else:
        print("✅ Hierarchical summarizer available as fallback")
    
except ImportError as e:
    print(f"⚠️ Hierarchical summarizer not available: {e}")
    HIERARCHICAL_AVAILABLE = False
    hierarchical_summarizer = None
    chunker = None
    
    # Create a dummy summarizer as final fallback
    class DummySummarizationService:
        def generate_hierarchical_summaries(self, chunks, duration_minutes=0):
            text = ' '.join([c.get('text', '') for c in chunks if c.get('text')])
            words = text.split()[:50]
            sample = " ".join(words)
            
            return {
                "short_summary": f"This video discusses various topics. {sample[:100]}",
                "detailed_summary": f"Video content summary. {sample[:200]}",
                "key_points": ["Content summary available", "Watch video for details"],
                "key_points_with_timestamps": [],
                "topics_covered": ["General"],
                "recommendations": ["Watch the full video for complete information"],
                "section_summaries": [],
                "chunk_summaries": [],
                "ai_model_used": "Dummy (Fallback)",
                "processing_method": "fallback",
                "chunks_processed": len(chunks) if chunks else 1
            }
    
    summarization_service = DummySummarizationService()
    print("⚠️ Using dummy summarizer as final fallback")

app = FastAPI(title="Insight Video Backend")

# ============================
# CORS
# ============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

# ============================
# STARTUP
# ============================

# Add FFmpeg to PATH if not found
ffmpeg_paths = [
    r"C:\ffmpeg\bin",
    r"C:\Users\LENOVO\Downloads\ffmpeg-master-latest-win64-gpl-shared\ffmpeg-master-latest-win64-gpl-shared\bin",
    r"C:\Program Files\ffmpeg\bin"
]

for ffmpeg_path in ffmpeg_paths:
    if os.path.exists(ffmpeg_path):
        os.environ['PATH'] = ffmpeg_path + ';' + os.environ['PATH']
        print(f"✅ Added FFmpeg path: {ffmpeg_path}")
        break
else:
    print("⚠️ FFmpeg not found in common locations. Trying system PATH...")

@app.on_event("startup")
def on_startup():
    init_db()
    # Test FFmpeg availability on startup
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ FFmpeg is available in PATH")
        else:
            print("⚠️ FFmpeg check returned non-zero")
    except Exception as e:
        print(f"⚠️ FFmpeg test failed: {e}")

# ============================
# URL NORMALIZATION FUNCTION (UPDATED FOR SHORTS)
# ============================
def normalize_youtube_url(url: str) -> str:
    """Normalize YouTube URL by removing tracking parameters and standardizing format"""
    video_id = extract_video_id(url)
    return f"https://www.youtube.com/watch?v={video_id}"

def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from various URL formats including shorts"""
    # Handle youtu.be format
    if "youtu.be/" in url:
        video_id = url.split("youtu.be/")[1].split("?")[0].split("&")[0]
        return video_id
    
    # Handle youtube.com/watch?v= format
    if "watch?v=" in url:
        video_id = url.split("watch?v=")[1].split("&")[0].split("?")[0]
        return video_id
    
    # Handle youtube.com/shorts/ format (ADDED)
    if "shorts/" in url:
        video_id = url.split("shorts/")[1].split("?")[0].split("&")[0]
        return video_id
    
    # Handle embedded URLs
    if "embed/" in url:
        video_id = url.split("embed/")[1].split("?")[0].split("&")[0]
        return video_id
    
    raise HTTPException(status_code=400, detail="INVALID_YOUTUBE_URL")

# ============================
# SCHEMAS
# ============================

class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str

    @validator("email")
    def validate_email(cls, v):
        if not re.match(r"^[^@]+@[^@]+\.[^@]+$", v):
            raise ValueError("Invalid email format")
        return v

    @validator("password")
    def validate_password(cls, v):
        if len(v) < 6:
            raise ValueError("Password must be at least 6 characters")
        return v

class LoginRequest(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user: dict

class UserResponse(BaseModel):
    id: int
    name: str
    email: str

class SummarizeRequest(BaseModel):
    youtube_url: str
    save_to_history: bool = False

# ============================
# FEEDBACK SCHEMAS
# ============================

class FeedbackRequest(BaseModel):
    rating: int  # 1-5
    comment: Optional[str] = None

# ============================
# AUTH DEPENDENCY WITH BETTER ERROR HANDLING
# ============================
async def get_current_user_optional(
    token: Optional[str] = Depends(oauth2_scheme),
    session: Session = Depends(get_session),
) -> Optional[User]:
    """Get current user if token is valid, otherwise return None"""
    if not token:
        return None
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if not email:
            return None
    except JWTError:
        return None

    user = crud.get_user_by_email(session, email)
    return user

async def get_current_user_required(
    token: str = Depends(oauth2_scheme),
    session: Session = Depends(get_session),
) -> User:
    """Get current user or raise 401 if not authenticated"""
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="MISSING_TOKEN",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="INVALID_TOKEN",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if not email:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = crud.get_user_by_email(session, email)
    if not user:
        raise credentials_exception

    return user

# ============================
# REGISTER
# ============================
@app.post("/register", response_model=Token)
def register(req: RegisterRequest, session: Session = Depends(get_session)):
    try:
        user = crud.create_user(session, req.name, req.email, req.password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    access_token = create_access_token(
        data={"sub": user.email, "user_id": user.id},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )

    return Token(
        access_token=access_token,
        token_type="bearer",
        user={"id": user.id, "name": user.name, "email": user.email},
    )

# ============================
# LOGIN
# ============================
@app.post("/token", response_model=Token)
def login(req: LoginRequest, session: Session = Depends(get_session)):
    user = crud.get_user_by_email(session, req.email)
    if not user:
        raise HTTPException(status_code=404, detail="USER_NOT_FOUND")

    if not verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="INVALID_PASSWORD")

    access_token = create_access_token(
        data={"sub": user.email, "user_id": user.id},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )

    return Token(
        access_token=access_token,
        token_type="bearer",
        user={"id": user.id, "name": user.name, "email": user.email},
    )

# ============================
# CURRENT USER
# ============================
@app.get("/me", response_model=UserResponse)
def me(current_user: User = Depends(get_current_user_required)):
    return UserResponse(
        id=current_user.id,
        name=current_user.name,
        email=current_user.email,
    )

# ============================
# PROGRESS ENDPOINT (ADDED FOR POLLING)
# ============================
@app.get("/progress/{video_id}")
def get_video_progress(video_id: str):
    """Get real-time progress of video summarization"""
    return get_progress(video_id)

# ============================
# YOUTUBE HELPERS (UPDATED REGEX FOR SHORTS)
# ============================

YOUTUBE_REGEX = re.compile(
    r"(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)[\w\-]+"
)

def is_valid_youtube_url(url: str) -> bool:
    return bool(YOUTUBE_REGEX.match(url))

def get_video_metadata(url: str) -> dict:
    try:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'no_color': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            
            metadata = {
                "title": info.get('title', 'Untitled Video'),
                "duration": info.get('duration', 0),
                "thumbnail": info.get('thumbnail'),
                "uploader": info.get('uploader'),
                "view_count": info.get('view_count'),
                "upload_date": info.get('upload_date'),
            }
            return metadata
            
    except Exception as e:
        print(f"❌ Video metadata error: {e}")
        raise HTTPException(status_code=400, detail=f"VIDEO_METADATA_FAILED: {str(e)}")

def get_video_transcript(video_id: str) -> str:
    """Extract audio from YouTube video and transcribe using Whisper"""
    try:
        # Try to get transcript using youtube_transcript_api first (faster, no download)
        try:
            from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
            
            print(f"📝 Attempting to fetch transcript directly for video: {video_id}")
            # Fix: Use correct method name
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Try to get English transcript
            transcript = None
            for t in transcript_list:
                if t.language_code.startswith('en'):
                    transcript = t
                    break
            
            if not transcript:
                transcript = transcript_list[0]  # Take first available
            
            transcript_data = transcript.fetch()
            transcript_text = " ".join([item['text'] for item in transcript_data])
            print(f"✅ Direct transcript fetched successfully: {len(transcript_text)} characters, {len(transcript_text.split())} words")
            return transcript_text
            
        except ImportError:
            print("⚠️ youtube_transcript_api not installed. Install with: pip install youtube-transcript-api")
        except NoTranscriptFound:
            print("⚠️ No transcript found for this video")
        except TranscriptsDisabled:
            print("⚠️ Transcripts are disabled for this video")
        except Exception as e:
            print(f"⚠️ Direct transcript fetch failed: {e}")
        
        # If direct transcript fails, fall back to audio download
        with tempfile.TemporaryDirectory() as tmpdir:
            print(f"🔊 Downloading audio for video: {video_id}")
            
            # Improved yt-dlp options to avoid 403 error
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(tmpdir, 'audio.%(ext)s'),
                'quiet': False,
                'no_warnings': False,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'extract_audio': True,
                'audio_format': 'mp3',
                'retries': 10,
                'fragment_retries': 10,
                'skip_unavailable_fragments': True,
                
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-us,en;q=0.5',
                    'Sec-Fetch-Mode': 'navigate',
                },
                
                'extractor_args': {
                    'youtube': {
                        'player_client': ['android', 'web'],
                        'skip': ['webpage', 'dash'],
                    }
                },
                
                'socket_timeout': 30,
                'verbose': False,
            }
            
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            audio_path = None
            
            try:
                print(f"📥 Starting download with yt-dlp...")
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(video_url, download=True)
                    video_title = info.get('title', 'Unknown Video')
                    print(f"✅ Successfully downloaded: {video_title}")
                
                # Find downloaded audio file
                audio_files = list(Path(tmpdir).glob("*.mp3"))
                if not audio_files:
                    for ext in ['.m4a', '.webm', '.opus']:
                        audio_files = list(Path(tmpdir).glob(f"*{ext}"))
                        if audio_files:
                            break
                
                if not audio_files:
                    raise Exception("No audio file found after download")
                
                audio_path = str(audio_files[0])
                print(f"🎵 Audio file: {os.path.basename(audio_path)}, Size: {os.path.getsize(audio_path) / 1024 / 1024:.2f} MB")
                
            except Exception as e:
                print(f"❌ Download failed: {str(e)[:200]}")
                raise HTTPException(
                    status_code=400, 
                    detail="Could not download video audio. YouTube may be blocking the request. Please try again later or try a different video."
                )
            
            # If we got here, we have an audio file
            if not audio_path:
                raise HTTPException(status_code=400, detail="Failed to download audio")
            
            # Load Whisper model
            print("🧠 Loading Whisper model...")
            print("STAGE:TRANSCRIBING")
            try:
                model = whisper.load_model("base")
                print("✅ Whisper model loaded (base)")
            except Exception as e:
                print(f"⚠️ Base model failed, trying tiny: {e}")
                model = whisper.load_model("tiny")
                print("✅ Whisper model loaded (tiny)")
            
            # Transcribe audio
            print("🎤 Transcribing audio with Whisper...")
            try:
                result = model.transcribe(
                    audio_path,
                    task='translate',
                    fp16=False,
                    verbose=False
                )
                
                transcript_text = result["text"].strip()
                
                if not transcript_text or len(transcript_text.split()) < 20:
                    result = model.transcribe(
                        audio_path,
                        task='translate',
                        fp16=False,
                        verbose=False,
                        temperature=0.0
                    )
                    transcript_text = result["text"].strip()
                
                print(f"✅ Transcription complete: {len(transcript_text)} characters, {len(transcript_text.split())} words")
                
                del model
                gc.collect()
                
                return transcript_text
                
            except Exception as e:
                print(f"❌ Transcription error: {e}")
                raise HTTPException(status_code=400, detail=f"Failed to transcribe audio: {str(e)}")
                
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)[:200]}")
        raise HTTPException(status_code=400, detail=f"Could not process video: {str(e)[:100]}")

# ============================
# DUPLICATE CHECKING FUNCTION
# ============================
def check_existing_video(session: Session, user_id: int, url: str) -> Optional[Video]:
    """Check if user already has this video in history using normalized URL"""
    # Normalize the URL for consistent comparison
    normalized_url = normalize_youtube_url(url)
    print(f"🔍 Checking for existing video with normalized URL: {normalized_url}")
    
    statement = select(Video).where(
        (Video.owner_id == user_id) & 
        (Video.url == normalized_url)
    )
    return session.exec(statement).first()

# ============================
# SUMMARIZE WITH AI (WITH PROGRESS TRACKING)
# ============================
@app.post("/summarize")
def summarize_video(
    req: SummarizeRequest,
    current_user: Optional[User] = Depends(get_current_user_optional),
    session: Session = Depends(get_session),
):
    """
    Summarize a YouTube video using Gemini (if available) or fallback to BART.
    """
    try:
        original_url = req.youtube_url.strip()
        
        # Extract video ID for progress tracking
        video_id = extract_video_id(original_url)
        update_progress(video_id, "starting", 0, "Initializing video analysis...")

        if not is_valid_youtube_url(original_url):
            raise HTTPException(status_code=400, detail="NOT_YOUTUBE_URL")
        
        # Normalize URL for consistent handling
        normalized_url = normalize_youtube_url(original_url)
        print(f"🌐 Original URL: {original_url}")
        print(f"🔗 Normalized URL: {normalized_url}")
        
        # Handle authentication status
        is_authenticated = current_user is not None
        user_id = current_user.id if current_user else None
        
        print(f"🔐 Request from {'authenticated user' if is_authenticated else 'unauthenticated user'}")
        
        # Check for existing video only if authenticated
        if is_authenticated:
            existing_video = check_existing_video(session, user_id, normalized_url)
            if existing_video:
                print(f"✅ Found existing video in history! ID: {existing_video.id}")
                update_progress(video_id, "complete", 100, "Found in history!")
                return {
                    "title": existing_video.title,
                    "duration_minutes": round(existing_video.duration_seconds / 60, 2) if existing_video.duration_seconds else 0,
                    "thumbnail": existing_video.thumbnail_url,
                    "video_id": existing_video.video_id or extract_video_id(normalized_url),
                    "summaries": {
                        "short": existing_video.short_summary or "No summary available",
                        "detailed": existing_video.detailed_summary or existing_video.short_summary or "No summary available",
                        "key_points": existing_video.key_points or [],
                        "key_points_with_timestamps": existing_video.key_points_with_timestamps or [],
                        "topics_covered": existing_video.topics_covered or [],
                        "recommendations": existing_video.recommendations or [existing_video.recommendation] if existing_video.recommendation else [],
                        "section_summaries": existing_video.section_summaries or [],
                    },
                    "ai_model_used": existing_video.ai_model_used or "Unknown",
                    "transcription_model": existing_video.transcription_model or "Whisper",
                    "processing_method": existing_video.processing_method or "standard",
                    "saved": True,
                    "saved_video_id": existing_video.id,
                    "status": "EXISTING_VIDEO",
                    "existing": True,
                    "authenticated": is_authenticated,
                }

        print("STAGE:DOWNLOADING")
        update_progress(video_id, "downloading", 5, "Validating YouTube URL...")
        
        update_progress(video_id, "downloading", 10, "Fetching video metadata...")
        
        metadata = get_video_metadata(normalized_url)
        update_progress(video_id, "downloading", 20, f"Found: {metadata.get('title', 'video')[:50]}...")

        duration_sec = metadata.get("duration")
        title = metadata.get("title", "Untitled Video")
        thumbnail = metadata.get("thumbnail")

        if not duration_sec:
            raise HTTPException(status_code=400, detail="DURATION_NOT_FOUND")

        duration_minutes = duration_sec / 60

        video_id = extract_video_id(normalized_url)
        
        # Get transcript
        update_progress(video_id, "transcribing", 30, "Extracting audio from video...")
        transcript_text = get_video_transcript(video_id)
        update_progress(video_id, "transcribing", 55, "Transcription complete! Processing text...")
        
        word_count = len(transcript_text.split())
        print(f"📝 Got transcript: {word_count} words, {duration_minutes:.1f} minutes")

        # ===== CHOOSE SUMMARIZER =====
        print("STAGE:SUMMARIZING")
        update_progress(video_id, "analyzing", 60, "Preparing content for AI analysis...")
        
        ai_results = None
        model_used = "Unknown"
        processing_method = "standard"
        
        # Try Gemini first if available
        if USE_GEMINI and gemini_summarizer:
            try:
                print("🤖" + "="*50)
                print("🤖 Using Gemini for high-quality summarization...")
                print("🤖" + "="*50)
                
                update_progress(video_id, "analyzing", 65, "Running Gemini AI analysis...")
                
                ai_results = gemini_summarizer.summarize_video(
                    transcript=transcript_text,
                    duration_minutes=duration_minutes,
                    detailed=True
                )
                
                if ai_results and ai_results.get("short_summary") and ai_results.get("short_summary") != "Summary not available":
                    model_used = "Gemini 1.5 Pro"
                    processing_method = ai_results.get("processing_method", "gemini_pro")
                    print("✅" + "="*50)
                    print("✅ Gemini summarization successful!")
                    print("✅" + "="*50)
                else:
                    print("⚠️ Gemini returned incomplete results, falling back...")
                    ai_results = None
                
            except Exception as e:
                print(f"⚠️ Gemini failed with error: {e}")
                print("🔄 Falling back to hierarchical summarizer...")
                ai_results = None
        
        # Fallback to hierarchical/BART if Gemini not available or failed
        if ai_results is None:
            if HIERARCHICAL_AVAILABLE and hierarchical_summarizer and chunker:
                print("📊" + "="*50)
                print("📊 Using hierarchical summarizer (BART-based)...")
                print("📊" + "="*50)
                
                update_progress(video_id, "analyzing", 65, "Chunking transcript for processing...")
                
                chunks = chunker.chunk_transcript(transcript_text)
                print(f"✅ Created {len(chunks)} chunks")
                
                update_progress(video_id, "analyzing", 70, f"Processing {len(chunks)} content segments...")
                
                ai_results = hierarchical_summarizer.generate_hierarchical_summaries(
                    chunks,
                    duration_minutes=duration_minutes
                )
                model_used = ai_results.get("ai_model_used", "BART (Hierarchical)")
                processing_method = ai_results.get("processing_method", "hierarchical")
                print("✅ Hierarchical summarization complete")
                
            else:
                print("⚠️" + "="*50)
                print("⚠️ Using dummy summarizer (no AI available)")
                print("⚠️" + "="*50)
                
                update_progress(video_id, "analyzing", 75, "Using basic text analysis...")
                
                if chunker:
                    chunks = chunker.chunk_transcript(transcript_text)
                else:
                    words = transcript_text.split()
                    chunk_size = 500
                    chunks = []
                    for i in range(0, len(words), chunk_size):
                        chunk_text = ' '.join(words[i:i+chunk_size])
                        chunks.append({'text': chunk_text, 'word_count': len(chunk_text.split())})
                
                ai_results = summarization_service.generate_hierarchical_summaries(
                    chunks,
                    duration_minutes=duration_minutes
                )
                model_used = "Dummy Fallback"
                processing_method = "dummy"
        
        update_progress(video_id, "analyzing", 80, "Extracting key insights...")
        
        if ai_results is None:
            ai_results = {
                "short_summary": "Unable to generate summary",
                "detailed_summary": "Processing failed",
                "key_points": [],
                "key_points_with_timestamps": [],
                "topics_covered": [],
                "recommendations": [],
                "section_summaries": [],
                "chunk_summaries": [],
                "processing_method": "error",
                "chunks_processed": 0
            }
        
        update_progress(video_id, "summarizing", 85, "Generating final summary...")
        
        saved_video_id = None
        
        # Save to database if requested AND authenticated
        if req.save_to_history and is_authenticated:
            try:
                update_progress(video_id, "saving", 90, "Saving to your history...")
                
                video_data = {
                    "owner_id": user_id,
                    "url": normalized_url,  # Save normalized URL
                    "video_id": video_id,
                    "title": title,
                    "duration_seconds": duration_sec,
                    "thumbnail_url": thumbnail,
                    "transcript_text": transcript_text,
                    
                    "short_summary": ai_results.get("short_summary", ""),
                    "detailed_summary": ai_results.get("detailed_summary", ai_results.get("short_summary", "")),
                    
                    "key_points": ai_results.get("key_points", []),
                    "key_points_with_timestamps": ai_results.get("key_points_with_timestamps", []),
                    
                    "topics_covered": ai_results.get("topics_covered", []),
                    "recommendation": ai_results.get("recommendations", ["No recommendation"])[0] if ai_results.get("recommendations") else "",
                    "recommendations": ai_results.get("recommendations", []),
                    
                    "section_summaries": ai_results.get("section_summaries", []),
                    "chunk_summaries": ai_results.get("chunk_summaries", []),
                    
                    "ai_model_used": model_used,
                    "processing_method": processing_method,
                    "chunks_processed": ai_results.get("chunks_processed", 1),
                    "transcription_model": "Whisper",
                }
                
                video = Video(**video_data)
                session.add(video)
                session.commit()
                session.refresh(video)
                saved_video_id = video.id
                print(f"💾 Video saved to history with ID: {saved_video_id}")
                update_progress(video_id, "complete", 98, "Summary saved to history!")
            except Exception as e:
                print(f"⚠️ Failed to save video to history: {e}")
                saved_video_id = None
        elif req.save_to_history and not is_authenticated:
            print("⚠️ Cannot save to history: User not authenticated")

        update_progress(video_id, "complete", 100, "Complete! Your summary is ready.")

        # Return results
        return {
            "title": title,
            "duration_minutes": round(duration_minutes, 2),
            "duration_seconds": duration_sec,
            "thumbnail": thumbnail,
            "transcript_length": word_count,
            "video_id": video_id,
            "summaries": {
                "short": ai_results.get("short_summary", "No summary available"),
                "detailed": ai_results.get("detailed_summary", ai_results.get("short_summary", "No summary available")),
                "key_points": ai_results.get("key_points", []),
                "key_points_with_timestamps": ai_results.get("key_points_with_timestamps", []),
                "topics_covered": ai_results.get("topics_covered", []),
                "recommendations": ai_results.get("recommendations", []),
                "section_summaries": ai_results.get("section_summaries", []),
                "value_summary": ai_results.get("value_summary", ""),
                "watch_decision": ai_results.get("watch_decision", ""),
            },
            "ai_model_used": model_used,
            "processing_method": processing_method,
            "chunks_processed": ai_results.get("chunks_processed", 1),
            "transcription_model": "Whisper",
            "saved": saved_video_id is not None,
            "saved_video_id": saved_video_id,
            "status": "SUMMARIZED",
            "existing": False,
            "authenticated": is_authenticated,
            "auth_message": "You are not logged in. Create an account to save videos to history." if not is_authenticated else None,
            "gemini_used": USE_GEMINI and model_used == "Gemini 1.5 Pro",
        }
    
    except HTTPException:
        raise
    except Exception as e:
        # On error, update progress with error
        try:
            video_id = extract_video_id(req.youtube_url.strip())
            update_progress(video_id, "error", 0, f"Error: {str(e)[:100]}")
        except:
            pass
        print(f"❌ Unexpected error in summarize endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ============================
# GET USER HISTORY
# ============================
@app.get("/history")
def get_user_history(
    current_user: User = Depends(get_current_user_required),
    session: Session = Depends(get_session),
):
    """Get all videos summarized by the current user with all data"""
    try:
        statement = select(Video).where(Video.owner_id == current_user.id).order_by(Video.added_at.desc())
        videos = session.exec(statement).all()
        
        history = []
        for video in videos:
            video_id_from_url = None
            try:
                video_id_from_url = extract_video_id(video.url)
            except:
                pass
                
            history.append({
                "id": video.id,
                "url": video.url,
                "video_id": video.video_id,
                "title": video.title,
                "duration_seconds": video.duration_seconds,
                "duration_minutes": round(video.duration_seconds / 60, 2) if video.duration_seconds else None,
                "thumbnail_url": video.thumbnail_url,
                "added_at": video.added_at.isoformat() if video.added_at else None,
                
                "short_summary": video.short_summary,
                "detailed_summary": video.detailed_summary,
                "key_points": video.key_points or [],
                "key_points_with_timestamps": video.key_points_with_timestamps or [],
                "topics_covered": video.topics_covered or [],
                "recommendation": video.recommendation,
                "recommendations": video.recommendations or [],
                "section_summaries": video.section_summaries or [],
                
                "ai_model_used": video.ai_model_used,
                "transcription_model": video.transcription_model,
                "processing_method": video.processing_method,
                "chunks_processed": video.chunks_processed,
                
                "video_id_from_url": video_id_from_url,
            })
        
        return {
            "user_id": current_user.id,
            "total_videos": len(history),
            "history": history
        }
    except Exception as e:
        print(f"❌ Error getting history: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching history: {str(e)}")

# ============================
# GET VIDEO DETAILS
# ============================
@app.get("/video/{video_id}")
def get_video_details(
    video_id: int,
    current_user: User = Depends(get_current_user_required),
    session: Session = Depends(get_session),
):
    """Get detailed information about a specific saved video"""
    try:
        statement = select(Video).where(
            (Video.id == video_id) & 
            (Video.owner_id == current_user.id)
        )
        video = session.exec(statement).first()
        
        if not video:
            raise HTTPException(status_code=404, detail="VIDEO_NOT_FOUND")
        
        video_id_from_url = None
        try:
            video_id_from_url = extract_video_id(video.url)
        except:
            pass
        
        return {
            "id": video.id,
            "url": video.url,
            "video_id": video.video_id,
            "title": video.title,
            "duration_seconds": video.duration_seconds,
            "duration_minutes": round(video.duration_seconds / 60, 2) if video.duration_seconds else None,
            "thumbnail_url": video.thumbnail_url,
            "added_at": video.added_at.isoformat() if video.added_at else None,
            
            "short_summary": video.short_summary,
            "detailed_summary": video.detailed_summary,
            "key_points": video.key_points or [],
            "key_points_with_timestamps": video.key_points_with_timestamps or [],
            "topics_covered": video.topics_covered or [],
            "recommendation": video.recommendation,
            "recommendations": video.recommendations or [],
            "section_summaries": video.section_summaries or [],
            
            "ai_model_used": video.ai_model_used,
            "transcription_model": video.transcription_model,
            "processing_method": video.processing_method,
            "chunks_processed": video.chunks_processed,
            
            "transcript_text": video.transcript_text,
            "video_id_from_url": video_id_from_url,
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error getting video details: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching video details: {str(e)}")

# ============================
# DELETE VIDEO
# ============================
@app.delete("/video/{video_id}")
def delete_video(
    video_id: int,
    current_user: User = Depends(get_current_user_required),
    session: Session = Depends(get_session),
):
    """Delete a saved video from history"""
    try:
        from sqlmodel import delete
        
        statement = select(Video).where(
            (Video.id == video_id) & 
            (Video.owner_id == current_user.id)
        )
        video = session.exec(statement).first()
        
        if not video:
            raise HTTPException(status_code=404, detail="VIDEO_NOT_FOUND")
        
        delete_statement = delete(Video).where(Video.id == video_id)
        session.exec(delete_statement)
        session.commit()
        
        return {
            "message": "Video deleted successfully",
            "deleted_video_id": video_id
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error deleting video: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting video: {str(e)}")

# ============================
# FEEDBACK ENDPOINTS
# ============================

@app.post("/feedback/{video_id}")
def submit_feedback(
    video_id: int,
    req: FeedbackRequest,
    current_user: User = Depends(get_current_user_required),
    session: Session = Depends(get_session),
):
    """Submit or update feedback for a video"""
    from models import VideoFeedback
    
    # Validate rating
    if req.rating < 1 or req.rating > 5:
        raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")
    
    # Check if video exists and belongs to user
    statement = select(Video).where(
        (Video.id == video_id) & (Video.owner_id == current_user.id)
    )
    video = session.exec(statement).first()
    if not video:
        raise HTTPException(status_code=404, detail="VIDEO_NOT_FOUND")
    
    # Check if feedback already exists
    statement = select(VideoFeedback).where(
        (VideoFeedback.user_id == current_user.id) & 
        (VideoFeedback.video_id == video_id)
    )
    existing = session.exec(statement).first()
    
    if existing:
        # Update existing feedback
        existing.rating = req.rating
        existing.comment = req.comment
        existing.updated_at = datetime.utcnow()
        session.commit()
        session.refresh(existing)
        return {
            "message": "Feedback updated successfully",
            "feedback": {
                "id": existing.id,
                "rating": existing.rating,
                "comment": existing.comment,
                "created_at": existing.created_at.isoformat(),
                "updated_at": existing.updated_at.isoformat()
            }
        }
    else:
        # Create new feedback
        feedback = VideoFeedback(
            user_id=current_user.id,
            video_id=video_id,
            rating=req.rating,
            comment=req.comment
        )
        session.add(feedback)
        session.commit()
        session.refresh(feedback)
        return {
            "message": "Feedback submitted successfully",
            "feedback": {
                "id": feedback.id,
                "rating": feedback.rating,
                "comment": feedback.comment,
                "created_at": feedback.created_at.isoformat()
            }
        }


@app.get("/feedback/{video_id}")
def get_feedback_status(
    video_id: int,
    current_user: User = Depends(get_current_user_required),
    session: Session = Depends(get_session),
):
    """Check if user has already given feedback for a video"""
    from models import VideoFeedback
    
    statement = select(VideoFeedback).where(
        (VideoFeedback.user_id == current_user.id) & 
        (VideoFeedback.video_id == video_id)
    )
    existing = session.exec(statement).first()
    
    if existing:
        return {
            "has_feedback": True,
            "rating": existing.rating,
            "comment": existing.comment,
            "created_at": existing.created_at.isoformat()
        }
    else:
        return {"has_feedback": False}


@app.get("/video/{video_id}/feedbacks")
def get_video_all_feedbacks(
    video_id: int,
    current_user: User = Depends(get_current_user_required),
    session: Session = Depends(get_session),
):
    """Get all feedbacks for a video (for the video owner)"""
    from models import VideoFeedback
    
    # Check if video exists and belongs to user
    statement = select(Video).where(
        (Video.id == video_id) & (Video.owner_id == current_user.id)
    )
    video = session.exec(statement).first()
    if not video:
        raise HTTPException(status_code=404, detail="VIDEO_NOT_FOUND")
    
    # Get all feedbacks for this video
    statement = select(VideoFeedback).where(VideoFeedback.video_id == video_id).order_by(VideoFeedback.created_at.desc())
    feedbacks = session.exec(statement).all()
    
    return {
        "video_id": video_id,
        "video_title": video.title,
        "total_feedbacks": len(feedbacks),
        "average_rating": video.average_rating,
        "feedbacks": [
            {
                "id": f.id,
                "rating": f.rating,
                "comment": f.comment,
                "user_name": f.user.name if f.user else None,
                "created_at": f.created_at.isoformat(),
                "updated_at": f.updated_at.isoformat() if f.updated_at else None
            }
            for f in feedbacks
        ]
    }


@app.delete("/feedback/{feedback_id}")
def delete_feedback(
    feedback_id: int,
    current_user: User = Depends(get_current_user_required),
    session: Session = Depends(get_session),
):
    """Delete a user's own feedback"""
    from models import VideoFeedback
    
    statement = select(VideoFeedback).where(
        (VideoFeedback.id == feedback_id) & 
        (VideoFeedback.user_id == current_user.id)
    )
    feedback = session.exec(statement).first()
    
    if not feedback:
        raise HTTPException(status_code=404, detail="FEEDBACK_NOT_FOUND")
    
    session.delete(feedback)
    session.commit()
    
    return {"message": "Feedback deleted successfully"}

# ============================
# ROOT
# ============================
@app.get("/")
def root():
    return {
        "message": "Insight Video API running",
        "status": "healthy",
        "ai_status": {
            "gemini_available": USE_GEMINI,
            "hierarchical_available": HIERARCHICAL_AVAILABLE,
            "active_model": "Gemini 1.5 Pro" if USE_GEMINI else "BART (Hierarchical)" if HIERARCHICAL_AVAILABLE else "Dummy"
        },
        "transcription_engine": "Whisper",
        "max_video_duration": "No limit (hierarchical processing)",
        "auth_info": {
            "endpoints": {
                "register": "/register (POST) - Create account",
                "login": "/token (POST) - Get JWT token",
                "summarize": "/summarize (POST) - Summarize video (works with or without auth)",
                "progress": "/progress/{video_id} (GET) - Get real-time progress",
                "history": "/history (GET) - Requires auth",
                "video": "/video/{id} (GET/DELETE) - Requires auth",
                "feedback": "/feedback/{video_id} (POST/GET) - Submit or check feedback"
            }
        },
        "setup_instructions": {
            "gemini": "Add GEMINI_API_KEY=your_key_here to .env file for better summaries"
        }
    }

# ============================
# HEALTH CHECK
# ============================
@app.get("/health")
def health_check():
    """Health check endpoint"""
    try:
        try:
            whisper.load_model("base")
            whisper_status = "ready"
        except Exception as e:
            whisper_status = f"error: {str(e)}"
            
        return {
            "status": "healthy",
            "whisper": whisper_status,
            "ai_service": {
                "gemini": "available" if USE_GEMINI else "unavailable",
                "hierarchical": "available" if HIERARCHICAL_AVAILABLE else "unavailable",
                "active": "gemini" if USE_GEMINI else "hierarchical" if HIERARCHICAL_AVAILABLE else "dummy"
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

# ============================
# TEST ENDPOINT
# ============================
@app.get("/test/ffmpeg")
def test_ffmpeg():
    """Test if FFmpeg is working"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            version_line = lines[0] if lines else "Unknown"
            
            return {
                "status": "working",
                "ffmpeg_version": version_line,
                "return_code": result.returncode
            }
        else:
            return {
                "status": "error",
                "message": f"FFmpeg returned code {result.returncode}",
                "stderr": result.stderr[:200]
            }
    except FileNotFoundError:
        return {
            "status": "not_found",
            "message": "FFmpeg not found in PATH"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }