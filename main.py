# ---------- COMPLETE main.py ----------
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from sqlmodel import Session, select
from datetime import timedelta, datetime
from jose import JWTError, jwt
import re
import subprocess
import whisper
import tempfile
import os
import gc
import requests
from pathlib import Path
from typing import Optional, List, Dict
from dotenv import load_dotenv
from pytube import YouTube

# ============================================
# CONDITIONAL IMPORT FOR BROWSER_COOKIE3
# ============================================
try:
    import browser_cookie3
    BROWSER_COOKIE_AVAILABLE = True
except ImportError:
    BROWSER_COOKIE_AVAILABLE = False

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
# PROGRESS TRACKING
# ============================
progress_tracker: Dict[str, Dict] = {}

def update_progress(video_id: str, stage: str, progress: int, detail: str):
    progress_tracker[video_id] = {
        "stage": stage,
        "progress": progress,
        "detail": detail,
        "timestamp": datetime.now().isoformat()
    }
    print(f"📊 PROGRESS: {video_id} - {stage} - {progress}% - {detail}")

def get_progress(video_id: str) -> Dict:
    return progress_tracker.get(video_id, {
        "stage": "starting",
        "progress": 0,
        "detail": "Starting video analysis..."
    })

# ============================
# COOKIES
# ============================
def get_youtube_cookies():
    import base64
    cookies_path = None
    cookies_b64 = os.environ.get('YOUTUBE_COOKIES_B64')
    if cookies_b64:
        try:
            cookies_content = base64.b64decode(cookies_b64).decode('utf-8')
            if '# Netscape HTTP Cookie File' in cookies_content:
                cookies_path = os.path.join(tempfile.gettempdir(), 'youtube_cookies_env.txt')
                with open(cookies_path, 'w') as f:
                    f.write(cookies_content)
                return cookies_path
        except:
            pass
    if BROWSER_COOKIE_AVAILABLE:
        for browser_name, browser_func in [('chrome', browser_cookie3.chrome), ('firefox', browser_cookie3.firefox)]:
            try:
                cookies = browser_func(domain_name='.youtube.com')
                if cookies:
                    cookies_path = os.path.join(tempfile.gettempdir(), f'youtube_cookies_{browser_name}.txt')
                    with open(cookies_path, 'w') as f:
                        f.write('# Netscape HTTP Cookie File\n')
                        for cookie in cookies:
                            if 'youtube.com' in cookie.domain:
                                expires = int(cookie.expires) if cookie.expires else 0
                                secure = 'TRUE' if cookie.secure else 'FALSE'
                                f.write(f"{cookie.domain}\tTRUE\t{cookie.path}\t{secure}\t{expires}\t{cookie.name}\t{cookie.value}\n")
                    return cookies_path
            except:
                continue
    return None

# ============================
# AUDIO DOWNLOAD - PYTUBE PRIMARY
# ============================
def download_audio_pytube(video_url: str, tmpdir: str) -> Optional[str]:
    try:
        yt = YouTube(video_url)
        audio_stream = yt.streams.filter(only_audio=True).first()
        if not audio_stream:
            return None
        out_file = audio_stream.download(output_path=tmpdir, filename="audio")
        base, ext = os.path.splitext(out_file)
        if ext.lower() not in ['.m4a', '.mp3', '.webm']:
            new_name = base + '.m4a'
            os.rename(out_file, new_name)
            out_file = new_name
        return out_file
    except Exception as e:
        print(f"⚠️ pytube failed: {e}")
        return None

# ============================
# AUDIO DOWNLOAD - FALLBACK YT-DLP
# ============================
def download_audio_ytdlp_fallback(video_url: str, tmpdir: str) -> Optional[str]:
    cookies_path = get_youtube_cookies()
    cmd = [
        'yt-dlp',
        '-f', 'bestaudio',
        '--extractor-args', 'youtube:skip=dash,hls',
        '--no-warnings',
        '--quiet',
        '--no-playlist',
        '-o', os.path.join(tmpdir, 'audio.%(ext)s'),
        video_url
    ]
    if cookies_path:
        cmd.extend(['--cookies', cookies_path])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return None
        files = list(Path(tmpdir).glob("*"))
        audio_extensions = {'.m4a', '.mp3', '.webm', '.opus', '.aac', '.wav'}
        audio_files = [f for f in files if f.suffix.lower() in audio_extensions]
        if not audio_files:
            audio_files = [f for f in files if f.is_file() and f.stat().st_size > 1000]
        if audio_files:
            return str(audio_files[0])
        return None
    except:
        return None

def download_audio(video_id: str) -> str:
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = download_audio_pytube(video_url, tmpdir)
        if audio_path:
            return audio_path
        audio_path = download_audio_ytdlp_fallback(video_url, tmpdir)
        if audio_path:
            return audio_path
        raise Exception("All audio download methods failed")

# ============================
# GEMINI & HIERARCHICAL (unchanged)
# ============================
try:
    from gemini_summarizer import GeminiSummarizer
    from api_key_rotator import key_rotator
    gemini_summarizer = GeminiSummarizer()
    USE_GEMINI = True
except Exception as e:
    print(f"⚠️ Gemini init failed: {e}")
    USE_GEMINI = False
    gemini_summarizer = None

try:
    from hierarchical_summarizer import HierarchicalSummarizer
    from smart_chunker import SmartChunker
    HIERARCHICAL_AVAILABLE = True
    hierarchical_summarizer = HierarchicalSummarizer()
    chunker = SmartChunker(max_chunk_size=800, min_chunk_size=250)
except ImportError:
    HIERARCHICAL_AVAILABLE = False
    hierarchical_summarizer = None
    chunker = None

app = FastAPI(title="Insight Video Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

@app.on_event("startup")
def on_startup():
    init_db()
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
        print("✅ FFmpeg available")
    except:
        print("⚠️ FFmpeg not found")
    try:
        cookies = get_youtube_cookies()
        if cookies:
            print("✅ Cookies available")
    except:
        pass

# ============================
# URL HELPERS
# ============================
def extract_video_id(url: str) -> str:
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    if "watch?v=" in url:
        return url.split("watch?v=")[1].split("&")[0]
    if "shorts/" in url:
        return url.split("shorts/")[1].split("?")[0]
    raise HTTPException(status_code=400, detail="INVALID_YOUTUBE_URL")

def normalize_youtube_url(url: str) -> str:
    return f"https://www.youtube.com/watch?v={extract_video_id(url)}"

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

class FeedbackRequest(BaseModel):
    rating: int
    comment: Optional[str] = None

# ============================
# AUTH
# ============================
async def get_current_user_optional(
    token: Optional[str] = Depends(oauth2_scheme),
    session: Session = Depends(get_session),
) -> Optional[User]:
    if not token:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if not email:
            return None
    except:
        return None
    return crud.get_user_by_email(session, email)

async def get_current_user_required(
    token: str = Depends(oauth2_scheme),
    session: Session = Depends(get_session),
) -> User:
    if not token:
        raise HTTPException(status_code=401, detail="MISSING_TOKEN")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if not email:
            raise HTTPException(status_code=401, detail="INVALID_TOKEN")
    except:
        raise HTTPException(status_code=401, detail="INVALID_TOKEN")
    user = crud.get_user_by_email(session, email)
    if not user:
        raise HTTPException(status_code=401, detail="USER_NOT_FOUND")
    return user

# ============================
# REGISTER / LOGIN / ME
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

@app.get("/me", response_model=UserResponse)
def me(current_user: User = Depends(get_current_user_required)):
    return UserResponse(
        id=current_user.id,
        name=current_user.name,
        email=current_user.email,
    )

# ============================
# PROGRESS
# ============================
@app.get("/progress/{video_id}")
def get_video_progress(video_id: str):
    return get_progress(video_id)

# ============================
# METADATA & TRANSCRIPT
# ============================
YOUTUBE_REGEX = re.compile(
    r"(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)[\w\-]+"
)

def is_valid_youtube_url(url: str) -> bool:
    return bool(YOUTUBE_REGEX.match(url))

def get_video_metadata(url: str) -> dict:
    try:
        cookies_path = get_youtube_cookies()
        import yt_dlp
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,
            "nocheckcertificate": True,
            "no_color": True,
            "cookiefile": cookies_path if cookies_path else None,
            "sleep_interval": 3,
            "max_sleep_interval": 6,
            "extractor_args": {"youtube": {"player_client": ["mweb"]}}
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False, process=False)
            return {
                "title": info.get('title', 'Unknown Title'),
                "duration": info.get('duration', 0),
                "thumbnail": info.get('thumbnail', ''),
                "uploader": info.get('uploader', 'Unknown'),
                "view_count": info.get('view_count', 0),
                "upload_date": info.get('upload_date', None),
            }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"VIDEO_METADATA_FAILED: {str(e)}")

def get_video_transcript(video_id: str) -> str:
    try:
        # Try YouTube transcript API
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            # Use the correct method
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            transcript = None
            for t in transcript_list:
                if t.language_code.startswith('en'):
                    transcript = t
                    break
            if not transcript:
                transcript = next(iter(transcript_list))
            data = transcript.fetch()
            text = " ".join([item['text'] for item in data])
            print("✅ Transcript fetched")
            return text
        except Exception as e:
            print(f"⚠️ Transcript API failed: {e}")
            # Try older method
            try:
                from youtube_transcript_api import YouTubeTranscriptApi
                data = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
                if data:
                    text = " ".join([item['text'] for item in data])
                    print("✅ Transcript fetched (old method)")
                    return text
            except:
                pass

        # Fallback: download audio and transcribe
        print("🔊 Downloading audio for transcription...")
        audio_path = download_audio(video_id)
        if not audio_path or not os.path.exists(audio_path):
            raise Exception("Failed to download audio")
        print(f"🎵 Audio: {os.path.basename(audio_path)}")
        
        # Load Whisper
        try:
            model = whisper.load_model("base")
        except:
            model = whisper.load_model("tiny")
        result = model.transcribe(audio_path, task='translate', fp16=False, verbose=False)
        transcript_text = result["text"].strip()
        print(f"✅ Transcription: {len(transcript_text)} chars")
        del model
        gc.collect()
        return transcript_text
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ get_video_transcript error: {e}")
        raise HTTPException(status_code=400, detail=f"Could not process video: {str(e)[:100]}")

# ============================
# DUPLICATE CHECK
# ============================
def check_existing_video(session: Session, user_id: int, url: str) -> Optional[Video]:
    normalized_url = normalize_youtube_url(url)
    statement = select(Video).where(
        (Video.owner_id == user_id) & 
        (Video.url == normalized_url)
    )
    return session.exec(statement).first()

# ============================
# SUMMARIZE ENDPOINT (unchanged)
# ============================
@app.post("/summarize")
def summarize_video(
    req: SummarizeRequest,
    current_user: Optional[User] = Depends(get_current_user_optional),
    session: Session = Depends(get_session),
):
    try:
        original_url = req.youtube_url.strip()
        video_id = extract_video_id(original_url)
        update_progress(video_id, "starting", 0, "Initializing...")

        if not is_valid_youtube_url(original_url):
            raise HTTPException(status_code=400, detail="NOT_YOUTUBE_URL")
        
        normalized_url = normalize_youtube_url(original_url)
        print(f"🌐 Original URL: {original_url}")
        print(f"🔗 Normalized: {normalized_url}")
        
        is_authenticated = current_user is not None
        user_id = current_user.id if current_user else None
        
        if is_authenticated:
            existing_video = check_existing_video(session, user_id, normalized_url)
            if existing_video:
                print(f"✅ Found existing video! ID: {existing_video.id}")
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
        update_progress(video_id, "downloading", 5, "Validating URL...")
        update_progress(video_id, "downloading", 10, "Fetching metadata...")
        
        metadata = get_video_metadata(normalized_url)
        update_progress(video_id, "downloading", 20, f"Found: {metadata.get('title', 'video')[:50]}...")

        duration_sec = metadata.get("duration")
        title = metadata.get("title", "Untitled Video")
        thumbnail = metadata.get("thumbnail")

        if not duration_sec:
            raise HTTPException(status_code=400, detail="DURATION_NOT_FOUND")

        duration_minutes = duration_sec / 60
        video_id = extract_video_id(normalized_url)
        
        update_progress(video_id, "transcribing", 30, "Extracting audio...")
        transcript_text = get_video_transcript(video_id)
        update_progress(video_id, "transcribing", 55, "Transcription complete!")
        
        word_count = len(transcript_text.split())
        print(f"📝 Got transcript: {word_count} words, {duration_minutes:.1f} minutes")

        print("STAGE:SUMMARIZING")
        update_progress(video_id, "analyzing", 60, "Preparing for AI analysis...")
        
        ai_results = None
        model_used = "Unknown"
        processing_method = "standard"
        
        if USE_GEMINI and gemini_summarizer:
            try:
                print("🤖 Using Gemini...")
                update_progress(video_id, "analyzing", 65, "Running Gemini AI...")
                ai_results = gemini_summarizer.summarize_video(
                    transcript=transcript_text,
                    duration_minutes=duration_minutes,
                    detailed=True
                )
                if ai_results and ai_results.get("short_summary") and ai_results.get("short_summary") != "Summary not available":
                    model_used = "Gemini 1.5 Pro"
                    processing_method = ai_results.get("processing_method", "gemini_pro")
                    print("✅ Gemini successful!")
                else:
                    ai_results = None
            except Exception as e:
                print(f"⚠️ Gemini failed: {e}")
                ai_results = None
        
        if ai_results is None:
            if HIERARCHICAL_AVAILABLE and hierarchical_summarizer and chunker:
                print("📊 Using hierarchical summarizer...")
                update_progress(video_id, "analyzing", 65, "Chunking transcript...")
                chunks = chunker.chunk_transcript(transcript_text)
                print(f"✅ Created {len(chunks)} chunks")
                update_progress(video_id, "analyzing", 70, f"Processing {len(chunks)} chunks...")
                ai_results = hierarchical_summarizer.generate_hierarchical_summaries(chunks, duration_minutes=duration_minutes)
                model_used = ai_results.get("ai_model_used", "BART (Hierarchical)")
                processing_method = ai_results.get("processing_method", "hierarchical")
                print("✅ Hierarchical complete")
            else:
                print("⚠️ Using dummy summarizer")
                update_progress(video_id, "analyzing", 75, "Using basic analysis...")
                words = transcript_text.split()
                chunk_size = 500
                chunks = []
                for i in range(0, len(words), chunk_size):
                    chunk_text = ' '.join(words[i:i+chunk_size])
                    chunks.append({'text': chunk_text, 'word_count': len(chunk_text.split())})
                
                class DummySummarizer:
                    def generate_hierarchical_summaries(self, chunks, duration_minutes=0):
                        text = ' '.join([c.get('text', '') for c in chunks if c.get('text')])
                        words = text.split()[:100]
                        sample = " ".join(words)
                        return {
                            "short_summary": f"Video summary: {sample[:200]}",
                            "detailed_summary": f"Detailed summary: {sample[:500]}",
                            "key_points": ["Key point 1", "Key point 2"],
                            "key_points_with_timestamps": [],
                            "topics_covered": ["General"],
                            "recommendations": ["Watch the full video"],
                            "section_summaries": [],
                            "chunk_summaries": [],
                            "ai_model_used": "Dummy",
                            "processing_method": "dummy",
                            "chunks_processed": len(chunks)
                        }
                
                dummy = DummySummarizer()
                ai_results = dummy.generate_hierarchical_summaries(chunks, duration_minutes=duration_minutes)
                model_used = "Dummy Fallback"
                processing_method = "dummy"
        
        update_progress(video_id, "analyzing", 80, "Extracting insights...")
        
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
        
        if req.save_to_history and is_authenticated:
            try:
                update_progress(video_id, "saving", 90, "Saving to history...")
                video_data = {
                    "owner_id": user_id,
                    "url": normalized_url,
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
                print(f"💾 Video saved! ID: {saved_video_id}")
                update_progress(video_id, "complete", 98, "Saved to history!")
            except Exception as e:
                print(f"⚠️ Failed to save: {e}")
                saved_video_id = None

        update_progress(video_id, "complete", 100, "Complete!")

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
        try:
            video_id = extract_video_id(req.youtube_url.strip())
            update_progress(video_id, "error", 0, f"Error: {str(e)[:100]}")
        except:
            pass
        print(f"❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ============================
# HISTORY, VIDEO DETAILS, DELETE, FEEDBACK (unchanged)
# ============================
# ... (copy the exact endpoints from previous versions)
# Since this is the final answer, I'll include them below.

@app.get("/history")
def get_user_history(
    current_user: User = Depends(get_current_user_required),
    session: Session = Depends(get_session),
):
    try:
        statement = select(Video).where(Video.owner_id == current_user.id).order_by(Video.added_at.desc())
        videos = session.exec(statement).all()
        history = []
        for video in videos:
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
            })
        return {
            "user_id": current_user.id,
            "total_videos": len(history),
            "history": history
        }
    except Exception as e:
        print(f"❌ Error getting history: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching history: {str(e)}")

@app.get("/video/{video_id}")
def get_video_details(
    video_id: int,
    current_user: User = Depends(get_current_user_required),
    session: Session = Depends(get_session),
):
    try:
        statement = select(Video).where(
            (Video.id == video_id) & 
            (Video.owner_id == current_user.id)
        )
        video = session.exec(statement).first()
        if not video:
            raise HTTPException(status_code=404, detail="VIDEO_NOT_FOUND")
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
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error getting video details: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching video details: {str(e)}")

@app.delete("/video/{video_id}")
def delete_video(
    video_id: int,
    current_user: User = Depends(get_current_user_required),
    session: Session = Depends(get_session),
):
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
        return {"message": "Video deleted successfully", "deleted_video_id": video_id}
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error deleting video: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting video: {str(e)}")

# ============================
# FEEDBACK ENDPOINTS (unchanged)
# ============================
from models import VideoFeedback

@app.post("/feedback/{video_id}")
def submit_feedback(
    video_id: int,
    req: FeedbackRequest,
    current_user: User = Depends(get_current_user_required),
    session: Session = Depends(get_session),
):
    if req.rating < 1 or req.rating > 5:
        raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")
    statement = select(Video).where(
        (Video.id == video_id) & (Video.owner_id == current_user.id)
    )
    video = session.exec(statement).first()
    if not video:
        raise HTTPException(status_code=404, detail="VIDEO_NOT_FOUND")
    statement = select(VideoFeedback).where(
        (VideoFeedback.user_id == current_user.id) & 
        (VideoFeedback.video_id == video_id)
    )
    existing = session.exec(statement).first()
    if existing:
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
    return {"has_feedback": False}

@app.get("/video/{video_id}/feedbacks")
def get_video_all_feedbacks(
    video_id: int,
    current_user: User = Depends(get_current_user_required),
    session: Session = Depends(get_session),
):
    statement = select(Video).where(
        (Video.id == video_id) & (Video.owner_id == current_user.id)
    )
    video = session.exec(statement).first()
    if not video:
        raise HTTPException(status_code=404, detail="VIDEO_NOT_FOUND")
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
# DEBUG, ROOT, HEALTH
# ============================
@app.get("/debug")
def debug():
    import yt_dlp
    return {
        "yt_dlp_version": yt_dlp.version.__version__,
        "cookies_present": bool(os.environ.get("YOUTUBE_COOKIES_B64"))
    }

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
    }

@app.get("/health")
def health_check():
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")
