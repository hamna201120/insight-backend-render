from sqlmodel import select, Session
from sqlalchemy.exc import IntegrityError
from typing import Optional, List

from models import User, Video
from auth import get_password_hash, verify_password


# ============================
# USER MANAGEMENT
# ============================

def create_user(session: Session, name: str, email: str, password: str) -> User:
    """
    Create a new user safely.
    Prevents duplicate emails using DB-level UNIQUE constraint.
    """
    password_hash = get_password_hash(password)
    user = User(name=name, email=email, password_hash=password_hash)

    try:
        session.add(user)
        session.commit()
        session.refresh(user)
        return user

    except IntegrityError:
        session.rollback()
        # Database UNIQUE(email) constraint was violated
        raise ValueError("EMAIL_EXISTS")


def authenticate_user(session: Session, email: str, password: str) -> Optional[User]:
    """
    Authenticate a user by email and password.
    Returns user object if valid, else None.
    """
    user = get_user_by_email(session, email)
    if not user:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user


def get_user_by_email(session: Session, email: str) -> Optional[User]:
    """Fetch a user by their email."""
    statement = select(User).where(User.email == email)
    return session.exec(statement).first()


def get_or_create_user(session: Session, name: str, email: str) -> User:
    """
    Legacy function.
    FIXED: No longer creates users without a password,
    instead raises an error.
    """
    user = get_user_by_email(session, email)
    if user:
        return user

    # Creating password-less accounts is insecure — block it.
    raise ValueError(
        "User does not exist. Use create_user() with a valid password instead."
    )


# ============================
# VIDEO MANAGEMENT (UPDATED)
# ============================

def add_video(
    session: Session,
    owner_id: int,
    url: str,
    title: Optional[str] = None,
    duration_seconds: Optional[int] = None,
    thumbnail_url: Optional[str] = None,
    transcript_text: Optional[str] = None,
    summary_text: Optional[str] = None,
    # NEW FIELDS FOR COMPATIBILITY
    video_id: Optional[str] = None,
    short_summary: Optional[str] = None,
    key_points: Optional[List[str]] = None,
    topics_covered: Optional[List[str]] = None,
    recommendation: Optional[str] = None,
    ai_model_used: Optional[str] = None,
    transcription_model: Optional[str] = None
) -> Video:
    """
    Add a new processed video for a user.
    UPDATED: Now includes all new fields from the Video model.
    """
    video = Video(
        owner_id=owner_id,
        url=url,
        video_id=video_id,
        title=title,
        duration_seconds=duration_seconds,
        thumbnail_url=thumbnail_url,
        transcript_text=transcript_text,
        # Old field for backward compatibility
        summary_text=summary_text,
        # New fields
        short_summary=short_summary,
        recommendation=recommendation,
        ai_model_used=ai_model_used,
        transcription_model=transcription_model
    )
    
    # Set list fields using property setters
    if key_points:
        video.key_points = key_points
    if topics_covered:
        video.topics_covered = topics_covered
    
    session.add(video)
    session.commit()
    session.refresh(video)
    return video


def get_user_videos(session: Session, owner_id: int) -> List[Video]:
    """Return all videos for a specific user, ordered by newest first."""
    statement = (
        select(Video)
        .where(Video.owner_id == owner_id)
        .order_by(Video.added_at.desc())
    )
    return session.exec(statement).all()


# ============================
# NEW: DUPLICATE CHECKING FUNCTION
# ============================

def check_duplicate_video(session: Session, owner_id: int, url: str) -> Optional[Video]:
    """
    Check if a video with the same URL already exists for this user.
    Returns the existing video if found, None otherwise.
    """
    statement = select(Video).where(
        (Video.owner_id == owner_id) & 
        (Video.url == url)
    )
    return session.exec(statement).first()


def check_duplicate_video_by_video_id(session: Session, owner_id: int, video_id: str) -> Optional[Video]:
    """
    Check if a video with the same YouTube video ID already exists for this user.
    Returns the existing video if found, None otherwise.
    """
    statement = select(Video).where(
        (Video.owner_id == owner_id) & 
        (Video.video_id == video_id)
    )
    return session.exec(statement).first()


def get_video_by_id(session: Session, video_id: int, owner_id: int) -> Optional[Video]:
    """
    Get a specific video by ID, ensuring it belongs to the owner.
    """
    statement = select(Video).where(
        (Video.id == video_id) & 
        (Video.owner_id == owner_id)
    )
    return session.exec(statement).first()