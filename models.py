# models.py - FIXED VERSION WITH FEEDBACK
from sqlmodel import SQLModel, Field, Relationship, Session, select
from typing import Optional, List, Any, Dict
from datetime import datetime
from sqlmodel import SQLModel, Field, Relationship
import json

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    email: str = Field(index=True, unique=True)
    password_hash: str = Field(default="")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    videos: List["Video"] = Relationship(back_populates="owner")
    history: List["History"] = Relationship(back_populates="user")
    feedbacks: List["VideoFeedback"] = Relationship(back_populates="user")

class Video(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Video identification
    url: str = Field(index=True)
    video_id: Optional[str] = Field(default=None, index=True)
    
    # Video metadata
    title: Optional[str] = None
    duration_seconds: Optional[int] = None
    thumbnail_url: Optional[str] = None
    added_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Summary data - BASIC
    short_summary: Optional[str] = None
    
    # NEW: Detailed summary for long videos
    detailed_summary: Optional[str] = None
    
    # Structured data - BASIC
    key_points_json: Optional[str] = None
    topics_covered_json: Optional[str] = None
    recommendation: Optional[str] = None
    
    # NEW: Enhanced structured data for long videos
    key_points_with_timestamps_json: Optional[str] = None
    recommendations_json: Optional[str] = None  # Multiple recommendations
    
    # NEW: Section and chunk summaries for hierarchical processing
    section_summaries_json: Optional[str] = None
    chunk_summaries_json: Optional[str] = None
    
    # AI/Processing info
    ai_model_used: Optional[str] = None
    transcription_model: Optional[str] = None
    
    # NEW: Processing method (standard, hierarchical)
    processing_method: Optional[str] = Field(default="standard")
    
    # NEW: Processing metrics
    chunks_processed: Optional[int] = Field(default=1)
    total_word_count: Optional[int] = None
    
    # User relationship
    owner_id: Optional[int] = Field(default=None, foreign_key="user.id", index=True)
    owner: Optional[User] = Relationship(back_populates="videos")
    
    # Raw transcript
    transcript_text: Optional[str] = None
    
    # History relationship
    history: List["History"] = Relationship(back_populates="video")
    
    # Feedback relationship
    feedbacks: List["VideoFeedback"] = Relationship(back_populates="video")
    
    # ==================== FIXED: INIT METHOD TO HANDLE DATA PROPERLY ====================
    def __init__(self, **data):
        """Override init to handle JSON fields properly"""
        # Extract JSON fields that need special handling
        key_points = data.pop('key_points', None) if 'key_points' in data else None
        topics_covered = data.pop('topics_covered', None) if 'topics_covered' in data else None
        key_points_with_timestamps = data.pop('key_points_with_timestamps', None) if 'key_points_with_timestamps' in data else None
        recommendations = data.pop('recommendations', None) if 'recommendations' in data else None
        section_summaries = data.pop('section_summaries', None) if 'section_summaries' in data else None
        chunk_summaries = data.pop('chunk_summaries', None) if 'chunk_summaries' in data else None
        
        # Initialize with remaining data
        super().__init__(**data)
        
        # Now set the JSON fields using the setters
        if key_points is not None:
            self.key_points = key_points
        if topics_covered is not None:
            self.topics_covered = topics_covered
        if key_points_with_timestamps is not None:
            self.key_points_with_timestamps = key_points_with_timestamps
        if recommendations is not None:
            self.recommendations = recommendations
        if section_summaries is not None:
            self.section_summaries = section_summaries
        if chunk_summaries is not None:
            self.chunk_summaries = chunk_summaries
    
    # ==================== PROPERTIES FOR JSON FIELDS ====================
    
    @property
    def key_points(self) -> List[str]:
        """Get key points as list"""
        if self.key_points_json:
            try:
                return json.loads(self.key_points_json)
            except:
                return []
        return []
    
    @key_points.setter
    def key_points(self, value: List[str]):
        self.key_points_json = json.dumps(value) if value is not None else None
    
    @property
    def topics_covered(self) -> List[str]:
        """Get topics covered as list"""
        if self.topics_covered_json:
            try:
                return json.loads(self.topics_covered_json)
            except:
                return []
        return []
    
    @topics_covered.setter
    def topics_covered(self, value: List[str]):
        self.topics_covered_json = json.dumps(value) if value is not None else None
    
    @property
    def key_points_with_timestamps(self) -> List[Dict]:
        """Get key points with timestamps as list of dicts"""
        if self.key_points_with_timestamps_json:
            try:
                return json.loads(self.key_points_with_timestamps_json)
            except:
                return []
        return []
    
    @key_points_with_timestamps.setter
    def key_points_with_timestamps(self, value: List[Dict]):
        self.key_points_with_timestamps_json = json.dumps(value) if value is not None else None
    
    @property
    def recommendations(self) -> List[str]:
        """Get multiple recommendations as list"""
        if self.recommendations_json:
            try:
                return json.loads(self.recommendations_json)
            except:
                return []
        # Fall back to single recommendation
        if self.recommendation:
            return [self.recommendation]
        return []
    
    @recommendations.setter
    def recommendations(self, value: List[str]):
        self.recommendations_json = json.dumps(value) if value is not None else None
        # Also set single recommendation for backward compatibility
        if value and len(value) > 0:
            self.recommendation = value[0]
    
    @property
    def section_summaries(self) -> List[Dict]:
        """Get section summaries as list of dicts"""
        if self.section_summaries_json:
            try:
                return json.loads(self.section_summaries_json)
            except:
                return []
        return []
    
    @section_summaries.setter
    def section_summaries(self, value: List[Dict]):
        self.section_summaries_json = json.dumps(value) if value is not None else None
    
    @property
    def chunk_summaries(self) -> List[Dict]:
        """Get chunk summaries as list of dicts"""
        if self.chunk_summaries_json:
            try:
                return json.loads(self.chunk_summaries_json)
            except:
                return []
        return []
    
    @chunk_summaries.setter
    def chunk_summaries(self, value: List[Dict]):
        self.chunk_summaries_json = json.dumps(value) if value is not None else None
    
    # ==================== HELPER PROPERTIES ====================
    
    @property
    def summary_preview(self) -> str:
        """Get first 100 chars of summary for preview"""
        if self.short_summary:
            if len(self.short_summary) > 100:
                return self.short_summary[:100] + "..."
            return self.short_summary
        return "No summary available"
    
    @property
    def duration_minutes(self) -> Optional[float]:
        """Get duration in minutes"""
        if self.duration_seconds:
            return round(self.duration_seconds / 60, 2)
        return None
    
    @property
    def has_timestamps(self) -> bool:
        """Check if video has timestamped key points"""
        return bool(self.key_points_with_timestamps)
    
    @property
    def is_long_video(self) -> bool:
        """Check if this was processed as a long video"""
        return self.processing_method == "hierarchical"
    
    @property
    def summary_stats(self) -> Dict:
        """Get summary statistics about the processing"""
        return {
            "processing_method": self.processing_method,
            "chunks_processed": self.chunks_processed or 1,
            "word_count": self.total_word_count or (len(self.transcript_text.split()) if self.transcript_text else 0),
            "duration_minutes": self.duration_minutes,
            "key_points_count": len(self.key_points),
            "topics_count": len(self.topics_covered),
            "has_timestamps": self.has_timestamps,
            "has_sections": len(self.section_summaries) > 0
        }
    
    @property
    def average_rating(self) -> Optional[float]:
        """Get average rating from all feedbacks"""
        if not self.feedbacks:
            return None
        ratings = [f.rating for f in self.feedbacks]
        return round(sum(ratings) / len(ratings), 1)
    
    @property
    def rating_count(self) -> int:
        """Get total number of ratings"""
        return len(self.feedbacks)
    
    # ==================== CONVERSION METHODS ====================
    
    def to_dict(self, include_transcript: bool = False) -> Dict:
        """Convert video to dictionary for API responses"""
        data = {
            "id": self.id,
            "url": self.url,
            "video_id": self.video_id,
            "title": self.title,
            "duration_seconds": self.duration_seconds,
            "duration_minutes": self.duration_minutes,
            "thumbnail_url": self.thumbnail_url,
            "added_at": self.added_at.isoformat() if self.added_at else None,
            
            # Summary data
            "short_summary": self.short_summary,
            "detailed_summary": self.detailed_summary or self.short_summary,
            "key_points": self.key_points,
            "topics_covered": self.topics_covered,
            "recommendation": self.recommendation,
            "recommendations": self.recommendations,
            
            # Enhanced data
            "key_points_with_timestamps": self.key_points_with_timestamps,
            "section_summaries": self.section_summaries,
            
            # Processing info
            "ai_model_used": self.ai_model_used,
            "transcription_model": self.transcription_model,
            "processing_method": self.processing_method,
            "chunks_processed": self.chunks_processed,
            "stats": self.summary_stats,
            
            # Rating info
            "average_rating": self.average_rating,
            "rating_count": self.rating_count,
        }
        
        if include_transcript and self.transcript_text:
            data["transcript_text"] = self.transcript_text
            
        return data


# History Model for tracking user actions
class History(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: Optional[int] = Field(default=None, foreign_key="user.id", index=True)
    video_id: Optional[int] = Field(default=None, foreign_key="video.id", index=True)
    action: str  # e.g., "added_video", "deleted_video"
    details: Optional[str] = None  # any extra JSON/text info
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationships
    user: Optional[User] = Relationship(back_populates="history")
    video: Optional[Video] = Relationship(back_populates="history")
    
    # ==================== HELPER METHODS ====================
    
    @property
    def details_dict(self) -> Optional[Dict]:
        """Parse details JSON if present"""
        if self.details:
            try:
                return json.loads(self.details)
            except:
                return None
        return None
    
    @details_dict.setter
    def details_dict(self, value: Dict):
        """Set details from dictionary"""
        self.details = json.dumps(value) if value else None
    
    def to_dict(self) -> Dict:
        """Convert history to dictionary for API responses"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "video_id": self.video_id,
            "action": self.action,
            "details": self.details_dict,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "video_title": self.video.title if self.video else None,
            "video_thumbnail": self.video.thumbnail_url if self.video else None,
        }


# VideoFeedback Model for storing user ratings and feedback
class VideoFeedback(SQLModel, table=True):
    """Store user ratings and feedback for videos"""
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    video_id: int = Field(foreign_key="video.id", index=True)
    rating: int  # 1-5 stars
    comment: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    
    # Relationships
    user: Optional[User] = Relationship(back_populates="feedbacks")
    video: Optional[Video] = Relationship(back_populates="feedbacks")
    
    # ==================== HELPER METHODS ====================
    
    def to_dict(self) -> Dict:
        """Convert feedback to dictionary for API responses"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "video_id": self.video_id,
            "rating": self.rating,
            "comment": self.comment,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "user_name": self.user.name if self.user else None,
        }
    
    def update_rating(self, new_rating: int, new_comment: Optional[str] = None) -> None:
        """Update rating and optionally comment, updating the timestamp"""
        self.rating = new_rating
        if new_comment is not None:
            self.comment = new_comment
        self.updated_at = datetime.utcnow()


# ==================== HELPER FUNCTIONS ====================

def delete_history(session: Session, history_id: int) -> bool:
    """Delete a history record by ID."""
    item = session.get(History, history_id)  # SQLModel way to get by PK
    if item:
        session.delete(item)
        session.commit()
        return True
    return False


def create_history_entry(
    session: Session, 
    user_id: int, 
    video_id: Optional[int], 
    action: str, 
    details: Optional[Dict] = None
) -> History:
    """
    Create a new history entry.
    
    Args:
        session: SQLModel session
        user_id: ID of the user
        video_id: ID of the video (can be None for actions not related to a specific video)
        action: Action string (e.g., "added_video", "deleted_video")
        details: Optional dictionary with additional details
        
    Returns:
        Created History object
    """
    history = History(
        user_id=user_id,
        video_id=video_id,
        action=action,
        details=json.dumps(details) if details else None
    )
    session.add(history)
    session.commit()
    session.refresh(history)
    return history


def get_user_history(
    session: Session, 
    user_id: int, 
    limit: int = 50, 
    offset: int = 0
) -> List[History]:
    """
    Get history entries for a user.
    
    Args:
        session: SQLModel session
        user_id: ID of the user
        limit: Maximum number of entries to return
        offset: Offset for pagination
        
    Returns:
        List of History objects
    """
    from sqlmodel import select
    
    statement = (
        select(History)
        .where(History.user_id == user_id)
        .order_by(History.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    results = session.exec(statement)
    return results.all()


def get_video_history(
    session: Session, 
    video_id: int, 
    limit: int = 50, 
    offset: int = 0
) -> List[History]:
    """
    Get history entries for a video.
    
    Args:
        session: SQLModel session
        video_id: ID of the video
        limit: Maximum number of entries to return
        offset: Offset for pagination
        
    Returns:
        List of History objects
    """
    from sqlmodel import select
    
    statement = (
        select(History)
        .where(History.video_id == video_id)
        .order_by(History.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    results = session.exec(statement)
    return results.all()


# ==================== FEEDBACK HELPER FUNCTIONS ====================

def create_feedback(
    session: Session,
    user_id: int,
    video_id: int,
    rating: int,
    comment: Optional[str] = None
) -> VideoFeedback:
    """
    Create a new feedback entry.
    
    Args:
        session: SQLModel session
        user_id: ID of the user
        video_id: ID of the video
        rating: Rating from 1-5
        comment: Optional comment
        
    Returns:
        Created VideoFeedback object
    """
    # Validate rating
    if rating < 1 or rating > 5:
        raise ValueError("Rating must be between 1 and 5")
    
    # Check if user already rated this video
    statement = select(VideoFeedback).where(
        VideoFeedback.user_id == user_id,
        VideoFeedback.video_id == video_id
    )
    existing = session.exec(statement).first()
    
    if existing:
        # Update existing feedback
        existing.update_rating(rating, comment)
        session.add(existing)
        session.commit()
        session.refresh(existing)
        return existing
    
    # Create new feedback
    feedback = VideoFeedback(
        user_id=user_id,
        video_id=video_id,
        rating=rating,
        comment=comment
    )
    session.add(feedback)
    session.commit()
    session.refresh(feedback)
    return feedback


def get_video_feedbacks(
    session: Session,
    video_id: int,
    limit: int = 50,
    offset: int = 0
) -> List[VideoFeedback]:
    """
    Get all feedbacks for a video.
    
    Args:
        session: SQLModel session
        video_id: ID of the video
        limit: Maximum number of entries to return
        offset: Offset for pagination
        
    Returns:
        List of VideoFeedback objects
    """
    statement = (
        select(VideoFeedback)
        .where(VideoFeedback.video_id == video_id)
        .order_by(VideoFeedback.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    results = session.exec(statement)
    return results.all()


def get_user_feedbacks(
    session: Session,
    user_id: int,
    limit: int = 50,
    offset: int = 0
) -> List[VideoFeedback]:
    """
    Get all feedbacks by a user.
    
    Args:
        session: SQLModel session
        user_id: ID of the user
        limit: Maximum number of entries to return
        offset: Offset for pagination
        
    Returns:
        List of VideoFeedback objects
    """
    statement = (
        select(VideoFeedback)
        .where(VideoFeedback.user_id == user_id)
        .order_by(VideoFeedback.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    results = session.exec(statement)
    return results.all()


def delete_feedback(
    session: Session,
    feedback_id: int,
    user_id: Optional[int] = None
) -> bool:
    """
    Delete a feedback entry.
    
    Args:
        session: SQLModel session
        feedback_id: ID of the feedback to delete
        user_id: Optional user ID for authorization check
        
    Returns:
        True if deleted, False if not found or unauthorized
    """
    feedback = session.get(VideoFeedback, feedback_id)
    if not feedback:
        return False
    
    # Check authorization if user_id provided
    if user_id and feedback.user_id != user_id:
        return False
    
    session.delete(feedback)
    session.commit()
    return True