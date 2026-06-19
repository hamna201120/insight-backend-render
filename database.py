from sqlmodel import SQLModel, create_engine, Session
from models import User, Video

# Database URL (SQLite file in the current folder)
DATABASE_URL = "sqlite:///./app.db"

# Create engine
engine = create_engine(
    DATABASE_URL, 
    echo=True,  # Optional: prints all SQL statements (useful for debugging)
    connect_args={"check_same_thread": False}  # Required for SQLite + FastAPI
)

# Initialize database tables
def init_db():
    SQLModel.metadata.create_all(engine)
    print("Database tables created (if they didn't exist).")

# FIXED: Dependency for FastAPI to provide a session
def get_session():
    with Session(engine) as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise  # Re-raise the exception
        # No finally block needed - the 'with' statement closes automatically