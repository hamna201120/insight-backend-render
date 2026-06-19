# add_feedback_table.py
import sqlite3

def add_feedback_table():
    """Add feedback table to existing database without losing data"""
    
    print("🔍 Checking existing database...")
    
    # Connect to existing database
    conn = sqlite3.connect('app.db')
    cursor = conn.cursor()
    
    # Check if feedback table already exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='videofeedback'")
    exists = cursor.fetchone()
    
    if exists:
        print("✅ Feedback table already exists! No changes needed.")
        conn.close()
        return True
    
    print("📝 Creating feedback table...")
    
    # Create the feedback table
    cursor.execute('''
    CREATE TABLE videofeedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        video_id INTEGER NOT NULL,
        rating INTEGER NOT NULL,
        comment TEXT,
        created_at TIMESTAMP NOT NULL,
        updated_at TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES user (id),
        FOREIGN KEY (video_id) REFERENCES video (id)
    )
    ''')
    
    # Create indexes for better performance
    cursor.execute('CREATE INDEX ix_videofeedback_user_id ON videofeedback (user_id)')
    cursor.execute('CREATE INDEX ix_videofeedback_video_id ON videofeedback (video_id)')
    
    conn.commit()
    print("✅ Feedback table added successfully!")
    
    # Verify the table was created
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print(f"📊 Current tables: {[t[0] for t in tables]}")
    
    conn.close()
    return True

def verify_data():
    """Verify existing data is still intact"""
    
    conn = sqlite3.connect('app.db')
    cursor = conn.cursor()
    
    # Check users count
    cursor.execute("SELECT COUNT(*) FROM user")
    user_count = cursor.fetchone()[0]
    print(f"👥 Users in database: {user_count}")
    
    # Check videos count
    cursor.execute("SELECT COUNT(*) FROM video")
    video_count = cursor.fetchone()[0]
    print(f"🎬 Videos in database: {video_count}")
    
    conn.close()
    
    if user_count > 0 or video_count > 0:
        print("✅ Your existing data is preserved and intact!")
    else:
        print("⚠️ No data found in database")

if __name__ == "__main__":
    print("=" * 50)
    print("Adding Feedback Table to Existing Database")
    print("=" * 50)
    
    add_feedback_table()
    verify_data()
    
    print("\n" + "=" * 50)
    print("Done! You can now restart your backend:")
    print("uvicorn main:app --reload --host 0.0.0.0 --port 8000")
    print("=" * 50)