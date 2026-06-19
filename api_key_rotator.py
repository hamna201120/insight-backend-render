# api_key_rotator.py - UPDATED FOR NEW SDK
import os
from google import genai
from typing import Optional
from datetime import datetime, timedelta

class GeminiKeyRotator:
    def __init__(self):
        """Load all Gemini API keys from environment"""
        self.keys = []
        self.key_status = {}
        self.current_index = 0
        
        # Load all keys (support up to 10 keys)
        for i in range(1, 11):
            key = os.getenv(f"GEMINI_API_KEY_{i}")
            if key and key.strip():
                self.keys.append(key.strip())
                self.key_status[key] = {
                    "available": True,
                    "reset_time": None,
                    "failed_count": 0
                }
        
        # Fallback: If no numbered keys, try single key
        if not self.keys:
            single_key = os.getenv("GEMINI_API_KEY")
            if single_key:
                self.keys.append(single_key)
                self.key_status[single_key] = {
                    "available": True,
                    "reset_time": None,
                    "failed_count": 0
                }
        
        if not self.keys:
            raise ValueError("No Gemini API keys found! Add GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc. to .env")
        
        print(f"✅ Loaded {len(self.keys)} Gemini API keys")
        for i, key in enumerate(self.keys):
            print(f"   Key {i+1}: {key[:10]}...")
    
    def get_available_key(self) -> Optional[str]:
        """Get the next available API key - instantly skip exhausted keys"""
        for attempt in range(len(self.keys) * 2):
            key = self.keys[self.current_index % len(self.keys)]
            status = self.key_status.get(key, {})
            
            # Check if key is available
            if not status.get("available", True):
                # Key is exhausted - skip immediately (no waiting)
                print(f"⏭️ Skipping exhausted key: {key[:10]}...")
                self.current_index += 1
                continue
            
            # Key is available, use it
            self.current_index += 1  # Round-robin for next call
            return key
        
        # All keys exhausted - return None so BART fallback kicks in
        print("❌ All Gemini keys are exhausted!")
        return None
    
    def mark_key_failed(self, key: str, error: Exception):
        """Mark a key as failed (quota exceeded) - NO cooldown wait"""
        if key not in self.key_status:
            return
        
        error_msg = str(error).lower()
        
        # Check if it's a quota/rate limit error
        if any(kw in error_msg for kw in ['quota', 'rate limit', '429', 'too many', 'daily']):
            # Mark as unavailable - NO wait time
            self.key_status[key]["available"] = False
            self.key_status[key]["failed_count"] += 1
            self.key_status[key]["reset_time"] = None  # No auto-reset
            
            print(f"⚠️ Key {key[:10]}... marked as EXHAUSTED (quota finished)")
            print(f"   Total failed: {self.key_status[key]['failed_count']} times")
        
        # For other errors (permission, auth, etc.)
        elif "permission" in error_msg or "auth" in error_msg:
            self.key_status[key]["available"] = False
            self.key_status[key]["failed_count"] += 1
            print(f"⚠️ Key {key[:10]}... marked as FAILED (auth error)")
    
    def reset_all_keys(self):
        """Reset all keys manually (emergency)"""
        for key in self.keys:
            self.key_status[key]["available"] = True
            self.key_status[key]["reset_time"] = None
            self.key_status[key]["failed_count"] = 0
        print("✅ All keys reset!")
    
    def get_status(self) -> dict:
        """Get status of all keys"""
        available_count = sum(1 for k in self.keys if self.key_status.get(k, {}).get("available", True))
        return {
            "total_keys": len(self.keys),
            "available_keys": available_count,
            "keys": [
                {
                    "key": k[:10] + "...",
                    "available": self.key_status.get(k, {}).get("available", True),
                    "failed_count": self.key_status.get(k, {}).get("failed_count", 0)
                }
                for k in self.keys
            ]
        }

# Global instance
key_rotator = GeminiKeyRotator()