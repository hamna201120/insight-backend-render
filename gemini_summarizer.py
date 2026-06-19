# gemini_summarizer.py - Gemini 2.5 Models (NO BART FALLBACK)
import os
from google import genai
from typing import List, Dict, Optional
import time
import json
import re
from api_key_rotator import key_rotator

class GeminiSummarizer:
    def __init__(self):
        """Initialize with automatic key rotation"""
        self.client = None
        self.current_key = None
        self.max_retries = 10
        
        # Initialize with first available key
        self._initialize_with_available_key()
    
    def _initialize_with_available_key(self) -> bool:
        """Initialize Gemini with an available API key"""
        key = key_rotator.get_available_key()
        if not key:
            print("❌ No available Gemini keys!")
            return False
        
        try:
            self.client = genai.Client(api_key=key)
            self.current_key = key
            print(f"✅ Gemini 2.5 Flash initialized with key: {key[:10]}...")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize with key {key[:10]}...: {e}")
            key_rotator.mark_key_failed(key, e)
            return self._initialize_with_available_key()
    
    def _rotate_key(self, error: Exception = None) -> bool:
        """Rotate to next available key"""
        if error and self.current_key:
            key_rotator.mark_key_failed(self.current_key, error)
        return self._initialize_with_available_key()
    
    def _parse_response(self, response_text: str) -> Dict:
        """Parse Gemini response, handling JSON and text formats"""
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                defaults = {
                    'short_summary': '',
                    'detailed_summary': '',
                    'key_points': [],
                    'topics_covered': [],
                    'recommendations': [],
                    'value_summary': '',
                    'watch_decision': {}
                }
                for key, default in defaults.items():
                    if key not in result:
                        result[key] = default
                return result
        except:
            pass
        
        # Create structured response from plain text
        lines = response_text.split('\n')
        key_points = []
        for line in lines:
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•') or line.startswith('*') or line.startswith('1.') or line.startswith('2.')):
                clean = re.sub(r'^[\d\-•*.\s]+', '', line)
                if len(clean) > 10:
                    key_points.append(clean)
        
        return {
            "short_summary": response_text[:300] + '...' if len(response_text) > 300 else response_text,
            "detailed_summary": response_text[:1000] if len(response_text) > 1000 else response_text,
            "key_points": key_points[:8],
            "topics_covered": ["Main Content", "Key Topics"],
            "recommendations": ["Watch the full video for detailed insights"],
            "value_summary": "Educational content about the topic",
            "watch_decision": {
                "best_for": "interested viewers",
                "worth_watching": True,
                "why": "Contains valuable information"
            }
        }
    
    def summarize_video(self, transcript: str, duration_minutes: float = 0, detailed: bool = True) -> Dict:
        """Generate comprehensive video summary with auto-rotation"""
        
        transcript = transcript.strip()
        
        # Determine video length category
        if duration_minutes > 30:
            length_category = f"long ({int(duration_minutes)} minutes)"
            focus = "Provide a comprehensive breakdown with main sections and detailed insights"
        elif duration_minutes > 10:
            length_category = f"medium ({int(duration_minutes)} minutes)"
            focus = "Balance detail with conciseness, highlight the most important concepts"
        else:
            length_category = f"short ({int(duration_minutes)} minutes)"
            focus = "Be concise but comprehensive, capture the essence of the video"
        
        # Truncate transcript if too long
        if len(transcript) > 1000000:
            transcript = transcript[:1000000]
        
        prompt = f"""You are an EXPERT video summarizer. Analyze this YouTube video transcript and provide a USEFUL summary.

VIDEO LENGTH: {length_category}
FOCUS: {focus}

TRANSCRIPT:
{transcript}

Please provide a JSON response with these fields:
- short_summary: A concise 2-3 sentence overview
- detailed_summary: A comprehensive 2-3 paragraph summary
- key_points: Array of 5-8 key takeaways (as strings)
- topics_covered: Array of main topics covered (as strings)
- recommendations: Array of 2-3 actionable recommendations
- value_summary: What a viewer will gain from watching this video
- watch_decision: {{"best_for": "target audience", "worth_watching": true/false, "why": "reason"}}

Format as valid JSON only. No other text."""

        # Try Gemini with all available keys - INSTANT rotation
        for attempt in range(self.max_retries):
            try:
                if not self.client:
                    if not self._initialize_with_available_key():
                        print("❌ No Gemini keys available!")
                        return {
                            "short_summary": "No Gemini API keys available",
                            "detailed_summary": "Please add valid Gemini API keys",
                            "key_points": ["No key points available"],
                            "topics_covered": ["N/A"],
                            "recommendations": ["Add Gemini API keys in Railway variables"],
                            "value_summary": "Processing failed due to missing API keys",
                            "watch_decision": {"best_for": "N/A", "worth_watching": False, "why": "No API keys available"},
                            "ai_model_used": "None",
                            "processing_method": "error"
                        }
                
                # Use Gemini 2.5 Flash
                response = self.client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt,
                    config={
                        'temperature': 0.3,
                        'top_p': 0.8,
                        'top_k': 40,
                        'max_output_tokens': 8192,
                    }
                )
                
                result = self._parse_response(response.text)
                result['ai_model_used'] = 'Gemini 2.5 Flash'
                result['processing_method'] = 'gemini_2_5_flash'
                result['video_length_minutes'] = duration_minutes
                result['key_used'] = self.current_key[:10] + '...' if self.current_key else 'Unknown'
                result['attempt'] = attempt + 1
                
                print(f"✅ Gemini 2.5 Flash successful! (Key: {self.current_key[:10]}...)")
                return result
                
            except Exception as e:
                error_msg = str(e).lower()
                print(f"⚠️ Attempt {attempt + 1} failed with key {self.current_key[:10] if self.current_key else 'None'}...: {e}")
                
                # Check if error is quota/rate limit related
                if any(kw in error_msg for kw in ['quota', 'rate limit', '429', 'too many', 'daily']):
                    if self.current_key:
                        key_rotator.mark_key_failed(self.current_key, e)
                    
                    if not self._initialize_with_available_key():
                        print("🚨 All Gemini keys exhausted!")
                        return {
                            "short_summary": "All Gemini API keys exhausted",
                            "detailed_summary": "Please add more API keys or wait for quota reset",
                            "key_points": ["No key points available"],
                            "topics_covered": ["N/A"],
                            "recommendations": ["Add more Gemini API keys in Railway variables"],
                            "value_summary": "Processing failed due to quota limits",
                            "watch_decision": {"best_for": "N/A", "worth_watching": False, "why": "API quota exhausted"},
                            "ai_model_used": "None",
                            "processing_method": "quota_exhausted"
                        }
                    continue
                else:
                    self._rotate_key(e)
                    continue
        
        # If all attempts fail
        print("🚨 All Gemini attempts failed!")
        return {
            "short_summary": "All Gemini API attempts failed",
            "detailed_summary": "Please check your API keys and try again",
            "key_points": ["No key points available"],
            "topics_covered": ["N/A"],
            "recommendations": ["Check Gemini API keys in Railway variables"],
            "value_summary": "Processing failed due to API errors",
            "watch_decision": {"best_for": "N/A", "worth_watching": False, "why": "API errors"},
            "ai_model_used": "None",
            "processing_method": "error"
        }
