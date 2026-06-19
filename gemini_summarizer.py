# gemini_summarizer.py - Gemini 2.5 Models (FIXED INDENTATION)
import os
from google import genai
from typing import List, Dict, Optional
import time
import json
import re
from api_key_rotator import key_rotator

# Try to import BART for fallback
try:
    from transformers import pipeline
    BART_AVAILABLE = True
except ImportError:
    BART_AVAILABLE = False
    print("⚠️ BART not available. Install with: pip install transformers torch")

class GeminiSummarizer:
    def __init__(self):
        """Initialize with automatic key rotation and BART fallback"""
        self.client = None
        self.current_key = None
        self.max_retries = 10
        self.bart_summarizer = None
        
        # Load BART fallback
        if BART_AVAILABLE:
            try:
                print("📥 Loading BART fallback model...")
                self.bart_summarizer = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",
                    device=-1,
                    truncation=True
                )
                print("✅ BART fallback loaded successfully")
            except Exception as e:
                print(f"⚠️ BART loading failed: {e}")
                self.bart_summarizer = None
        
        # Initialize with first available key
        self._initialize_with_available_key()
    
    def _initialize_with_available_key(self) -> bool:
        """Initialize Gemini with an available API key"""
        key = key_rotator.get_available_key()
        if not key:
            print("❌ No available Gemini keys!")
            return False
        
        try:
            # NEW SDK: Use genai.Client
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
    
    def _use_bart_fallback(self, transcript: str, duration_minutes: float) -> Dict:
        """Use BART as fallback when all Gemini keys fail"""
        print("🚨 ALL GEMINI KEYS EXHAUSTED! Using BART fallback...")
        
        if not self.bart_summarizer:
            return self._extractive_fallback(transcript)
        
        try:
            words = transcript.split()
            chunk_size = 500
            chunks = []
            
            for i in range(0, min(len(words), 3000), chunk_size):
                chunk = ' '.join(words[i:i+chunk_size])
                chunks.append(chunk)
            
            summaries = []
            for chunk in chunks[:3]:
                try:
                    summary = self.bart_summarizer(
                        chunk,
                        max_length=150,
                        min_length=50,
                        do_sample=False
                    )[0]["summary_text"]
                    summaries.append(summary)
                except:
                    summaries.append(chunk[:200] + "...")
            
            final_summary = ' '.join(summaries)
            
            if len(final_summary.split()) > 200:
                try:
                    final_summary = self.bart_summarizer(
                        final_summary,
                        max_length=150,
                        min_length=60,
                        do_sample=False
                    )[0]["summary_text"]
                except:
                    pass
            
            return {
                "short_summary": final_summary[:300] + "..." if len(final_summary) > 300 else final_summary,
                "detailed_summary": final_summary[:1000] if len(final_summary) > 1000 else final_summary,
                "key_points": self._extract_key_points(transcript),
                "topics_covered": self._extract_topics(transcript),
                "recommendations": [
                    "This summary was generated using BART (Gemini API keys were exhausted)",
                    "Consider watching the full video for complete understanding"
                ],
                "value_summary": "Content summary using BART fallback",
                "watch_decision": {
                    "best_for": "All viewers",
                    "worth_watching": True,
                    "why": "Contains valuable information"
                },
                "processing_method": "bart_fallback",
                "ai_model_used": "BART (Fallback)",
                "gemini_failed": True,
                "chunks_processed": len(chunks)
            }
        except Exception as e:
            print(f"❌ BART fallback failed: {e}")
            return self._extractive_fallback(transcript)
    
    def _extractive_fallback(self, transcript: str) -> Dict:
        """Ultimate extractive fallback when everything fails"""
        sentences = re.split(r'(?<=[.!?])\s+', transcript)
        
        if len(sentences) <= 10:
            selected = sentences
        else:
            selected = sentences[:5] + sentences[-5:]
        
        summary = ' '.join(selected)
        
        return {
            "short_summary": summary[:300] + "..." if len(summary) > 300 else summary,
            "detailed_summary": summary[:800] + "..." if len(summary) > 800 else summary,
            "key_points": ["Video content extracted using fallback method"],
            "topics_covered": ["General Content"],
            "recommendations": ["Watch the full video for complete understanding"],
            "value_summary": "Extractive summary from transcript",
            "watch_decision": {
                "best_for": "All viewers",
                "worth_watching": True,
                "why": "Contains valuable information"
            },
            "processing_method": "extractive_fallback",
            "ai_model_used": "Extractive (Final Fallback)",
            "gemini_failed": True
        }
    
    def _extract_key_points(self, text: str) -> List[str]:
        """Extract key points from text (for BART fallback)"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        keywords = ['important', 'key', 'critical', 'main', 'essential', 'significant']
        result = []
        for sent in sentences[:20]:
            if any(kw in sent.lower() for kw in keywords):
                if len(sent.split()) > 5:
                    result.append(sent)
        return result[:5] if result else sentences[:5]
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from text (for BART fallback)"""
        words = text.lower().split()
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'by', 'is', 'are', 'was', 'were', 'this', 'that', 'it'}
        word_freq = {}
        for word in words:
            word = word.strip('.,!?;:"\'()[]{}')
            if word not in stop_words and len(word) > 3 and word.isalpha():
                word_freq[word] = word_freq.get(word, 0) + 1
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        return [w.capitalize() for w, _ in top_words] if top_words else ["General Content"]
    
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
        """Generate comprehensive video summary with auto-rotation - NO WAITING"""
        
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
                        # No keys available, immediately use BART
                        print("🚨 No Gemini keys available! Using BART fallback...")
                        return self._use_bart_fallback(transcript, duration_minutes)
                
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
                    # Mark current key as exhausted
                    if self.current_key:
                        key_rotator.mark_key_failed(self.current_key, e)
                    
                    # IMMEDIATELY try next key (no waiting!)
                    if not self._initialize_with_available_key():
                        # No keys available, use BART
                        print("🚨 All Gemini keys exhausted! Using BART fallback...")
                        return self._use_bart_fallback(transcript, duration_minutes)
                    
                    # Continue loop with next key
                    continue
                else:
                    # Other error (not quota related) - still try next key
                    self._rotate_key(e)
                    continue
        
        # If all attempts fail, use BART fallback
        print("🚨 All Gemini attempts failed! Using BART fallback...")
        return self._use_bart_fallback(transcript, duration_minutes)