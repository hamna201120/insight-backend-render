# ai_service.py - ENHANCED FOR LONG VIDEOS
import os
import google.generativeai as genai
import re
import json
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import local models (FREE)
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️ transformers not installed. Install with: pip install transformers torch")
    TRANSFORMERS_AVAILABLE = False

# Try OpenAI (Optional - only if you have API key)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Import our new modules
try:
    from smart_chunker import SmartChunker
    from hierarchical_summarizer import HierarchicalSummarizer
    ADVANCED_AVAILABLE = True
except ImportError:
    print("⚠️ Advanced modules not found. Install with: pip install sentence-transformers")
    ADVANCED_AVAILABLE = False
    # Create dummy classes
    class SmartChunker:
        def chunk_transcript(self, transcript, timestamps=None):
            return [{'text': transcript, 'word_count': len(transcript.split())}]
    
    class HierarchicalSummarizer:
        def __init__(self, summarizer):
            self.summarizer = summarizer
        def generate_hierarchical_summaries(self, chunks):
            return {
                'short_summary': 'Advanced summarization not available',
                'detailed_summary': 'Advanced summarization not available',
                'section_summaries': [],
                'chunk_summaries': [],
                'key_points': [],
                'key_points_with_timestamps': [],
                'topics_covered': [],
                'recommendations': []
            }

class SummarizationService:
    def __init__(self):
        self.openai_client = None
        self.local_summarizer = None
        self.smart_chunker = None
        self.hierarchical_summarizer = None
        self.gemini_model = None
        
        print("🚀 Initializing AI Summarization Service...")
        
        # Try OpenAI first
        if OPENAI_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key and api_key != "your_actual_api_key_here":
                self.openai_client = OpenAI(api_key=api_key)
                print("✅ OpenAI client initialized")
            else:
                print("ℹ️ Using free local models")
        
        # Initialize Gemini (FREE)
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                genai.configure(api_key=api_key)
                self.gemini_model = genai.GenerativeModel("gemini-1.5-flash")
                print("✅ Gemini model initialized (for long videos)")
            except Exception as e:
                print(f"⚠️ Gemini init failed: {e}")
        
        # Initialize local models (FREE)
        if TRANSFORMERS_AVAILABLE:
            try:
                print("📥 Loading local summarization model (BART-large-cnn)...")
                self.local_summarizer = pipeline(
                    "summarization", 
                    model="facebook/bart-large-cnn",
                    tokenizer="facebook/bart-large-cnn",
                    device=-1,  # Use CPU
                    truncation=True
                )
                print("✅ Local summarizer initialized")
                
                # Initialize advanced processors
                self.smart_chunker = SmartChunker()
                self.hierarchical_summarizer = HierarchicalSummarizer(self.local_summarizer)
                print("✅ Advanced processors initialized")
                
            except Exception as e:
                print(f"⚠️ Could not load BART model: {e}")
                self.local_summarizer = None
        
        # Fallback
        if not self.openai_client and not self.local_summarizer and not self.gemini_model:
            print("⚠️ No AI model available. Using dummy responses.")
    
    # ⭐ 1️⃣ REPETITION + HALLUCINATION GUARD
    def _anti_repetition(self, text: str) -> str:
        """Remove repetitive phrases and normalize spacing."""
        text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _hallucination_guard(self, summary: str, source: str) -> str:
        """Ensure summary is grounded in source text."""
        overlap = len(set(summary.lower().split()) & set(source.lower().split())) / max(1, len(summary.split()))
        if overlap < 0.15:  # Less than 15% word overlap -> possible hallucination
            return source[:400]  # fallback extractive
        return summary
    
    def _remove_youtube_noise(self, text: str) -> str:
        """Remove YouTube outro, subscription prompts, and engagement noise."""
        noise_patterns = [
            r"subscribe.*",
            r"hit the bell.*",
            r"like and share.*",
            r"thanks for watching.*",
            r"see you in the next video.*",
            r"follow me.*",
            r"channel.*",
            r"comment below.*",
            r"don't forget to.*",
            r"please subscribe.*",
            r"smash that like.*",
            r"turn on notifications.*",
            r"support the channel.*",
            r"patreon.*",
            r"merch.*",
            r"check out my.*"
        ]
        
        for pattern in noise_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        
        # Clean up extra spaces and newlines
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r' +', ' ', text)
        
        return text.strip()
    
    # ⭐ 2️⃣ UPGRADED GEMINI CHUNK SUMMARIZATION
    def _gemini_summarize_chunk(self, text: str) -> str:
        """Use Gemini to create research-quality summaries of chunks."""
        if not self.gemini_model:
            return text[:200]

        text = self._remove_youtube_noise(text)
        
        # Truncate if too long for Gemini context
        if len(text) > 30000:  # Gemini 1.5 Flash has high context but we'll be safe
            text = text[:30000]

        prompt = f"""
You are a research summarization system.

Create an abstractive analytical summary explaining:
• what the video segment teaches
• core insights
• cause-effect reasoning
• practical implications

Avoid:
• repetition
• transcript copying
• promotional language
• filler words

Segment:
{text}

ANALYTICAL SUMMARY:
"""

        try:
            response = self.gemini_model.generate_content(prompt)
            summary = response.text.strip()
            
            # Apply quality guards
            summary = self._anti_repetition(summary)
            summary = self._hallucination_guard(summary, text)
            
            return summary if summary else text[:200]
        except Exception as e:
            print(f"Gemini error: {e}")
            return text[:200]
    
    # ⭐ 6️⃣ BETTER KEYPOINT EXTRACTION
    def _gemini_keypoints(self, text: str) -> List[str]:
        """Extract research-quality key points using Gemini."""
        if not self.gemini_model:
            return self._generate_key_points_fallback(text)
        
        # Truncate if too long
        if len(text) > 30000:
            text = text[:30000]

        prompt = f"""
Extract 5 high-value insights.

Each insight must:
• be specific
• contain learning value
• avoid generic wording
• avoid promotion

Transcript:
{text}

INSIGHTS:
"""
        try:
            response = self.gemini_model.generate_content(prompt)
            points = response.text.strip().split("\n")
            
            # Clean and filter points
            cleaned_points = []
            for p in points:
                # Remove bullet points and numbering
                clean_p = re.sub(r'^[\d•\-*.\s]+', '', p).strip()
                # Remove empty or too short points
                if len(clean_p) > 10 and clean_p not in cleaned_points:
                    # Remove any remaining promotional content
                    if not any(word in clean_p.lower() for word in ['subscribe', 'like', 'share', 'comment', 'bell']):
                        # Apply anti-repetition to each point
                        clean_p = self._anti_repetition(clean_p)
                        cleaned_points.append(clean_p)
            
            return cleaned_points[:5]
        except Exception as e:
            print(f"Gemini keypoints error: {e}")
            return self._generate_key_points_fallback(text)
    
    def _clean_whisper_transcript(self, text: str) -> str:
        """Clean and format Whisper transcripts."""
        if not text:
            return text
        
        # Remove excessive filler words
        filler_words = [' um ', ' uh ', ' like ', ' you know ', ' basically ', 
                       ' actually ', ' literally ', ' I mean ', ' so ', ' well ',
                       ' okay ', ' right ', ' anyway ', ' sort of ', ' kind of ']
        
        for filler in filler_words:
            text = text.replace(filler, ' ')
        
        # Remove repeated words
        text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
        
        # Remove timestamps
        text = re.sub(r'\d{1,2}:\d{2}(:\d{2})?', '', text)
        
        # Apply YouTube noise removal
        text = self._remove_youtube_noise(text)
        
        # Normalize spacing
        text = ' '.join(text.split())
        
        return text
    
    def _parse_whisper_output(self, transcript_text: str) -> tuple:
        """
        Parse Whisper output to extract text and approximate timestamps.
        Returns (clean_text, timestamps_list)
        """
        # Simple parsing - in real scenario, you'd get this from Whisper directly
        # For now, we'll create approximate timestamps based on sentence count
        sentences = re.split(r'(?<=[.!?])\s+', transcript_text)
        timestamps = []
        
        # Approximate 3 seconds per sentence on average
        current_time = 0
        for sentence in sentences:
            duration = len(sentence.split()) * 0.3  # ~0.3 sec per word
            timestamps.append({
                'text': sentence,
                'start': current_time,
                'end': current_time + duration
            })
            current_time += duration
        
        return transcript_text, timestamps
    
    def _generate_key_points_fallback(self, text: str) -> List[str]:
        """Fallback key points extraction."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        key_sentences = []
        for sentence in sentences[:10]:  # Check first 10 sentences
            sentence = sentence.strip()
            if len(sentence.split()) >= 8 and len(sentence.split()) <= 30:
                if any(word in sentence.lower() for word in ['important', 'key', 'main', 'essential']):
                    key_sentences.append(sentence)
        
        # If not enough, take first few meaningful sentences
        if len(key_sentences) < 3:
            for sentence in sentences[:5]:
                if len(sentence.split()) >= 6:
                    key_sentences.append(sentence)
                    if len(key_sentences) >= 5:
                        break
        
        return key_sentences[:5]
    
    def _generate_topics_fallback(self, text: str) -> List[str]:
        """Fallback topic extraction."""
        cleaned_text = self._clean_whisper_transcript(text[:2000])
        words = cleaned_text.lower().split()
        
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'by', 'is', 'are', 'was', 'were', 'this', 'that', 'it',
                     'we', 'they', 'you', 'i', 'he', 'she', 'be', 'been', 'have', 'has'}
        
        # Count words
        word_count = {}
        for word in words:
            word = word.strip('.,!?;:"\'()[]{}')
            if (word not in stop_words and 
                len(word) > 3 and 
                word.isalpha()):
                word_count[word] = word_count.get(word, 0) + 1
        
        # Get top words
        topics = sorted(word_count.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return [word.capitalize() for word, _ in topics] if topics else ["Main Concepts"]
    
    def _generate_recommendation_fallback(self, text: str) -> str:
        """Fallback recommendation."""
        word_count = len(text.split())
        
        if word_count > 2000:
            return "This comprehensive video is highly recommended. Consider watching in segments for better understanding."
        elif word_count > 500:
            return "This video provides good insights. Recommended for anyone interested in the topic."
        else:
            return "A concise video covering key points. Good for quick learning."
    
    def generate_all_summaries(self, transcript: str) -> Dict:
        """
        Generate all summaries - automatically detects video length and uses appropriate method.
        
        Args:
            transcript: Full transcript text
            
        Returns:
            Dictionary with summaries, key points, topics, recommendations
        """
        word_count = len(transcript.split())
        print(f"📊 Processing transcript of {word_count} words...")
        
        # Clean transcript
        cleaned_transcript = self._clean_whisper_transcript(transcript)
        
        # DECISION POINT: Use Gemini hybrid pipeline for long videos
        if word_count > 1500 and self.gemini_model and self.smart_chunker:
            print("🚀 Long video detected → Using Gemini hybrid pipeline for research-quality summaries")
            
            try:
                # Parse Whisper output for timestamps
                _, timestamps = self._parse_whisper_output(cleaned_transcript)
                
                # Smart chunking
                print("📦 Smart chunking transcript...")
                chunks = self.smart_chunker.chunk_transcript(cleaned_transcript, timestamps)
                print(f"✅ Created {len(chunks)} intelligent chunks")
                
                # Gemini summarization with context overlap and quota optimization
                print("🤖 Generating Gemini summaries for each chunk...")
                chunk_summaries = []
                for i, chunk in enumerate(chunks):
                    chunk_text = chunk["text"]
                    
                    # ⭐ 3️⃣ CONTEXT-AWARE CHUNK CONTINUITY
                    if i > 0:
                        context = chunk_summaries[-1][-200:]
                        chunk_text = context + " " + chunk_text
                    
                    print(f"  Processing chunk {i+1}/{len(chunks)}...")
                    
                    # ⭐ 7️⃣ QUOTA OPTIMIZATION
                    if i % 2 == 0:
                        summary = self._gemini_summarize_chunk(chunk_text)
                    else:
                        if self.local_summarizer:
                            try:
                                bart_summary = self.local_summarizer(
                                    chunk_text[:1000], 
                                    max_length=80, 
                                    min_length=30,
                                    do_sample=False
                                )[0]["summary_text"]
                                summary = bart_summary
                            except:
                                summary = self._gemini_summarize_chunk(chunk_text)
                        else:
                            summary = self._gemini_summarize_chunk(chunk_text)
                    
                    chunk_summaries.append(summary)
                
                # ⭐ 4️⃣ FINAL RESEARCH-LEVEL SYNTHESIS
                print("📝 Generating final research synthesis...")
                fusion_prompt = f"""
You are creating a research-grade executive briefing.

Synthesize the following segment summaries into:
• a coherent explanation
• major themes
• deep insights
• decision usefulness
• practical takeaways

Avoid repetition.

Segments:
{" ".join(chunk_summaries)}

EXECUTIVE BRIEFING:
"""
                
                response = self.gemini_model.generate_content(fusion_prompt)
                final_summary = response.text.strip()
                final_summary = self._anti_repetition(final_summary)
                
                # ⭐ 5️⃣ HIERARCHICAL + GEMINI FUSION
                if self.local_summarizer:
                    try:
                        bart_refined = self.local_summarizer(
                            final_summary,
                            max_length=120,
                            min_length=50,
                            do_sample=False
                        )[0]["summary_text"]
                        final_summary = bart_refined
                        print("✅ Applied BART refinement to final summary")
                    except Exception as e:
                        print(f"⚠️ BART refinement failed: {e}")
                
                # ⭐ 8️⃣ FINAL KEYPOINT SUPER-FUSION (using final_summary instead of chunk_summaries)
                print("🔑 Extracting key points from final synthesis...")
                key_points = self._gemini_keypoints(final_summary)
                
                # Generate topics from final summary
                topics = self._generate_topics_fallback(final_summary)
                
                # Prepare results
                results = {
                    "short_summary": final_summary,
                    "detailed_summary": final_summary,
                    "section_summaries": chunk_summaries,
                    "chunk_summaries": chunk_summaries,
                    "key_points": key_points if key_points else ["No key points extracted"],
                    "key_points_with_timestamps": [],  # Could be enhanced later
                    "topics_covered": topics,
                    "recommendations": ["Highly informative video recommended"],
                    "ai_model_used": "Gemini + Smart Chunker Hybrid + BART Fusion",
                    "processing_method": "hybrid_gemini_advanced",
                    "chunks_processed": len(chunks)
                }
                
                print("✅ Gemini hybrid summarization complete!")
                return results
                
            except Exception as e:
                print(f"⚠️ Gemini hybrid summarization failed: {e}. Falling back to standard method.")
                # Fall through to standard method
        
        # Use hierarchical BART for long videos when Gemini not available
        elif word_count > 1500 and self.local_summarizer and ADVANCED_AVAILABLE:
            print(f"🎯 Long video detected ({word_count} words). Using advanced hierarchical summarization...")
            
            try:
                # Parse Whisper output for timestamps
                _, timestamps = self._parse_whisper_output(cleaned_transcript)
                
                # Smart chunking
                print("📦 Smart chunking transcript...")
                chunks = self.smart_chunker.chunk_transcript(cleaned_transcript, timestamps)
                print(f"✅ Created {len(chunks)} intelligent chunks")
                
                # Hierarchical summarization
                print("🏗️ Generating hierarchical summaries...")
                results = self.hierarchical_summarizer.generate_hierarchical_summaries(chunks)
                
                # Add metadata
                results['ai_model_used'] = "Advanced Hierarchical BART (FREE)"
                results['processing_method'] = "hierarchical"
                results['chunks_processed'] = len(chunks)
                
                print("✅ Advanced summarization complete!")
                return results
                
            except Exception as e:
                print(f"⚠️ Advanced summarization failed: {e}. Falling back to standard method.")
                # Fall through to standard method
        
        # STANDARD METHOD FOR SHORT VIDEOS
        print("📝 Using standard summarization for shorter video...")
        
        # Generate short summary
        if self.local_summarizer:
            try:
                # For shorter videos, still use chunking but simpler
                if word_count > 600:
                    chunks = self._chunk_text_fallback(cleaned_transcript, max_words=300)
                    summaries = []
                    
                    for chunk in chunks[:3]:  # Limit to 3 chunks
                        try:
                            result = self.local_summarizer(
                                chunk,
                                max_length=min(60, len(chunk.split())),
                                min_length=20,
                                do_sample=False,
                                truncation=True
                            )
                            summaries.append(result[0]['summary_text'])
                        except:
                            summaries.append(chunk[:100] + "...")
                    
                    short_summary = " ".join(summaries)
                    short_summary = self._anti_repetition(short_summary)
                    
                    if len(short_summary.split()) > 150:
                        try:
                            final = self.local_summarizer(
                                short_summary,
                                max_length=80,
                                min_length=30,
                                do_sample=False
                            )
                            short_summary = final[0]['summary_text']
                        except:
                            pass
                else:
                    result = self.local_summarizer(
                        cleaned_transcript,
                        max_length=80,
                        min_length=30,
                        do_sample=False,
                        truncation=True
                    )
                    short_summary = result[0]['summary_text']
                    short_summary = self._anti_repetition(short_summary)
            except Exception as e:
                print(f"⚠️ Summarization error: {e}")
                short_summary = f"This video discusses important concepts related to the topic."
        else:
            short_summary = f"This video covers key concepts and provides valuable insights."
        
        # Generate other elements
        key_points = self._generate_key_points_fallback(cleaned_transcript)
        topics_covered = self._generate_topics_fallback(cleaned_transcript)
        recommendation = self._generate_recommendation_fallback(cleaned_transcript)
        
        return {
            "short_summary": short_summary,
            "detailed_summary": short_summary,  # Same for short videos
            "section_summaries": [],
            "chunk_summaries": [],
            "key_points": key_points,
            "key_points_with_timestamps": [],
            "topics_covered": topics_covered,
            "recommendations": [recommendation],
            "ai_model_used": "Local BART (FREE)" if self.local_summarizer else "Dummy",
            "processing_method": "standard"
        }
    
    def _chunk_text_fallback(self, text: str, max_words: int = 300) -> List[str]:
        """Fallback chunking method."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_count = 0
        
        for sentence in sentences:
            word_count = len(sentence.split())
            
            if current_count + word_count > max_words and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_count = word_count
            else:
                current_chunk.append(sentence)
                current_count += word_count
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

# Global instance
summarization_service = SummarizationService()