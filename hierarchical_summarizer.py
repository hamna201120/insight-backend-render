# hierarchical_summarizer.py
import re
from typing import List, Dict, Optional
from collections import defaultdict
import numpy as np
import json
import random
import torch

# Try to import transformers for abstractive summarization
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ transformers not installed. Install with: pip install transformers torch")

# Try to import sentence-transformers for better topic extraction
try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️ sentence-transformers not installed. Install with: pip install sentence-transformers")

class HierarchicalSummarizer:
    """
    Generates multi-level ABSTRACTIVE summaries using BART model with GPT-style enhancements.
    True abstractive summarization with paraphrasing, synthesis, and connection-building.
    """
    
    def __init__(self):
        """Initialize with BART model for abstractive summarization."""
        self.embedding_model = None
        self.summarizer = None
        self.tokenizer = None
        self.model = None
        
        # Initialize BART for abstractive summarization
        if TRANSFORMERS_AVAILABLE:
            try:
                print("📥 Loading BART model for abstractive summarization...")
                # Use pipeline for summarization
                self.summarizer = pipeline(
                    "summarization", 
                    model="facebook/bart-large-cnn",
                    device=-1,
                    truncation=True
                )
                # Also load tokenizer for better control
                self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
                self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
                
                # ⭐ 9️⃣ Play Store production optimization
                self.model.config.use_cache = True
                torch.set_grad_enabled(False)
                
                print("✅ BART model loaded successfully")
            except Exception as e:
                print(f"⚠️ Could not load BART model: {e}")
                print("Will use extractive summarization as fallback")
        
        # Initialize sentence transformer for topic extraction
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                print("📥 Loading sentence transformer for topic extraction...")
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("✅ Sentence transformer loaded")
            except Exception as e:
                print(f"⚠️ Could not load sentence transformer: {e}")
    
    # ✅ 1️⃣ ADD YOUTUBE OUTRO / CTA NOISE FILTER (CRITICAL)
    def _remove_youtube_outro(self, text: str) -> str:
        """Remove YouTube outro, CTA, and engagement prompts."""
        noise_patterns = [
            r"like and subscribe.*",
            r"hit the bell icon.*",
            r"thanks for watching.*",
            r"see you in the next video.*",
            r"subscribe to.*channel.*",
            r"thumbs up.*",
            r"follow me.*",
            r"comment below.*",
            r"share this video.*",
            r"don't forget to like.*",
            r"please subscribe.*",
            r"smash that like button.*",
            r"turn on notifications.*",
            r"support the channel.*",
            r"check out my.*",
            r"patreon.*",
            r"merch.*"
        ]

        for pattern in noise_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        return text
    
    # ✅ 2️⃣ IMPROVE WHISPER CLEANING (REMOVE REPETITION + BROKEN WORDS)
    def _clean_whisper_transcript(self, text: str) -> str:
        """Clean Whisper transcript by removing filler words, artifacts, and YouTube noise."""
        text = re.sub(r'\[.*?\]', '', text)

        # remove filler words
        text = re.sub(r'(um|uh|you know|like|ah|er|hmm|basically|actually)', '', text, flags=re.I)

        # fix broken words (e.g., "hel lo" -> "hello")
        text = re.sub(r'\b(\w{1,2})\s+(\w+)', r'\1\2', text)

        # remove repetition
        text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)

        text = re.sub(r'\s+', ' ', text).strip()

        # ⭐ remove youtube outro
        text = self._remove_youtube_outro(text)

        return text
    
    def _fix_common_errors(self, text: str) -> str:
        """Fix common transcription and generation errors."""
        # Fix number formatting - "lack" to "00,000"
        text = re.sub(r'(\d+)\s+lack', r'\100,000', text)
        text = re.sub(r'(\d+)\s+million', r'$\1 million', text)
        text = re.sub(r'(\d+)\s+billion', r'$\1 billion', text)
        
        # Fix missing dollar signs
        text = re.sub(r'(\d+)\s*(million|billion)', r'$\1 \2', text)
        
        # Remove repeated phrases
        sentences = re.split(r'(?<=[.!?])\s+', text)
        unique_sentences = []
        seen = set()
        for sent in sentences:
            # Check first 40 chars for duplicates (ignoring case)
            sent_lower = sent.lower()[:40]
            if sent_lower not in seen:
                unique_sentences.append(sent)
                seen.add(sent_lower)
        
        # If we removed too many, keep at least the first and last
        if len(unique_sentences) < 2 and len(sentences) > 2:
            unique_sentences = [sentences[0], sentences[-1]]
        elif len(unique_sentences) == 0 and sentences:
            unique_sentences = [sentences[0]]
        
        return ' '.join(unique_sentences)
    
    def _coherence_score(self, text: str) -> float:
        """Calculate coherence score based on sentence length variance."""
        sentences = re.split(r'[.!?]', text)
        lengths = [len(s.split()) for s in sentences if s.strip()]
        if not lengths:
            return 0
        # Return average sentence length as simple coherence metric
        return np.mean(lengths)
    
    def _refine_summary(self, text: str) -> str:
        """⭐ 2️⃣ Pegasus-style refinement stage using BART."""
        if not self.summarizer:
            return text
        
        try:
            prompt = "Refine and compress this summary into a more abstract, coherent version:"
            refined = self._abstractive_summarize(
                text,
                prompt=prompt,
                max_length=120,
                min_length=60
            )
            return refined
        except Exception as e:
            print(f"⚠️ Refinement error: {e}")
            return text
    
    def _abstractive_summarize(self, text: str, prompt: str = "", max_length: int = 150, min_length: int = 50) -> str:
        """Generate proper abstractive summary using BART with GPT-style enhancements."""
        if not self.summarizer:
            return self._extractive_summarize(text)
        
        try:
            text = self._clean_whisper_transcript(text)
            
            # Combine prompt and text if prompt provided
            if prompt:
                full_text = prompt + " " + text
            else:
                # Default prompt for abstraction
                full_text = (
                    "Paraphrase aggressively and summarize conceptually without copying sentences. "
                    "Fuse ideas, remove redundancy, and produce a coherent narrative. "
                    "Compress meaning, not sentences. Remove repetition and synthesize ideas: " + text
                )
            
            # ⭐ token control
            if self.tokenizer:
                tokens = self.tokenizer.encode(full_text, truncation=True, max_length=1024)
                full_text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            
            summary = self.summarizer(
                full_text,
                max_length=max_length,
                min_length=min_length,
                do_sample=True,
                temperature=0.7,  # Lower temperature for more coherent output
                top_p=0.92,
                top_k=40,
                num_beams=8,  # More beams for better quality
                repetition_penalty=1.5,  # Higher penalty to avoid repeats
                length_penalty=2.0,
                no_repeat_ngram_size=3,
                early_stopping=True
            )[0]["summary_text"]
            
            # ⭐ Post-processing fixes
            summary = self._fix_common_errors(summary)
            
            # ✅ 6️⃣ HALLUCINATION GUARD UPGRADE
            overlap = len(set(summary.lower().split()) & set(text.lower().split())) / max(1, len(summary.split()))
            if overlap < 0.15:
                print("⚠️ Hallucination detected → fallback extractive")
                summary = self._extractive_summarize(text)
            
            summary = re.sub(r'\s+', ' ', summary).strip()
            if summary and not summary.endswith(('.', '!', '?')):
                summary += '.'
            if summary:
                summary = summary[0].upper() + summary[1:]
            
            return summary
            
        except Exception as e:
            print(f"⚠️ Abstractive summarization error: {e}")
            return self._extractive_summarize(text)
    
    def _extractive_summarize(self, text: str, num_sentences: int = 5) -> str:
        """Fallback extractive summarization."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) <= num_sentences:
            return text
        
        # Take first 2, last 1, and important middle sentences
        important_sentences = []
        important_sentences.extend(sentences[:2])  # First 2
        
        if len(sentences) > 3:
            important_sentences.append(sentences[-1])  # Last 1
        
        # Look for important sentences
        importance_keywords = ['important', 'key', 'critical', 'main', 'conclusion', 
                              'summary', 'therefore', 'result', 'significant']
        
        for sent in sentences[2:-1]:
            if len(important_sentences) >= num_sentences:
                break
            if any(kw in sent.lower() for kw in importance_keywords):
                important_sentences.append(sent)
        
        # If we need more, add middle sentences
        while len(important_sentences) < num_sentences and len(sentences) > len(important_sentences):
            important_sentences.append(sentences[len(important_sentences)])
        
        return ' '.join(important_sentences)
    
    def _generate_abstractive_key_points(self, text: str, num_points: int = 5) -> List[str]:
        """Generate key points using improved prompt for useful takeaways."""
        if not self.summarizer:
            return self._extractive_key_points(text, num_points)
        
        try:
            # Split text into sentences
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            if len(sentences) < num_points:
                # Text is short, just summarize it
                prompt = "Summarize the key points from this text:"
                summary = self._abstractive_summarize(
                    text,
                    prompt=prompt,
                    max_length=100, 
                    min_length=50
                )
                # Split into sentences for key points
                points = re.split(r'(?<=[.!?])\s+', summary)
                return [p for p in points if len(p.split()) > 3][:num_points]
            
            # Get key points from different parts of the text
            points = []
            
            # Calculate segment sizes
            segment_size = max(1, len(sentences) // num_points)
            
            for i in range(num_points):
                start_idx = i * segment_size
                end_idx = min((i + 1) * segment_size, len(sentences))
                
                if start_idx < len(sentences):
                    segment_text = ' '.join(sentences[start_idx:end_idx])
                    if segment_text:
                        # ⭐ 8️⃣ GPT-style key point generator - IMPROVED PROMPT
                        point_prompt = "List the most useful takeaways a viewer should remember:"
                        point = self._abstractive_summarize(
                            segment_text,
                            prompt=point_prompt,
                            max_length=40,
                            min_length=15
                        )
                        if point and len(point.split()) > 3:
                            points.append(point)
            
            # If we don't have enough points, generate more from the beginning
            while len(points) < num_points:
                additional_text = ' '.join(sentences[:10])
                point = self._abstractive_summarize(
                    additional_text,
                    prompt=point_prompt,
                    max_length=40,
                    min_length=15
                )
                if point and point not in points:
                    points.append(point)
                else:
                    # Avoid infinite loop
                    points.append(f"Key insight {len(points) + 1}")
            
            return points[:num_points]
            
        except Exception as e:
            print(f"⚠️ Key points generation error: {e}")
            return self._extractive_key_points(text, num_points)
    
    def _extractive_key_points(self, text: str, num_points: int = 5) -> List[str]:
        """Fallback extractive key points."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Score sentences
        scores = []
        for i, sent in enumerate(sentences):
            score = 0
            
            # Position score
            if i < len(sentences) * 0.2:  # First 20%
                score += 3
            elif i > len(sentences) * 0.8:  # Last 20%
                score += 2
            
            # Keyword score
            keywords = ['important', 'key', 'critical', 'main', 'conclusion', 
                       'summary', 'result', 'significant', 'essential']
            for kw in keywords:
                if kw in sent.lower():
                    score += 2
            
            # Length score
            word_count = len(sent.split())
            if 8 <= word_count <= 25:
                score += 1
            
            scores.append(score)
        
        # Get top sentences
        top_indices = np.argsort(scores)[-num_points:]
        top_indices = sorted(top_indices)
        
        return [sentences[i] for i in top_indices]
    
    # ✅ 3️⃣ KEY POINT FILTER (REMOVE FLUFF + CTA + GENERIC LINES)
    def _filter_low_value_points(self, points: List[str]) -> List[str]:
        """Remove low-value, promotional, or generic key points."""
        blacklist = [
            "subscribe",
            "like this video",
            "thank you",
            "see you",
            "comment below",
            "follow",
            "channel",
            "bell icon",
            "next video",
            "watch",
            "thanks for",
            "please",
            "don't forget"
        ]

        filtered = []
        for p in points:
            p_lower = p.lower()
            # Check if point contains any blacklisted terms
            if not any(b in p_lower for b in blacklist):
                # Also ensure point has meaningful length
                if len(p.split()) > 5:
                    filtered.append(p)

        # If filtering removed everything, return original points (safety)
        if not filtered and points:
            return points[:3]
        
        return filtered
    
    def _generate_intelligent_recommendations(self, text: str, topics: List[str], duration_minutes: float) -> List[str]:
        """Generate intelligent recommendations based on content analysis."""
        recommendations = []
        
        # Recommendation based on video length
        if duration_minutes > 20:
            recommendations.append(
                f"This {int(duration_minutes)}-minute video provides comprehensive coverage. " +
                "Consider watching in segments and taking notes for better retention."
            )
        else:
            recommendations.append(
                f"This concise {int(duration_minutes)}-minute video is perfect for a quick understanding " +
                "of the core concepts. Watch it in one sitting for maximum impact."
            )
        
        # Topic-based recommendations
        if topics and len(topics) >= 3:
            topic_text = ', '.join(topics[:3])
            recommendations.append(
                f"If you're specifically interested in {topic_text}, this video offers " +
                "focused insights on these areas."
            )
        elif topics:
            recommendations.append(
                f"For those wanting to learn about {topics[0]}, this video serves as " +
                "an excellent starting point."
            )
        
        # Learning path recommendation
        recommendations.append(
            "To deepen your understanding, consider exploring these related topics: " +
            "practical applications, real-world case studies, and advanced concepts " +
            "in this domain."
        )
        
        # Content engagement recommendation
        if len(text.split()) > 2000:
            recommendations.append(
                "This video contains detailed information. We recommend pausing at key " +
                "points to reflect on the concepts and take notes."
            )
        else:
            recommendations.append(
                "After watching, try to explain the key concepts to someone else or " +
                "apply them to a practical scenario to reinforce your learning."
            )
        
        return recommendations[:4]
    
    def _extract_topics_embeddings(self, chunks: List[Dict]) -> List[str]:
        """Extract topics using embeddings."""
        if not self.embedding_model or not chunks:
            return self._extract_topics_keywords(chunks)
        
        try:
            # Get chunk texts
            chunk_texts = [chunk.get('text', '') for chunk in chunks if chunk.get('text')]
            
            if not chunk_texts:
                return []
            
            # Generate embeddings for first sentence of each chunk
            first_sentences = []
            for text in chunk_texts:
                sentences = text.split('.')[:2]
                first_sentences.append('. '.join(sentences))
            
            if len(first_sentences) < 2:
                return self._extract_topics_keywords(chunks)
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(first_sentences, convert_to_tensor=True)
            
            # Find representative topics using clustering
            topics = []
            used = set()
            
            for i in range(len(first_sentences)):
                if i in used or len(topics) >= 5:
                    continue
                
                # Find similar chunks
                cluster = [i]
                for j in range(i+1, len(first_sentences)):
                    if j in used:
                        continue
                    similarity = util.pytorch_cos_sim(embeddings[i], embeddings[j])
                    if similarity > 0.6:
                        cluster.append(j)
                
                # Use the first sentence of the first chunk in cluster as topic
                topic_text = first_sentences[cluster[0]]
                # Clean up topic
                topic_text = re.sub(r'\s+', ' ', topic_text).strip()
                if len(topic_text.split()) > 8:
                    topic_text = ' '.join(topic_text.split()[:6]) + '...'
                
                topics.append(topic_text)
                used.update(cluster)
            
            return topics[:5]
            
        except Exception as e:
            print(f"Embedding topic extraction error: {e}")
            return self._extract_topics_keywords(chunks)
    
    def _extract_topics_keywords(self, chunks: List[Dict]) -> List[str]:
        """Extract topics using keyword frequency."""
        word_freq = defaultdict(int)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'by', 'is', 'are', 'was', 'were', 'this', 'that', 'it',
                     'from', 'as', 'be', 'have', 'has', 'had', 'will', 'would', 'could',
                     'should', 'can', 'may', 'might', 'than', 'then', 'there', 'their',
                     'they', 'them', 'these', 'those', 'what', 'which', 'who', 'whom',
                     'when', 'where', 'why', 'how', 'about', 'into', 'during', 'before',
                     'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under'}
        
        for chunk in chunks:
            text = chunk.get('text', '').lower()
            words = re.findall(r'\b[a-z]{4,}\b', text)  # Words with 4+ letters
            for word in words:
                if word not in stop_words:
                    word_freq[word] += 1
        
        # Get top words and create phrases
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:8]
        
        # Group related words (simple approach - just use top words)
        topics = []
        for word, count in top_words[:5]:
            if count > 1:  # Only include if appears multiple times
                topics.append(word.capitalize())
        
        return topics if topics else ["General Topic"]
    
    def _identify_sections(self, chunks: List[Dict]) -> List[Dict]:
        """Group chunks into coherent sections."""
        if len(chunks) <= 2:
            return [{
                'title': 'Complete Video',
                'chunk_indices': list(range(len(chunks))),
                'start_time': chunks[0].get('start_time'),
                'end_time': chunks[-1].get('end_time')
            }]
        
        sections = []
        current_indices = []
        section_start_time = chunks[0].get('start_time')
        
        for i, chunk in enumerate(chunks):
            text = chunk.get('text', '').lower()
            
            # Detect section boundaries
            is_section_start = any(indicator in text for indicator in [
                'chapter', 'section', 'part', 'module', 'lesson',
                'introduction', 'conclusion', 'summary', 'overview',
                'next', 'moving on', 'finally', 'first:', 'second:',
                'let\'s talk about', 'now we\'ll discuss'
            ])
            
            # Create new section every 3-4 chunks for coherence
            if (is_section_start or len(current_indices) >= 3) and len(current_indices) >= 2:
                if current_indices:
                    sections.append({
                        'title': f"Section {len(sections) + 1}",
                        'chunk_indices': current_indices,
                        'start_time': section_start_time,
                        'end_time': chunks[current_indices[-1]].get('end_time')
                    })
                
                current_indices = [i]
                section_start_time = chunk.get('start_time')
            else:
                current_indices.append(i)
        
        # Add final section
        if current_indices:
            sections.append({
                'title': f"Section {len(sections) + 1}",
                'chunk_indices': current_indices,
                'start_time': section_start_time,
                'end_time': chunks[current_indices[-1]].get('end_time')
            })
        
        return sections
    
    def _deduplicate_points(self, points: List[Dict]) -> List[Dict]:
        """Remove duplicate or very similar points."""
        if not points:
            return []
        
        unique = []
        seen = set()
        
        for point in points:
            text = point.get('point', '').lower().strip()
            text = re.sub(r'\s+', ' ', text)
            
            # Check if similar to existing
            is_duplicate = False
            for existing in seen:
                # Check for substantial overlap
                words1 = set(text.split())
                words2 = set(existing.split())
                if words1 and words2:
                    overlap = len(words1 & words2) / max(len(words1), len(words2))
                    if overlap > 0.6:
                        is_duplicate = True
                        break
            
            if not is_duplicate and len(text.split()) > 3:
                unique.append(point)
                seen.add(text)
        
        return unique
    
    def _watch_decision(self, text: str) -> str:
        """⭐ NEW: Determine watch decision based on content length."""
        wc = len(text.split())
        if wc > 2000:
            return "Best for deep learning viewers"
        elif wc > 800:
            return "Good for intermediate understanding"
        else:
            return "Quick overview video"
    
    def generate_hierarchical_summaries(self, chunks: List[Dict], duration_minutes: float = 0) -> Dict:
        """
        Generate true abstractive summaries at multiple levels.
        
        Args:
            chunks: List of chunk dicts from smart_chunker
            duration_minutes: Video duration in minutes
            
        Returns:
            Dictionary with multi-level abstractive summaries
        """
        if not chunks:
            return self._get_empty_response()
        
        print("🎯 Generating TRUE abstractive summaries with BART (GPT-style enhanced)...")
        
        # ⭐ 3️⃣ Semantic clustering before summarization
        if self.embedding_model and len(chunks) > 3:
            print("📊 Performing semantic clustering to group related chunks...")
            try:
                texts = [c["text"] for c in chunks if c.get("text")]
                if len(texts) >= len(chunks):  # Ensure all chunks have text
                    emb = self.embedding_model.encode(texts, convert_to_tensor=True)
                    sim = util.pytorch_cos_sim(emb, emb)
                    
                    # Reorder chunks by semantic similarity
                    order = np.argsort(-sim.sum(axis=1).cpu().numpy())
                    chunks = [chunks[i] for i in order]
                    print("✅ Semantic clustering complete")
            except Exception as e:
                print(f"⚠️ Semantic clustering error: {e}")
        
        print(f"📊 Level 1: Abstractively summarizing {len(chunks)} chunks...")
        
        chunk_summaries = []
        all_key_points = []
        full_text_parts = []
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.get('text', '')
            if not chunk_text or len(chunk_text.split()) < 20:
                continue
            
            full_text_parts.append(chunk_text)
            
            try:
                # 🔗 SEMANTIC OVERLAP IMPROVEMENT (chunk boundary fix)
                if i > 0:
                    chunk_text = chunks[i-1].get("text","")[-150:] + " " + chunk_text
                
                # ⭐ 6️⃣ Adaptive chunk size logic
                wc = len(chunk_text.split())
                if wc < 80:
                    max_l, min_l = 50, 20
                elif wc < 200:
                    max_l, min_l = 80, 30
                else:
                    max_l, min_l = 120, 50
                
                # Generate abstractive summary for chunk with adaptive lengths
                chunk_summary = self._abstractive_summarize(
                    chunk_text,
                    max_length=max_l,
                    min_length=min_l
                )
                
                # ⭐ 4️⃣ Coherence scorer filter
                if self._coherence_score(chunk_summary) < 5:
                    print(f"⚠️ Chunk {i} summary lacks coherence, using extractive fallback")
                    chunk_summary = self._extractive_summarize(chunk_text)
                
                chunk_summaries.append({
                    'chunk_id': i,
                    'summary': chunk_summary,
                    'word_count': len(chunk_text.split()),
                    'start_time': chunk.get('start_time'),
                    'end_time': chunk.get('end_time')
                })
                
                # Generate key points for this chunk - using improved prompt
                key_points = self._generate_abstractive_key_points(chunk_text, num_points=2)
                for j, point in enumerate(key_points):
                    all_key_points.append({
                        'point': point,
                        'timestamp': chunk.get('start_time'),
                        'chunk_id': i
                    })
                
            except Exception as e:
                print(f"⚠️ Chunk {i} summarization error: {e}")
                continue
        
        if not chunk_summaries:
            return self._get_empty_response()
        
        # Get full text for watch decision and value summary
        full_text = ' '.join(full_text_parts)
        
        # Level 2: Group into sections
        print("📊 Level 2: Identifying sections and generating abstractive section summaries...")
        sections = self._identify_sections(chunks)
        
        # Generate section summaries
        section_summaries = []
        for section in sections:
            section_text = ' '.join([
                chunks[i].get('text', '') for i in section['chunk_indices'] 
                if i < len(chunks)
            ])
            if section_text:
                section_prompt = "Summarize this section's main points concisely:"
                section_summary = self._abstractive_summarize(
                    section_text[:3000],
                    prompt=section_prompt,
                    max_length=120,
                    min_length=50
                )
                
                section_summaries.append({
                    'section_title': section['title'],
                    'summary': section_summary,
                    'start_time': section.get('start_time'),
                    'end_time': section.get('end_time')
                })
        
        # Level 3: Generate final summary
        print("📊 Level 3: Generating final abstractive summary with synthesized insights...")
        
        # ⭐ 7️⃣ Multi-stage hierarchical summarization upgrade
        unique = list(dict.fromkeys([s['summary'] for s in chunk_summaries]))
        
        # ✅ 5️⃣ STRONGER ABSTRACTION PROMPT (CRITICAL)
        final_prompt = (
            "Create a decision-oriented executive briefing explaining:\n"
            "• what the video teaches\n"
            "• core insights\n"
            "• practical takeaways\n"
            "Avoid repetition and avoid copying transcript sentences."
        )
        combined_text = ' '.join(unique)
        
        final_summary = self._abstractive_summarize(
            combined_text,
            prompt=final_prompt,
            max_length=250,
            min_length=120
        )
        
        # ⭐ 2️⃣ Apply Pegasus-style refinement
        print("📊 Level 4: Refining final summary for better abstraction...")
        final_summary = self._refine_summary(final_summary)
        
        # ✅ 4️⃣ REPETITION CONTROL FOR FINAL SUMMARY
        final_summary = self._fix_common_errors(final_summary)
        final_summary = re.sub(r'\b(\w+)( \1\b)+', r'\1', final_summary)
        
        # ⭐ NEW: Generate value summary (what user gains)
        print("📊 Level 5: Generating value summary...")
        value_prompt = "Explain what a viewer will gain after watching this video:"
        value_summary = self._abstractive_summarize(
            full_text[:1500],
            prompt=value_prompt,
            max_length=80,
            min_length=40
        )
        
        # Extract topics
        topics = self._extract_topics_embeddings(chunks)
        
        # Deduplicate key points
        unique_key_points = self._deduplicate_points(all_key_points)
        
        # Generate recommendations
        recommendations = self._generate_intelligent_recommendations(
            full_text, topics, duration_minutes
        )
        
        # ⭐ NEW: Get watch decision
        watch_decision = self._watch_decision(full_text)
        
        # Prepare final response with new fields and filtering
        result = {
            'short_summary': final_summary,
            'detailed_summary': final_summary,
            'section_summaries': section_summaries[:3],
            'chunk_summaries': chunk_summaries[:5],
            # ✅ 3️⃣ KEY POINT FILTER applied here
            'key_points': self._filter_low_value_points([p['point'] for p in unique_key_points[:8]]),
            'key_points_with_timestamps': unique_key_points[:8],
            'topics_covered': topics,
            'recommendations': recommendations,
            'watch_decision': watch_decision,  # NEW FIELD
            'value_summary': value_summary,    # NEW FIELD
            'processing_method': 'hierarchical_abstractive_with_clustering',
            'chunks_processed': len(chunk_summaries)
        }
        
        return result
    
    def _get_empty_response(self) -> Dict:
        """Return empty response structure."""
        return {
            'short_summary': 'No content available to summarize.',
            'detailed_summary': 'The video transcript could not be processed for summarization.',
            'section_summaries': [],
            'chunk_summaries': [],
            'key_points': ['No key points could be extracted.'],
            'key_points_with_timestamps': [],
            'topics_covered': ['General'],
            'recommendations': [
                'Try a different video with clearer content.',
                'Ensure the video has spoken English content.'
            ],
            'watch_decision': 'No content to analyze',
            'value_summary': 'No value summary available',
            'processing_method': 'hierarchical_abstractive',
            'chunks_processed': 0
        }