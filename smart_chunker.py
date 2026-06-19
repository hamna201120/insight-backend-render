# smart_chunker.py
import re
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict

class SmartChunker:
    """
    Intelligently chunks long transcripts based on:
    - Topic shifts
    - Natural pauses
    - Content coherence
    - Context overlap for continuity
    """
    
    # ⭐ FIX 1: Added overlap parameter
    def __init__(self, max_chunk_size: int = 1000, min_chunk_size: int = 300, overlap: int = 40):
        self.max_chunk_size = max_chunk_size  # words
        self.min_chunk_size = min_chunk_size  # words
        self.overlap = overlap  # words of overlap between chunks
        self.topic_indicators = [
            'next', 'moving on', 'let\'s talk about', 'another', 'additionally',
            'finally', 'first', 'second', 'third', 'lastly', 'in conclusion',
            'now let\'s', 'the next topic', 'switching to', 'turning to',
            'introduce', 'chapter', 'section', 'part'
        ]
    
    def _detect_topic_boundaries(self, text: str) -> List[int]:
        """Detect natural topic boundaries in text."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        boundaries = []
        
        for i, sentence in enumerate(sentences):
            # Check for topic shift indicators
            if any(indicator in sentence.lower() for indicator in self.topic_indicators):
                # This sentence likely starts a new topic
                char_count = sum(len(s) for s in sentences[:i])
                boundaries.append(char_count)
        
        return boundaries
    
    def _calculate_importance_score(self, text: str) -> float:
        """Calculate importance score for a text segment."""
        words = text.lower().split()
        
        # Importance signals
        importance_signals = {
            'important': 3, 'key': 3, 'critical': 3, 'essential': 3,
            'main': 2, 'primary': 2, 'major': 2, 'significant': 2,
            'note': 1, 'remember': 1, 'crucial': 3, 'vital': 3,
            'conclusion': 3, 'summary': 3, 'overview': 2, 'introduction': 2
        }
        
        score = 1.0  # base score
        for word, weight in importance_signals.items():
            if word in words:
                score += weight
        
        return score
    
    def chunk_transcript(self, transcript: str, timestamps: List[Dict] = None) -> List[Dict]:
        """
        Intelligently chunk transcript based on content and timestamps.
        
        Args:
            transcript: Full transcript text
            timestamps: Optional list of {text, start, end} from Whisper
            
        Returns:
            List of chunks with metadata
        """
        # Clean transcript first
        transcript = re.sub(r'\s+', ' ', transcript).strip()
        
        # ⭐ FIX 4: Remove repetitive words
        transcript = re.sub(r'\b(\w+)( \1\b)+', r'\1', transcript)
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', transcript)
        
        # If timestamps are available, use them for smarter chunking
        if timestamps:
            chunks = self._chunk_with_timestamps(sentences, timestamps)
        else:
            # Otherwise, use semantic chunking
            chunks = self._chunk_by_semantics(sentences)
        
        # ⭐ OPTIONAL: Sort chunks by importance (research-level boost)
        chunks = sorted(chunks, key=lambda x: x.get("importance", 1), reverse=False)
        
        return chunks
    
    def _chunk_with_timestamps(self, sentences: List[str], timestamps: List[Dict]) -> List[Dict]:
        """Chunk using timestamp information from Whisper."""
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_start_time = timestamps[0]['start'] if timestamps else 0
        
        for i, sentence in enumerate(sentences):
            word_count = len(sentence.split())
            
            # If this sentence alone exceeds max size, split it
            if word_count > self.max_chunk_size:
                if current_chunk:
                    # ⭐ FIX 1 & 2: Add overlap and check minimum size
                    chunk_text, chunk_words = self._build_chunk_with_overlap(chunks, current_chunk)
                    if chunk_words >= 120:  # ⭐ FIX 2: Minimum semantic guard
                        importance = self._calculate_importance_score(chunk_text)
                        if importance >= 1.5:  # ⭐ FIX 5: Importance filter
                            chunks.append({
                                'text': chunk_text,
                                'word_count': chunk_words,
                                'start_time': chunk_start_time,
                                'end_time': timestamps[i-1]['end'] if i-1 < len(timestamps) else None,
                                'importance': importance
                            })
                
                # Split long sentence
                words = sentence.split()
                for j in range(0, len(words), self.max_chunk_size):
                    chunk_text = ' '.join(words[j:j + self.max_chunk_size])
                    chunk_words = len(chunk_text.split())
                    if chunk_words >= 120:  # ⭐ FIX 2: Minimum semantic guard
                        importance = self._calculate_importance_score(chunk_text)
                        if importance >= 1.5:  # ⭐ FIX 5: Importance filter
                            chunks.append({
                                'text': chunk_text,
                                'word_count': chunk_words,
                                'start_time': timestamps[i]['start'] if i < len(timestamps) else None,
                                'end_time': timestamps[i]['end'] if i < len(timestamps) else None,
                                'importance': importance
                            })
                current_chunk = []
                current_size = 0
                continue
            
            # If adding this sentence exceeds max size, create new chunk
            if current_size + word_count > self.max_chunk_size and current_size >= self.min_chunk_size:
                # ⭐ FIX 1 & 2: Build chunk with overlap and check minimum size
                chunk_text, chunk_words = self._build_chunk_with_overlap(chunks, current_chunk)
                if chunk_words >= 120:  # ⭐ FIX 2: Minimum semantic guard
                    importance = self._calculate_importance_score(chunk_text)
                    if importance >= 1.5:  # ⭐ FIX 5: Importance filter
                        chunks.append({
                            'text': chunk_text,
                            'word_count': chunk_words,
                            'start_time': chunk_start_time,
                            'end_time': timestamps[i-1]['end'] if i-1 < len(timestamps) else None,
                            'importance': importance
                        })
                
                # Start new chunk
                current_chunk = [sentence]
                current_size = word_count
                chunk_start_time = timestamps[i]['start'] if i < len(timestamps) else None
            else:
                # Add to current chunk
                current_chunk.append(sentence)
                current_size += word_count
        
        # Add final chunk
        if current_chunk:
            chunk_text, chunk_words = self._build_chunk_with_overlap(chunks, current_chunk)
            if chunk_words >= 120:  # ⭐ FIX 2: Minimum semantic guard
                importance = self._calculate_importance_score(chunk_text)
                if importance >= 1.5:  # ⭐ FIX 5: Importance filter
                    chunks.append({
                        'text': chunk_text,
                        'word_count': chunk_words,
                        'start_time': chunk_start_time,
                        'end_time': timestamps[-1]['end'] if timestamps else None,
                        'importance': importance
                    })
        
        return chunks
    
    def _chunk_by_semantics(self, sentences: List[str]) -> List[Dict]:
        """Chunk based on semantic coherence."""
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            word_count = len(sentence.split())
            
            # ⭐ FIX 3: Strengthened topic detection
            is_new_topic = (
                any(indicator in sentence.lower() for indicator in self.topic_indicators)
                and len(sentence.split()) > 8  # Prevents micro splits
            )
            
            # If new topic and current chunk is substantial, split
            if is_new_topic and current_size >= self.min_chunk_size:
                if current_chunk:
                    # ⭐ FIX 1 & 2: Build chunk with overlap and check minimum size
                    chunk_text, chunk_words = self._build_chunk_with_overlap(chunks, current_chunk)
                    if chunk_words >= 120:  # ⭐ FIX 2: Minimum semantic guard
                        importance = self._calculate_importance_score(chunk_text)
                        if importance >= 1.5:  # ⭐ FIX 5: Importance filter
                            chunks.append({
                                'text': chunk_text,
                                'word_count': chunk_words,
                                'importance': importance
                            })
                current_chunk = [sentence]
                current_size = word_count
            # If current chunk would exceed max size
            elif current_size + word_count > self.max_chunk_size and current_size >= self.min_chunk_size:
                if current_chunk:
                    # ⭐ FIX 1 & 2: Build chunk with overlap and check minimum size
                    chunk_text, chunk_words = self._build_chunk_with_overlap(chunks, current_chunk)
                    if chunk_words >= 120:  # ⭐ FIX 2: Minimum semantic guard
                        importance = self._calculate_importance_score(chunk_text)
                        if importance >= 1.5:  # ⭐ FIX 5: Importance filter
                            chunks.append({
                                'text': chunk_text,
                                'word_count': chunk_words,
                                'importance': importance
                            })
                current_chunk = [sentence]
                current_size = word_count
            else:
                current_chunk.append(sentence)
                current_size += word_count
        
        # Add final chunk
        if current_chunk:
            chunk_text, chunk_words = self._build_chunk_with_overlap(chunks, current_chunk)
            if chunk_words >= 120:  # ⭐ FIX 2: Minimum semantic guard
                importance = self._calculate_importance_score(chunk_text)
                if importance >= 1.5:  # ⭐ FIX 5: Importance filter
                    chunks.append({
                        'text': chunk_text,
                        'word_count': chunk_words,
                        'importance': importance
                    })
        
        return chunks
    
    def _build_chunk_with_overlap(self, chunks: List[Dict], current_chunk: List[str]) -> Tuple[str, int]:
        """
        Build chunk text with overlap from previous chunk.
        
        Args:
            chunks: List of already created chunks
            current_chunk: Current chunk sentences
            
        Returns:
            Tuple of (chunk_text, word_count)
        """
        # ⭐ FIX 1: Add overlap from previous chunk
        overlap_text = ""
        if chunks:
            prev_words = chunks[-1]["text"].split()
            overlap_text = " ".join(prev_words[-self.overlap:]) if len(prev_words) > self.overlap else " ".join(prev_words)
        
        chunk_text = overlap_text + " " + ' '.join(current_chunk) if overlap_text else ' '.join(current_chunk)
        chunk_text = chunk_text.strip()
        word_count = len(chunk_text.split())
        
        return chunk_text, word_count