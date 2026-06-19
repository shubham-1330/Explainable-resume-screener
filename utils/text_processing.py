"""
Text preprocessing utilities for resume and job description cleaning.
"""

import re
import string
from typing import List


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep important punctuation
    text = re.sub(r'[^\w\s.,;:()\-+#@/]', ' ', text)
    return text.strip()


def extract_emails(text: str) -> List[str]:
    """Extract email addresses from text."""
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.findall(pattern, text)


def extract_urls(text: str) -> List[str]:
    """Extract URLs from text."""
    pattern = r'https?://\S+|www\.\S+'
    return re.findall(pattern, text)


def extract_phone_numbers(text: str) -> List[str]:
    """Extract phone numbers from text."""
    pattern = r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]'
    return re.findall(pattern, text)


def extract_years(text: str) -> List[int]:
    """Extract years from text (e.g., 2019, 2020–2023)."""
    pattern = r'\b(19|20)\d{2}\b'
    years = re.findall(pattern, text)
    return [int(y) for y in years]


def calculate_experience_years(text: str) -> float:
    """Estimate years of experience from text."""
    years = extract_years(text)
    if len(years) >= 2:
        return max(years) - min(years)
    
    # Look for explicit mentions
    patterns = [
        r'(\d+)\+?\s*years?\s*of\s*experience',
        r'(\d+)\+?\s*years?\s*experience',
        r'(\d+)\+?\s*yrs?\s*exp',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1))
    
    return 0.0


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def extract_bullet_points(text: str) -> List[str]:
    """Extract bullet point items from text."""
    bullets = []
    for line in text.split('\n'):
        line = line.strip()
        if line.startswith(('-', '•', '*', '·', '–')):
            cleaned = line.lstrip('-•*·– ').strip()
            if cleaned:
                bullets.append(cleaned)
    return bullets
