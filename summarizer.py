from transformers import BartForConditionalGeneration, BartTokenizer
import os
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('punkt')

# Load BART model
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Ensure outputs folder exists
os.makedirs("outputs", exist_ok=True)

def chunk_text(text, max_tokens=900):
    """Split text into chunks under the model's token limit (approx words)."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_tokens):
        chunks.append(" ".join(words[i:i + max_tokens]))
    return chunks

def extract_keywords(text, top_n=15):
    """Simple keyword extraction based on frequency."""
    words = re.findall(r'\b\w+\b', text.lower())
    common_words = Counter(words).most_common(top_n*2)
    keywords = [word for word, _ in common_words if len(word) > 3]
    return list(dict.fromkeys(keywords))[:top_n]  # top_n unique keywords

def extract_top_sentences(text, top_k=5):
    """
    Lightweight extractive summarization:
    - Split into sentences
    - TF-IDF vectorize sentences
    - Score sentences by cosine similarity to document centroid vector
    - Return top_k sentences in original order
    """
    from nltk.tokenize import sent_tokenize
    sentences = sent_tokenize(text)
    if len(sentences) <= top_k:
        return sentences

    # Build TF-IDF for sentences
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    try:
        tfidf = vectorizer.fit_transform(sentences)
    except ValueError:
        return sentences[:top_k]

    # Convert centroid from np.matrix to np.array
    import numpy as np
    doc_centroid = np.asarray(tfidf.mean(axis=0))
    sims = cosine_similarity(tfidf, doc_centroid.reshape(1, -1)).flatten()

    ranked_idx = sims.argsort()[::-1][:top_k]
    ranked_idx_sorted = sorted(ranked_idx)  # keep original order
    top_sentences = [sentences[i] for i in ranked_idx_sorted]
    return top_sentences


def summarize_chunk(chunk, max_length=300):
    """Summarize a single text chunk with more detail."""
    inputs = tokenizer.batch_encode_plus([chunk], return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        length_penalty=1.5,
        max_length=max_length,
        min_length=80,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def generate_summary(text, chunk_max_len=500, final_max_len=900, extract_k=10):
    """
    Generate a long, detailed summary (~1000 words for long papers).
    """
    chunks = chunk_text(text)
    print(f"[INFO] Splitting into {len(chunks)} chunks for summarization...")

    partial_summaries = []
    for idx, chunk in enumerate(chunks):
        print(f"[INFO] Summarizing chunk {idx+1}/{len(chunks)}...")
        partial_summary = summarize_chunk(chunk, max_length=chunk_max_len)
        partial_summaries.append(partial_summary)

        # Save each partial summary
        with open(f"outputs/summary_chunk_{idx+1}.txt", "w", encoding="utf-8") as f:
            f.write(partial_summary)

    # Extractive top sentences from original full text
    top_sentences = extract_top_sentences(text, top_k=extract_k)
    extractive_block = " ".join(top_sentences)
    with open("outputs/extractive_block.txt", "w", encoding="utf-8") as f:
        f.write(extractive_block)

    # Merge partial summaries + extractive block
    combined_text = extractive_block + "\n\n" + " ".join(partial_summaries)

    # Final summarization step with much higher max_length
    print("[INFO] Running final abstractive summarization (long output)...")
    final_summary = summarize_chunk(combined_text, max_length=final_max_len)

    # Append keywords
    keywords = extract_keywords(text)
    keyword_str = "\n\n[Key Terms]: " + ", ".join(keywords)
    final_summary_with_keywords = final_summary + keyword_str

    # Save final summary
    with open("outputs/final_summary.txt", "w", encoding="utf-8") as f:
        f.write(final_summary_with_keywords)

    return final_summary_with_keywords

def generate_title(text, max_length=20):
    """Generate a short title from the full text (use BART directly)."""
    inputs = tokenizer.batch_encode_plus([text], return_tensors="pt", max_length=1024, truncation=True)
    title_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        length_penalty=2.0,
        max_length=max_length,
        min_length=5,
        early_stopping=True
    )
    title = tokenizer.decode(title_ids[0], skip_special_tokens=True)

    # Save title
    with open("outputs/final_title.txt", "w", encoding="utf-8") as f:
        f.write(title)

    return title
