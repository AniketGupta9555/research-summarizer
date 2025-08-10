from pdf_processor import extract_text_from_pdf
from text_cleaner import clean_text
from summarizer import generate_summary, generate_title

# Path to your research paper PDF
pdf_path = "sample_papers/Impact of  Doraemon  on adolescent development  a qualitative study of cognitive  moral  and cultural influences in Vietnamese teenagers.pdf"  # change to your file

# Step 1: Extract PDF text
raw_text = extract_text_from_pdf(pdf_path)
print(f"[INFO] Extracted {len(raw_text)} characters from PDF.")

# Step 2: Clean text
cleaned_text = clean_text(raw_text)
print(f"[INFO] Cleaned text length: {len(cleaned_text)} characters.")

# Step 3: Generate summary
summary = generate_summary(cleaned_text, chunk_max_len=500, final_max_len=900, extract_k=10)

print("\n=== Summary ===\n", summary)

# Step 4: Generate title
title = generate_title(cleaned_text)
print("\n=== Suggested Title ===\n", title)
