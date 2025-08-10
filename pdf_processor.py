import pdfplumber

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()

if __name__ == "__main__":
    pdf_file = "sample_papers/sample1.pdf"  # place a PDF in sample_papers/
    extracted_text = extract_text_from_pdf(pdf_file)
    print("Extracted Text:\n", extracted_text[:1000])  # print first 1000 chars
