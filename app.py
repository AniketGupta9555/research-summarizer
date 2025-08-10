import streamlit as st
import os
from pdf_processor import extract_text_from_pdf
from text_cleaner import clean_text
from summarizer import generate_summary, generate_title
import nltk

# Download required NLTK data at runtime if not present
nltk.download("punkt")
nltk.download("stopwords")


# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="📄 Research Paper Summarizer",
    page_icon="📚",
    layout="wide"
)

# ---------------------- STYLES: ANIMATED BACKGROUND + GLASS ----------------------
st.markdown(
    """
    <style>
    /* Animated gradient background */
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .stApp {
        background: linear-gradient(-45deg, #0f172a, #0b2545, #2b6fb8, #6fb1ff);
        background-size: 400% 400%;
        animation: gradientBG 18s ease infinite;
        color: #e6eef8;
    }

    /* Glass card */
    .glass-card {
        background: rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 18px;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid rgba(255,255,255,0.08);
    }

    /* Header styling */
    .header-title {
        font-size: 36px;
        font-weight: 700;
        color: #e6f2ff;
        text-align: center;
        margin-top: 6px;
        margin-bottom: 0;
    }
    .header-sub {
        text-align: center;
        color: #d0eaff;
        margin-top: 6px;
        margin-bottom: 20px;
    }

    /* File uploader label text */
    .stFileUploader label {
        color: #f8fbff !important;
        font-weight: 600;
    }

    /* File uploader description text */
    .stFileUploader div[data-testid="stFileUploaderDropzone"] p {
        color: #e0f3ff !important;
    }

    /* Info / tips text */
    .stAlert {
        background-color: rgba(255,255,255,0.15) !important;
        color: #f5fbff !important;
        font-weight: 500;
    }
    .stAlert p {
        color: #ffffff !important;
        font-weight: 500;
    }

    /* Buttons */
    .stDownloadButton>button, .stButton>button {
        background: linear-gradient(90deg,#4dd0e1,#1976d2) !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
    }

    /* small caption color */
    .caption, .stCaption, .stMarkdown p {
        color: #e2f5ff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------- HEADER ----------------------
st.markdown("<div class='glass-card'><h1 class='header-title'>📄 Automated Research Paper Summarizer</h1>"
            "<p class='header-sub'>Upload a research paper and get a detailed AI-generated summary, suggested title, and keywords.</p></div>",
            unsafe_allow_html=True)

# ---------------------- UPLOAD CONTAINER ----------------------
with st.container():
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📤 Upload Your PDF", type=["pdf"])
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- PROCESSING ----------------------
if uploaded_file:
    # save file
    os.makedirs("sample_papers", exist_ok=True)
    pdf_path = os.path.join("sample_papers", uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # extract
    with st.spinner("📄 Extracting text from PDF..."):
        raw_text = extract_text_from_pdf(pdf_path)
    st.success(f"✅ Extracted {len(raw_text)} characters from document.")

    # clean
    with st.spinner("🧹 Cleaning text..."):
        cleaned_text = clean_text(raw_text)
    st.info(f"🧾 Cleaned text length: {len(cleaned_text)} characters")

    # summarise (this may take time)
    with st.spinner("🧠 Generating summary (this can take a bit)..."):
        # tune these numbers if you want shorter/faster or longer/more detailed
        final_summary = generate_summary(cleaned_text, chunk_max_len=500, final_max_len=900, extract_k=10)

    # title
    with st.spinner("📝 Generating title..."):
        final_title = generate_title(cleaned_text)

    # ---------------------- OUTPUT ----------------------
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🏷 Suggested Title")
    st.markdown(f"<h3 style='color: #dff6ff'>{final_title}</h3>", unsafe_allow_html=True)

    st.subheader("📜 Detailed Summary")
    st.text_area("Summary", value=final_summary, height=360)

    word_count = len(final_summary.split())
    st.caption(f"📝 Word count: {word_count}", unsafe_allow_html=True)

    st.download_button(
        label="💾 Download Summary",
        data=final_summary,
        file_name="research_summary.txt",
        mime="text/plain"
    )
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("Upload a PDF to start. Tip: large papers will take longer to process.")

