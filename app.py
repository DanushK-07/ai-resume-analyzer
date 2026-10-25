import re
from io import BytesIO

import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide",
)

# Custom colorful website theme
st.markdown("""
<style>
/* Main page background */
.stApp {
    background: linear-gradient(135deg, #10172A 0%, #202A54 50%, #3B246A 100%);
    color: #F8FAFC;
}

/* Main content text */
.stApp, .stApp p, .stApp label,
.stApp li, .stApp span, .stApp div {
    color: #F8FAFC;
}

/* Main title */
h1 {
    color: #67E8F9 !important;
    font-weight: 800 !important;
    text-shadow: 0 0 18px rgba(103, 232, 249, 0.25);
}

/* Section headings */
h2, h3 {
    color: #C4B5FD !important;
    font-weight: 700 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #171E3A, #30204D);
    border-right: 1px solid #635BCE;
}

/* Text input and job description box */
.stTextArea textarea,
.stTextInput input {
    background-color: #18233F !important;
    color: #FFFFFF !important;
    border: 1px solid #818CF8 !important;
    border-radius: 12px !important;
    caret-color: #67E8F9;
}

/* Placeholder text */
.stTextArea textarea::placeholder,
.stTextInput input::placeholder {
    color: #B8C3D9 !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(99, 102, 241, 0.13);
    border: 1px dashed #A5B4FC;
    border-radius: 14px;
    padding: 14px;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(90deg, #6366F1, #A855F7) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    padding: 0.65rem 1rem !important;
    transition: 0.2s ease-in-out;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: linear-gradient(90deg, #4F46E5, #9333EA) !important;
    border: 1px solid #67E8F9 !important;
    transform: translateY(-2px);
}

/* Similarity and coverage metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #202F55, #34275E);
    border: 1px solid #6366F1;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 5px 20px rgba(0, 0, 0, 0.18);
}

[data-testid="stMetricLabel"] {
    color: #C4B5FD !important;
}

[data-testid="stMetricValue"] {
    color: #67E8F9 !important;
    font-weight: 800 !important;
}

/* Success, information and error messages */
[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Horizontal separators */
hr {
    border-color: #6366F1 !important;
}

/* Caption text */
.stCaption, [data-testid="stCaptionContainer"] {
    color: #CBD5E1 !important;
}

/* Checkbox and radio labels, if used */
.stCheckbox label, .stRadio label {
    color: #F8FAFC !important;
}

/* Download button spacing */
.stDownloadButton {
    margin-top: 12px;
}
</style>
""", unsafe_allow_html=True)

# A beginner-friendly skill list. Add more skills as needed.
SKILLS = [
    "python", "java", "c++", "c", "sql", "mysql", "postgresql",
    "machine learning", "deep learning", "artificial intelligence",
    "data analysis", "data visualization", "pandas", "numpy",
    "scikit-learn", "tensorflow", "pytorch", "keras", "opencv",
    "natural language processing", "nlp", "computer vision",
    "power bi", "tableau", "excel", "statistics", "html", "css",
    "javascript", "react", "node.js", "flask", "django", "streamlit",
    "git", "github", "linux", "aws", "azure", "docker",
    "communication", "teamwork", "problem solving",
]


def extract_text(uploaded_file):
    """Extract text from a PDF or TXT upload."""
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        reader = PdfReader(BytesIO(uploaded_file.getvalue()))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages)

    # TXT files may be UTF-8 or have a different encoding.
    raw_bytes = uploaded_file.getvalue()
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return raw_bytes.decode("latin-1", errors="replace")


def find_skills(text):
    """Find known skills while avoiding partial-word matches."""
    text_lower = text.lower()
    found = []

    for skill in SKILLS:
        # Use word boundaries for simple skills; escape punctuation safely.
        pattern = r"(?<!\w)" + re.escape(skill.lower()) + r"(?!\w)"
        if re.search(pattern, text_lower):
            found.append(skill)

    return sorted(set(found))


def calculate_similarity(resume_text, job_text):
    """Return cosine similarity between the two pieces of text as a percentage."""
    documents = [resume_text, job_text]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(documents)
    score = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
    return round(float(score) * 100, 1)


st.title("📄 AI Resume Analyzer")
st.write(
    "Compare a resume with a job description, discover matching skills, "
    "and find areas you may want to improve."
)

with st.sidebar:
    st.header("About this project")
    st.write("Built with Python, Streamlit, PyPDF, and scikit-learn.")
    st.caption(
        "Your uploaded resume is processed by this app. This starter version "
        "does not intentionally save uploaded files."
    )

st.subheader("1. Upload your resume")
uploaded_file = st.file_uploader(
    "Choose a PDF or TXT resume",
    type=["pdf", "txt"],
    help="Scanned image-only PDFs may not contain extractable text.",
)

st.subheader("2. Paste the job description")
job_description = st.text_area(
    "Job description",
    height=220,
    placeholder=(
        "Example: We are looking for a Python developer with SQL, "
        "machine learning, pandas, and communication skills..."
    ),
)

analyze_clicked = st.button("Analyze Resume", type="primary", use_container_width=True)

if analyze_clicked:
    if uploaded_file is None:
        st.error("Please upload a PDF or TXT resume first.")
    elif not job_description.strip():
        st.error("Please paste a job description first.")
    else:
        try:
            resume_text = extract_text(uploaded_file)

            if not resume_text.strip():
                st.error(
                    "No readable text was found. If this is a scanned PDF, "
                    "try a text-based PDF or TXT file."
                )
            else:
                resume_skills = find_skills(resume_text)
                job_skills = find_skills(job_description)
                matched_skills = sorted(set(resume_skills) & set(job_skills))
                missing_skills = sorted(set(job_skills) - set(resume_skills))

                try:
                    similarity_score = calculate_similarity(
                        resume_text, job_description
                    )
                except ValueError:
                    similarity_score = 0.0

                if job_skills:
                    skill_score = round(
                        len(matched_skills) / len(job_skills) * 100, 1
                    )
                else:
                    skill_score = None

                st.divider()
                st.header("3. Analysis results")

                col1, col2 = st.columns(2)
                col1.metric("Text similarity", f"{similarity_score}%")
                if skill_score is not None:
                    col2.metric("Job skill coverage", f"{skill_score}%")
                else:
                    col2.metric("Job skill coverage", "N/A")

                st.caption(
                    "Text similarity compares the wording in the resume and job "
                    "description. Skill coverage is the percentage of detected "
                    "job-description skills also found in the resume. Neither "
                    "score is a hiring probability."
                )

                left, right = st.columns(2)
                with left:
                    st.subheader("✅ Matching skills")
                    if matched_skills:
                        for skill in matched_skills:
                            st.write(f"- {skill}")
                    else:
                        st.info("No skills from the built-in list matched yet.")

                with right:
                    st.subheader("🎯 Skills to consider adding")
                    if missing_skills:
                        for skill in missing_skills:
                            st.write(f"- {skill}")
                    elif job_skills:
                        st.success("All detected job skills were found in the resume.")
                    else:
                        st.info(
                            "No skills from the built-in list were detected "
                            "in the job description."
                        )

                st.subheader("Skills detected in the resume")
                if resume_skills:
                    st.write(", ".join(resume_skills))
                else:
                    st.write("No skills from the built-in list were detected.")

                suggestions = []
                if missing_skills:
                    suggestions.append(
                        "Review the missing-skill list. Add a skill to your resume "
                        "only if you genuinely have that skill or experience."
                    )
                if similarity_score < 30:
                    suggestions.append(
                        "Tailor the resume summary and project descriptions to "
                        "the job requirements, using accurate and relevant wording."
                    )
                if not re.search(r"\b(project|projects)\b", resume_text, re.I):
                    suggestions.append(
                        "If relevant, include a Projects section with your role, "
                        "tools used, and measurable outcomes."
                    )
                if not suggestions:
                    suggestions.append(
                        "Keep your resume accurate and concise, and verify that "
                        "each relevant skill is supported by real experience."
                    )

                st.subheader("Improvement suggestions")
                for suggestion in suggestions:
                    st.write(f"- {suggestion}")

                report = f"""AI RESUME ANALYZER REPORT
========================
Text similarity: {similarity_score}%
Job skill coverage: {skill_score if skill_score is not None else 'N/A'}%

Matching skills:
{', '.join(matched_skills) if matched_skills else 'None detected'}

Skills to consider adding:
{', '.join(missing_skills) if missing_skills else 'None detected'}

Skills detected in resume:
{', '.join(resume_skills) if resume_skills else 'None detected'}

Improvement suggestions:
{chr(10).join('- ' + item for item in suggestions)}
"""
                st.download_button(
                    "Download analysis report (.txt)",
                    data=report,
                    file_name="resume_analysis_report.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

        except Exception as error:
            st.error(
                "Something went wrong while reading or analyzing the file. "
                "Please try another text-based PDF or TXT resume."
            )
            st.exception(error)

st.divider()
st.caption(
    "Educational project only. Results depend on readable text and the built-in "
    "skill list. Review the output yourself; it may miss skills or context."
)