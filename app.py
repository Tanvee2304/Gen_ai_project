import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer

# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
except AttributeError:
    st.error("API key not found. Please create a .env file with your GOOGLE_API_KEY.")
    st.stop()

# --- Functions ---

def get_keywords(text):
    """Extracts top 15 keywords using TF-IDF."""
    if not text:
        return []
    vectorizer = TfidfVectorizer(stop_words='english', max_features=15)
    vectorizer.fit_transform([text])
    return vectorizer.get_feature_names_out()

def generate_content_from_gemini(job_desc, resume_text, keywords):
    """Generates tailored content using the Gemini API."""
    keyword_str = ", ".join(keywords)
    model = genai.GenerativeModel('models/gemini-2.5-flash')

    prompt = f"""
    You are an expert career coach and resume writer. Your task is to help a candidate tailor their resume and write a cover letter for a specific job.

    **Job Description:**
    {job_desc}

    **Candidate's Base Resume:**
    {resume_text}

    **Key Keywords to Include:**
    {keyword_str}

    **Instructions:**
    1.  **Rewrite Resume Bullet Points:** Analyze the candidate's resume and rewrite 3-5 of their most relevant bullet points to align perfectly with the job description. Integrate the keywords naturally. Frame the achievements using the STAR (Situation, Task, Action, Result) method where possible.
    2.  **Generate Cover Letter:** Write a concise, professional, and impactful three-paragraph cover letter. The letter should directly address the requirements in the job description and highlight the candidate's most relevant skills and experiences from their resume.

    **Output Format (Strictly follow this):**

    ### Tailored Resume Bullet Points:
    - [Rewritten Bullet Point 1]
    - [Rewritten Bullet Point 2]
    - [Rewritten Bullet Point 3]

    ---

    ### Draft Cover Letter:
    [Your Name]
    [Your Contact Info]

    [Date]

    [Hiring Manager Name/Title]
    [Company Name]
    [Company Address]

    Dear [Hiring Manager Name],

    [Paragraph 1: Introduction - State the position you're applying for and how you found it. Briefly mention your key qualifications.]

    [Paragraph 2: Body - Elaborate on your experience from the resume, connecting it directly to the key requirements from the job description and using the keywords.]

    [Paragraph 3: Closing - Reiterate your interest and enthusiasm for the role. Mention your attached resume and state your eagerness for an interview.]

    Sincerely,
    [Your Name]
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred: {str(e)}"

# --- Streamlit UI ---

st.set_page_config(page_title="AI Resume Tailor", layout="wide")
st.title("🚀 Gen AI Resume & Cover Letter Tailor")
st.write("Paste your resume and a job description below to get customized content.")

col1, col2 = st.columns(2)

with col1:
    st.header("Your Base Resume")
    resume_input = st.text_area("Paste your full resume here...", height=400, label_visibility="collapsed")

with col2:
    st.header("Job Description")
    job_desc_input = st.text_area("Paste the job description here...", height=400, label_visibility="collapsed")

if st.button("✨ Generate Tailored Content", type="primary", use_container_width=True):
    if not resume_input or not job_desc_input:
        st.warning("Please paste both your resume and the job description.")
    else:
        with st.spinner("Analyzing and generating content..."):
            keywords = get_keywords(job_desc_input)
            st.info(f"**Extracted Keywords:** {', '.join(keywords)}")
            
            generated_content = generate_content_from_gemini(job_desc_input, resume_input, keywords)
            
            st.divider()
            st.header("Your Customized Content")
            st.markdown(generated_content)