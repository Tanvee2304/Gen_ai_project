# Gen_ai_project

Gen AI Resume & Cover Letter Tailoring Engine

A web application built with Streamlit and powered by the Google Gemini API that automates the process of customizing resumes and cover letters for specific job descriptions.


#### ✨ Key Features
- **Keyword Extraction:** Analyzes job descriptions using TF-IDF to identify the most critical skills and keywords.
- **AI-Powered Rewriting:** Leverages the Gemini API to rewrite resume bullet points, aligning them with the job's requirements and the STAR method.
- **Automated Cover Letter Generation:** Creates a professional, three-paragraph cover letter tailored to the role and the user's resume.
- **Time Efficiency:** Reduces the manual effort of tailoring job applications from 30+ minutes to under a minute, achieving a **95%+ reduction in time**.

#### 🛠️ Tech Stack
- **Language:** Python
- **Framework:** Streamlit
- **Gen AI:** Google Gemini API (`google-generativeai`), LangChain
- **NLP:** Scikit-learn (`TfidfVectorizer`)
