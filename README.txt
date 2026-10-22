CareerVista AI — Complete Starter Project
==========================================

Files
- app.py: Streamlit interface with user-selectable Dark/Light theme, resume upload/paste, job matching, charts, job exploration and record search/export.
- careervista_pk_starter_dataset.csv: 40 curated prototype records (not live vacancies).
- skill_aliases.csv: starter skill alias list.
- requirements.txt: Python dependencies.
- CareerVista_AI_Pakistan_Career_Guidance.ipynb: Google Colab notebook for preprocessing, baseline ML, MLP neural-network demonstration, evaluation and recommendations.

RUN LOCALLY
1. Install Python 3.10 or newer.
2. Extract this ZIP.
3. Open the extracted folder in VS Code.
4. Open Terminal in that folder.
5. Run: python -m venv .venv
6. Activate:
   Windows PowerShell: .\.venv\Scripts\Activate.ps1
   Windows CMD: .venv\Scripts\activate.bat
   macOS/Linux: source .venv/bin/activate
7. Run: pip install -r requirements.txt
8. Run: streamlit run app.py
9. Browser opens. If not, open http://localhost:8501

DEPLOY TO STREAMLIT COMMUNITY CLOUD
1. Push app.py, requirements.txt, CSV files and notebook to a GitHub repository.
2. Visit https://share.streamlit.io/ and sign in.
3. Create app, select repository/branch and app.py.
4. Deploy; test the public URL in a private/incognito window.
5. Never commit API keys or private resumes.

IMPORTANT LIMITATIONS
This starter dataset has 40 curated prototype records and 32 categories. It is not a representative sample of Pakistan's job market or live vacancy data. Category prediction may be unavailable for categories without enough examples. The app's match score is a ranking aid, not a hiring probability. The MLP in the notebook is an educational demonstration; report actual metrics and do not overclaim model performance. Check licenses/terms before using external datasets.

SUBMISSION CHECKLIST
- PDF/Word report with problem, objectives, dataset source/limitations, preprocessing, model methodology, evaluation metrics, screenshots, outputs, limitations and references.
- Code + dataset + requirements.txt (ZIP or GitHub repository).
- Public Streamlit URL, tested in incognito.
- Screen recording of the working app.
- LinkedIn post link and full name sent to coordinator.
- Email final submission to devsphere1.info@gmail.com.
