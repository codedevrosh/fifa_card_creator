# Model Files — Explanation

This folder contains the trained machine learning models used in the Streamlit app:

| File Name | Description | Format |
|-----------|-------------|--------|
| `classification_pipeline.pkl` | Predicts the player **role** (Attacker / Defender / Midfielder / Goalkeeper) based on input attributes |  pickle |
| `regression_pipeline.pkl`     | Predicts the **overall rating** of the player (0–99 scale) | pickle |

---

##  Important Notes

- Both models were trained using **Scikit-Learn Pipelines**.
- They contain custom preprocessing steps.  

##  Hugging Face Model Repository

All model files are hosted on Hugging Face and automatically downloaded in the Streamlit app:

 https://huggingface.co/codedevrosh/fifa_card_creator

The app uses this code to download models:
```python
file = hf_hub_download(
    repo_id="codedevrosh/fifa_card_creator",
    filename="models/classification_pipeline & regression_pipeline .pkl"
)
role_model = joblib.load(file)
