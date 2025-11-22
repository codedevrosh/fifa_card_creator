# Model Files — Explanation

This folder contains the trained machine learning models used in the Streamlit app:

| File Name | Description | Format |
|-----------|-------------|--------|
| `classification_pipeline.pkl` | Predicts the player **role** (Attacker / Defender / Midfielder / Goalkeeper) based on input attributes |  pickle |
| `regression_pipeline.pkl`     | Predicts the **overall rating** of the player (0–99 scale) | pickle |

---

## ⚠ Important Notes

- Both models were trained using **Scikit-Learn Pipelines**.
- They contain custom preprocessing steps.  
  ⚠ If the custom classes are not available in the server, `joblib.load()` may fail.
- The models are hosted on **Hugging Face Hub** and downloaded during runtime:
  ```python
  file = hf_hub_download(
      repo_id="codedevrosh/fifa_card_creator",
      filename="models/classification_pipeline.pkl"
  )
  model = joblib.load(file)
