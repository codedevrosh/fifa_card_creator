##  Project Description
FIFA Card Creator is a machine learning and Streamlit-based web application that allows users to generate custom FIFA-style player cards.  
The app predicts:
- Player **role**
- Player **overall rating**
- Card **rarity (Gold / Silver / Bronze / Icon)**

##  Project Structure
```
fifa_card_creator/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Dependencies for deployment
│
├── models/                         # Models hosted on Hugging Face
│ └── README.md                     # Info + Hugging Face link
│
├── notebooks/                      # Jupyter Notebooks used for EDA & Training
│  └── pipeline_fifa.ipynb          # Model training + preprocessing pipeline
│  └── fifa20_cluster.ipynb         # FIFA player clustering analysis
│
└── README.md                       # Project documentation
```
##  Technologies Used

| Category | Tools |
|----------|-----------------------------|
| Web Framework | Streamlit |
| ML Models | Scikit-Learn |
| Hosting | Hugging Face Hub |
| Deployment | Streamlit Cloud (+ GitHub) |
| Language | Python |
| Image Processing | PIL (Pillow) |
| Data Handling | NumPy, Pandas |
| Version Control | Git & GitHub |

##  Deployment
###  Deploy via Streamlit Cloud
```
1. Push project to GitHub
2. Go to https://share.streamlit.io
3. Click "New App"
4. Select:
   Repo: fifa_card_creator
   Branch: main
   Entry File: app.py
5. Click Deploy
```
## Conclusion
This project demonstrates how machine learning can be deployed in a real-world web application using Streamlit, Hugging Face, and GitHub CI/CD.
It provides an interactive FIFA card generator and serves as a great example of end-to-end ML deployment.

##  Author

Arockia Roshan A

Data Science Enthusiast | ML Developer
