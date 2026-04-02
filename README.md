# 🚀 JobShield AI

> **“Don’t apply. Verify first.”**

## 🎬 Demo Preview

![Demo](assets/demo.gif)

---

## 🧠 Problem

Fake job offers and internships are increasing rapidly, especially targeting:

- Students  
- Freshers  
- Job seekers  

Common scam patterns include:

- Registration fees  
- Fake HR contacts (WhatsApp/Telegram)  
- Unrealistic salary promises  

---

## 💡 Solution

JobShield AI analyzes job descriptions and provides:

- 📊 **Risk Score (0–100)**  
- 🚨 **Scam Verdict (Safe / Suspicious / Scam)**  
- 🧾 **Clear explanation of red flags**  

---

## ✨ Features

- 🔍 Smart job description analysis  
- 🧠 Scam pattern detection (rule-based + logic)  
- 📧 Email/domain verification  
- 📱 Phone number detection  
- 🖼️ Image upload support (simulated OCR)  
- ⚡ Fast, real-time results  
- 🧾 Human-readable explanations  
- 🛑 Invalid input detection  

---

## 🏗️ Project Structure

```bash
jobshield/
│
├── backend/
│   ├── main.py
│   ├── utils.py
│   ├── requirements.txt
│   └── data/
│       └── scam_dataset.csv
│
├── frontend/
│   ├── index.html
│   ├── styles.css
│   ├── script.js
│   └── config.json
│
└── README.md
```
⚙️ Tech Stack
Backend
Python
FastAPI
Frontend
HTML
CSS
JavaScript
Data
Scam dataset (GitHub sourced)
Rule-based + pattern detection
🚀 How to Run Locally
1️⃣ Clone the repository
git clone https://github.com/tanaznazneen/jobshield-ai.git
cd jobshield-ai
2️⃣ Setup Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

👉 Backend runs at:
http://127.0.0.1:8000

3️⃣ Run Frontend

Open:

frontend/index.html
🧪 How It Works
User pastes job description
System checks if input is job-related
Detects scam signals:
Payment requests
Informal contact methods
Unrealistic offers
Calculates risk score
Displays verdict with explanation
🎯 Use Cases
Students verifying internships
Job seekers checking offers
Awareness against online fraud
⚠️ Disclaimer

This tool provides risk analysis based on patterns and is not 100% accurate.

🏆 Hackathon Value
Solves real-world problem
Fast and interactive
Explainable AI approach

👥 Team

Scam Hunters

💬 Tagline

“From doubt to clarity — instantly.”
