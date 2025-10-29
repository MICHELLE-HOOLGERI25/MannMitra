# 🌿 MannMitra – Youth Wellness App

**MannMitra** is a youth wellness and emotional support platform built with **Streamlit**.  
It provides a safe, calming, and interactive space for self-reflection, mood tracking, and positive affirmations — helping users take small, mindful steps toward better mental well-being.

---

## ✨ Key Features

### 💬 AI Chat Support
- Friendly & Mentor chat modes powered by **Google Gemini** (Generative AI)
- Voice input (mic-recorder) and Text-to-Speech responses for accessibility
- Crisis phrase detection with automatic helpline suggestions

### 🧘 Guided Exercises
- Breathing & mindfulness activities with relaxing background sounds  
- Visual timers and auditory feedback for deep focus  

### 📔 Daily Journal & Mood Tracker
- Record gratitude reflections, emotional states, and WHO-5 well-being scores  
- Auto-saves entries locally and builds a reflective diary

### 🎁 Affirmation Cards
- Modern, minimalist affirmation cards with soft pastel visuals  
- Refresh daily for a new positive message  

### 🏅 Badges & XP Logs
- Earn XP for journaling, consistency, and daily activities  
- Unlock badges for 7-, 14-, 30-, 60-, and 90-day streaks  
- Detailed logs of daily reflections, affirmations, and progress  

---

## 🛠️ Tech Stack

| Layer | Technology |
|--------|-------------|
| Frontend / UI | Streamlit |
| AI Engine | Google Gemini API |
| Database | SQLite |
| Voice & Audio | gTTS, `streamlit_mic_recorder` |
| Data Processing | pandas, numpy |
| Visualization | Streamlit Components |
| Deployment | Streamlit Cloud |
| Version Control | Git + GitHub |

---

## 📂 Project Structure

MannMitra/
│
├── app.py # Main Streamlit app
├── badges_logs.py # XP, badges, and logs
├── requirements.txt # Dependencies
├── README.md # Documentation
├── .env # API keys (not uploaded)
├── .gitignore # Ignored files
│
├── assets/ # Images, icons, and audio files
├── content/ # Text/JSON content (helplines, etc.)
└── data/ # Local DB and user data
├── mannmitra.db
├── mood_log.csv
├── gratitude.csv
└── users.json

yaml
Copy code

---


## 🚀 Run the App Locally

Clone this repo:
```bash
git clone https://github.com/MICHELLE-HOOLGERI25/MannMitra.git
cd MannMitra
