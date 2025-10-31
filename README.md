# 🌿 MannMitra — AI-Powered Youth Wellness Companion  

MannMitra is an **AI-driven emotional wellness app** that helps youth reflect, grow, and build mental resilience through guided journaling, chat-based support, mindfulness exercises, and gamified motivation.  

Built with **Streamlit**, **Google Gemini AI**, and **SQLite**, it combines calm design with powerful mental wellness features — all in one compassionate digital space.  

---

## 🧠 Overview  

> 💬 "MannMitra" means *Mind Friend* — your gentle digital companion for self-reflection and daily mindfulness.  

The app helps users:  
- Reflect on their thoughts through **AI-guided conversations**.  
- Track **moods, streaks, and XP** to build positive habits.  
- Earn **badges** for consistency and growth.  
- Write daily **journals and affirmations**.  
- Relax with **guided breathing and music exercises**.  
- Access **helplines** if feeling distressed or unsafe.  

---

## ✨ Key Features  

### 💬 AI Chat Support  
- Powered by **Google Gemini (1.5/2.5 Flash)** API.  
- Two unique personas:  
  - 🪷 **Friendly Mode:** Warm, supportive, and emoji-filled.  
  - 🎓 **Mentor Mode:** Structured, professional, and focused.  
- Multilingual chat options — English, Hindi, Hinglish.  

### 🏅 XP, Streaks & Badges  
- Gain XP for activities like journaling, mood tracking, and daily visits.  
- Build streaks and unlock milestone badges (7, 14, 30 days, etc.).  
- Levels (XP-based): Level 1 to Level 5 (100–1500 XP).  
- All progress is stored locally in `mannmitra.db`.  

### 📔 Journal & Reflection Logs  
- Daily journaling prompts and affirmations.  
- Tracks WHO-5 wellbeing scores.  
- CSV logs stored securely in `/data/` folder.  

### 🧘 Guided Exercises  
- Calming music sessions with progress indicators.  
- Themes: Forest, Ocean, Piano, and Focus soundscapes.  

### 🔒 Stealth & Privacy Mode  
- One-click *Stealth Mode* hides all app content.  
- Unlock using a 4-digit PIN.  
- All passwords are securely stored using **bcrypt hashing**.  

### 🚨 Crisis Detection & Helplines  
- Automatically detects distress signals in chat (e.g. “I want to hurt myself”).  
- Displays national helpline numbers immediately.  
- Promotes help-seeking behavior with compassion.  

---

## 🏗️ Tech Stack  

| Component | Technology |
|------------|-------------|
| **Frontend / UI** | Streamlit (Custom CSS for modern UI) |
| **AI Engine** | Google Gemini 1.5 / 2.5 Flash |
| **Database** | SQLite3 |
| **Speech Input** | streamlit-mic-recorder |
| **Text-to-Speech** | gTTS + Browser Speech API |
| **Authentication** | bcrypt password hashing |
| **Environment Config** | python-dotenv |
| **Design** | GPT-style layout with gradient background |

---

## 📂 Project Structure  
~~~
MannMitra/
│
├── app.py # Main Streamlit app
├── badges_logs.py # XP, streaks, and badges logic
│
├── data/ # Local database & logs
│ ├── mannmitra.db
│ ├── users.json
│ ├── mood_log.csv
│ └── reflection_log.csv
│
├── content/ # Preloaded wellness content
│ ├── who5.json
│ └── helplines_in.json
│
├── assets/ # Audio tracks for relaxation
│ ├── calm_sleep1.mp3
│ ├── calm_sleep2.mp3
│ ├── calm_sleep3.mp3
│ └── exercise.mp3
│
├── requirements.txt # Python dependencies
└── README.md # Project documentation
~~~

---


## ⚙️ Installation  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/MICHELLE-HOOLGERI25/MannMitra.git
cd MannMitra
