# 🚀 Worker Allocation System - Setup Guide

A complete guide for setting up and running the AI-powered Worker Allocation System.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Detailed Installation](#detailed-installation)
4. [Configuration](#configuration)
5. [Running the System](#running-the-system)
6. [Project Structure](#project-structure)
7. [API Endpoints](#api-endpoints)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- **Python 3.9+** (3.10 recommended)
- **pip** or **conda** for package management
- **Git** for version control
- **8GB RAM minimum** (16GB recommended for model training)

---

## Quick Start

```powershell
# 1. Clone/Navigate to the project
cd e:\Git\Test\ml

# 2. Create virtual environment (recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Prepare worker data and train models
python prepare_worker_data.py
python train_decision_tree.py

# 5. Configure email notifications (optional)
copy backend\.env.example backend\.env
# Edit .env with your SMTP credentials

# 6. Start the API server
cd backend
python ml_service.py

# 7. Open the frontend in browser
# Navigate to: frontend/worker_chatbot.html
```

---

## Detailed Installation

### Step 1: Set Up Python Environment

```powershell
# Create a virtual environment
python -m venv venv

# Activate it (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Or for Command Prompt
.\venv\Scripts\activate.bat
```

### Step 2: Install Dependencies

```powershell
# Install all required packages
pip install -r requirements.txt
```

> **GPU Support (Optional):** For faster training with NVIDIA GPU:
> ```powershell
> pip install torch --index-url https://download.pytorch.org/whl/cu118
> ```

### Step 3: Prepare Data & Train Models

```powershell
# Generate worker embeddings and processed data
python prepare_worker_data.py

# Train the decision tree classifier
python train_decision_tree.py
```

Expected output:
```
✅ Created worker_embeddings.npy
✅ Created tfidf_vectorizer.pkl
✅ Decision tree trained with accuracy: 0.95+
```

---

## Configuration

### Email Notifications Setup

To enable email notifications for task assignments:

1. Copy the example environment file:
   ```powershell
   copy backend\.env.example backend\.env
   ```

2. Edit `backend/.env` with your credentials:
   ```env
   # Gmail SMTP (use App Password, not regular password)
   SMTP_SERVER=smtp.gmail.com
   SMTP_PORT=587
   SMTP_USERNAME=your-email@gmail.com
   SMTP_PASSWORD=your-app-password
   FROM_EMAIL=your-email@gmail.com
   ```

> **Gmail Users:** Generate an App Password at:
> https://myaccount.google.com/apppasswords

### Optional: Push Notifications

For web push notifications, configure VAPID keys in `.env`:
```env
VAPID_PUBLIC_KEY=your-public-key
VAPID_PRIVATE_KEY=your-private-key
VAPID_EMAIL=mailto:admin@yourcompany.com
```

---

## Running the System

### Start the Backend API

```powershell
cd backend
python ml_service.py
```

The API will start at: **http://localhost:8000**

### Access the Frontend

Open in your browser:
- **Main Chatbot:** `frontend/worker_chatbot.html`
- **Worker Dashboard:** `frontend/worker_dashboard.html`
- **Appointment Page:** `frontend/appointment.html`

### API Documentation

Interactive API docs available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## Project Structure

```
ml/
├── backend/                    # API Server
│   ├── ml_service.py          # FastAPI main application
│   ├── notification_service.py # Email & push notifications
│   ├── task_db.py             # SQLite database for tasks
│   ├── requirements.txt       # Backend-only dependencies
│   ├── .env.example           # Environment template
│   └── .env                   # Your credentials (create this)
│
├── frontend/                   # Web UI
│   ├── worker_chatbot.html    # Main AI chat interface
│   ├── worker_dashboard.html  # Worker task management
│   └── appointment.html       # Booking confirmation page
│
├── data/                       # Source data
│   └── workers_db.csv         # Worker database
│
├── processed_data/            # Generated files
│   ├── worker_embeddings.npy  # Vector embeddings
│   ├── tfidf_vectorizer.pkl   # Text vectorizer
│   └── worker_scaler.pkl      # Feature scaler
│
├── models/                     # Trained models
│   └── decision_tree.pkl      # Suitability classifier
│
├── hybrid_recommender.py      # Core recommendation engine
├── prepare_worker_data.py     # Data preprocessing script
├── train_decision_tree.py     # Model training script
├── requirements.txt           # All dependencies
├── config.yaml               # System configuration
└── SETUP_GUIDE.md            # This file
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/health` | Detailed system status |
| POST | `/chat` | AI chatbot endpoint |
| POST | `/recommend-workers` | Get worker recommendations |
| GET | `/worker/{id}` | Get worker details |
| POST | `/assign-task` | Assign task to worker |
| GET | `/worker/{id}/tasks` | Get worker's tasks |
| PUT | `/task/{id}/status` | Update task status |
| GET | `/tasks` | List all tasks (admin) |

### Example: Chat Request

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I need an electrician with high voltage experience"}'
```

---

## Troubleshooting

### Common Issues

**1. "Worker recommender not loaded" error**
```powershell
# Run data preparation and training
python prepare_worker_data.py
python train_decision_tree.py
```

**2. Module not found errors**
```powershell
# Ensure virtual environment is active
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**3. Email notifications not working**
- Verify `.env` file exists in `backend/` folder
- Check SMTP credentials are correct
- For Gmail, use an App Password (not your regular password)

**4. CORS errors in browser**
- Ensure the backend is running on http://localhost:8000
- Check that `allow_origins=["*"]` is set in `ml_service.py`

**5. Port 8000 already in use**
```powershell
# Find and kill the process
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

---

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review API documentation at http://localhost:8000/docs
3. Check server logs for detailed error messages

---

*Last Updated: January 2026*
