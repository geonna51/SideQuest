# 4300 Flask React Template

## Contents

- [Summary](#summary)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Deploying on the Server](#deploying-on-the-server)
- [Running Locally](#running-locally)
- [Troubleshooting Deployment Issues](#troubleshooting-deployment-issues)
- [Virtual Environments and Dependency Tracking](#virtual-environments-and-dependency-tracking)
- [Additional Resources](#additional-resources)
- [Feedback and Support](#feedback-and-support)

## Summary

This is a **Flask + React + TypeScript** template for **CS/INFO 4300 class at Cornell University**

This template combines:
- **Backend**: Flask REST API with SQLite database
- **Frontend**: React with TypeScript and Vite

You will use this template to directly deploy your Flask + React + TypeScript code on the project server.

After you follow the steps below, you should have set up a public address dedicated to your team's project at (for the moment) a template app will be running. In future milestones you will be updating the code to replace that template with your very own app.


## Quick Start

For the fastest way to get started with development:

### Windows
```bash
# 1. Set up Python virtual environment
python -m venv venv
venv\Scripts\activate

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Start Flask backend (in one terminal)
python src/app.py

# 4. In a NEW terminal, install and start React
cd frontend
npm install
npm run dev
```

### Mac/Linux
```bash
# 1. Set up Python virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Start Flask backend (in one terminal)
python src/app.py

# 4. In a NEW terminal, install and start React
cd frontend
npm install
npm run dev
```

Then open `http://localhost:5173` in your browser!

## Architecture

```
4300-Flask-React-Template/
├── src/
│   ├── app.py          # Flask app entry point
│   ├── models.py       # SQLAlchemy database models
│   ├── routes.py       # Search API routes (+ USE_LLM toggle)
│   ├── llm_routes.py   # LLM chat route (only used when USE_LLM = True)
│   └── init.json       # Seed data
├── frontend/
│   └── src/
│       ├── App.tsx     # Search UI (always shown)
│       ├── Chat.tsx    # AI chat component (rendered when USE_LLM = True)
│       ├── App.css
│       └── types.ts
├── requirements.txt
├── Dockerfile
└── .env                # SPARK_API_KEY goes here (not committed)
```

### Backend (Flask)
- **Entry point**: `src/app.py`
- **Database**: SQLite with SQLAlchemy ORM
- **API Routes**: prefixed with `/api` (e.g., `GET /api/episodes`, `POST /api/chat`)
- **Config endpoint**: `GET /api/config` — tells the frontend whether `USE_LLM` is on

### Frontend (React + TypeScript)
- **Build tool**: Vite
- **Dev server**: port 5173, proxies `/api` calls to Flask on port 5001
- **Production**: React is built into `frontend/dist` and served by Flask

## Deploying on the server

For the initial deployment, only one member of your team needs to follow the steps below.

### Step 0: Fork this template

- **Fork** this repository on your GitHub account
- Make sure your repository is set to **PUBLIC** (required for deployment)
- Keep in mind that other students may be able to see your repository

### Step 1: Login to the deployment dashboard

- Login to the dashboard at https://4300showcase.infosci.cornell.edu/login using the Google account associated with your Cornell Email/NetID. Click the "INFO 4300: Language Information" **Spring 2026** course in the list of course offerings.

![Project Server Home](assets/server-home.png)

### Step 2: Navigate to Your Team Dashboard

- You'll see a list of all teams in the course
- Find your team and click **"Dashboard"** to go to your team's deployment dashboard

![Teams Page For a Course](assets/teams.png)

### Step 3: Deploy Your Project

![Team Dashboard](assets/dashboard-new.png)

1. **Add Your GitHub URL** — paste the URL of your forked (public) repository
2. **Click "Deploy"** — this builds the React frontend, installs Python dependencies, and starts your app
3. **Click "Open Project"** once the status updates to visit your live app

Expand **"Build Logs"** or **"Container Logs"** if something goes wrong.

#### Redeploying After Updates

Push changes to GitHub, then click **"Deploy"** again.

## Running locally

### Prerequisites
- **Python 3.10 or above**
- **Node.js 18 or above**

### Development Mode (recommended)

Run Flask and React separately with hot-reloading.

**Terminal 1 — Flask backend:**
```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python src/app.py
```
Flask API runs on `http://localhost:5001`.

**Terminal 2 — React frontend:**
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173`. The Vite dev server proxies `/api` requests to Flask automatically.

### Production Mode

Build React and serve everything through Flask:
```bash
cd frontend && npm install && npm run build && cd ..
python src/app.py
```
Open `http://localhost:5001`.

### Modifying the Data

Edit `src/init.json` to replace the dummy episode data with your own. You can add additional data files as needed.

## Troubleshooting Deployment Issues

### My app isn't loading after deployment
- Wait 30–60 seconds after deployment — larger apps need time to start up
- Try refreshing your browser or clicking **"Open Project"** again

### How do I see what went wrong?
- **Build Logs** — errors during Docker build (Python or Node dependency issues)
- **Container Logs** — runtime errors from your running application
- Common causes:
  - Missing packages in `requirements.txt`
  - React build errors (check `frontend/package.json`)
  - Malformed `src/init.json`

### Login/Authentication Issues
- If you get a 401 error, try logging out and back in with your Cornell email

### Still Having Issues?
Post on Ed Discussion with your team name, what you tried, and screenshots of error logs.

## Virtual Environments and Dependency Tracking

Keep your virtual environment out of git — it inflates repository size and will break deployment.

1. Make sure `venv/` (or whatever you named it) is listed in `.gitignore`
2. If you already committed it, untrack it with `git rm -r --cached venv/`
3. When you install new packages, update `requirements.txt`:
   ```bash
   pip freeze > requirements.txt
   ```

## Additional Resources

📋 **Known Issues Database**: https://docs.google.com/document/d/1sF2zsubii_SYJLfZN02UB9FvtH1iLmi9xd-X4wbpbo8

## Feedback and Support

- **Problems with deployment?** Post on Ed Discussion or email course staff
- **Questions about the deployment system?** Course staff are happy to help!
