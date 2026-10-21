# PocketLig Deployment Guide

This guide covers deploying the **PocketLig** platform using the recommended architecture:
- **Frontend**: [Vercel](https://vercel.com) (Fast global CDN, zero-config Vite/React deployment).
- **Backend API**: [Render](https://render.com) or [Hugging Face Spaces](https://huggingface.co/spaces) (Free persistent Python/Docker hosting).
- **Instant Live Demo**: [ngrok](https://ngrok.com) (Immediate public HTTPS URL with real GPU diffusion active).

---

## 1. Frontend Deployment on Vercel (Recommended)

Because the project is already pushed to your GitHub repository `PocketLig_Workspace`, deploying on Vercel takes less than a minute:

1. Go to [vercel.com](https://vercel.com) and log in with your GitHub account.
2. Click **"Add New..."** → **"Project"**.
3. Select your repository: **`Abhin-Krishna-MP/PocketLig_Workspace`**.
4. Configure Project Settings:
   - **Framework Preset**: `Vite`
   - **Root Directory**: `pocketlig/web` (click Edit and select `pocketlig/web`)
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
5. *(Optional)* Under **Environment Variables**, add:
   - `VITE_API_BASE`: The URL of your deployed backend (e.g. `https://pocketlig-api.onrender.com` or your ngrok tunnel).
6. Click **Deploy**. Vercel will build and deploy the WebGL2 studio in ~40 seconds.

---

## 2. Backend Deployment on Render (Free Tier)

Render can deploy the FastAPI service directly from your GitHub repository using the included `render.yaml` Blueprint or as a Web Service:

### Option A: Using Render Blueprint (1-Click)
1. Go to [dashboard.render.com](https://dashboard.render.com).
2. Click **"New"** → **"Blueprint"**.
3. Connect your repository: `Abhin-Krishna-MP/PocketLig_Workspace`.
4. Render will detect `render.yaml` and configure:
   - Service name: `pocketlig-api`
   - Build command: `pip install -r pocketlig/requirements.txt`
   - Start command: `cd pocketlig && python -m uvicorn pocketlig.api.main:app --host 0.0.0.0 --port $PORT`
   - Environment variables: `POCKETLIG_ARTIFACTS=artifacts`, `PYTHONPATH=src:.`
5. Click **Apply**. Once live, Render gives you a public HTTPS URL (e.g. `https://pocketlig-api.onrender.com`).

### Option B: Using Docker on Render
1. Click **"New"** → **"Web Service"**.
2. Select `Abhin-Krishna-MP/PocketLig_Workspace`.
3. Choose **Docker** as runtime (uses the root `Dockerfile`).
4. Set Port to `7860` or let Render bind to `$PORT`.

---

## 3. Backend Deployment on Hugging Face Spaces (Free Docker)

Hugging Face Spaces offers free 16 GB RAM Docker hosting, ideal for scientific machine learning applications:

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Set Space Name: `pocketlig-api`.
3. Select **Docker** (Blank) as SDK.
4. Set Space to **Public**.
5. Push the repository to the Space remote:
   ```bash
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/pocketlig-api
   git push space main
   ```
6. The backend will automatically build from `Dockerfile` and expose the API on port `7860`.

---

## 4. Instant Live Public URL via ngrok (With Active GPU Diffusion)

If you want to immediately showcase the site to a mentor or review panel with **live local GPU diffusion** enabled:

1. Ensure the local services are running:
   ```bash
   # Terminal 1: FastAPI on port 8008
   cd pocketlig
   POCKETLIG_ARTIFACTS=artifacts python -m uvicorn pocketlig.api.main:app --host 127.0.0.1 --port 8008

   # Terminal 2: Vite dev server on port 5173
   cd pocketlig/web
   npm run dev -- --host 127.0.0.1 --port 5173
   ```

2. Expose the unified Vite portal (which proxies backend calls automatically):
   ```bash
   ngrok http 5173
   ```

3. ngrok will output a public HTTPS address:
   ```
   Forwarding   https://<random-id>.ngrok-free.app -> http://localhost:5173
   ```
   Open that URL on any mobile device, tablet, or external laptop to experience the full 3D studio and live GPU diffusion!
