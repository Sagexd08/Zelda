# Deploying the Zelda Backend to Render.com

## Steps to Deploy

1. **Create a Render.com Account**
   - Go to [Render.com](https://render.com/) and sign up for an account
   - Verify your email address

2. **Connect Your GitHub Repository**
   - In the Render dashboard, click "New +"
   - Select "Web Service"
   - Connect your GitHub account
   - Select the Zelda repository

3. **Configure the Web Service**
   - Name: zelda-facial-auth-api
   - Environment: Python
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `cd app && uvicorn main:app --host 0.0.0.0 --port $PORT`
   - Select the appropriate plan (Free tier works for testing)

4. **Set Environment Variables**
   - Add the following environment variables:
     - `PYTHON_VERSION`: 3.9.0
     - `CORS_ORIGINS`: https://zelda-facial-auth.vercel.app

5. **Deploy**
   - Click "Create Web Service"
   - Wait for the deployment to complete (this may take several minutes)

## After Deployment

Once deployed, your backend API will be available at the URL provided by Render (typically `https://zelda-facial-auth-api.onrender.com`).

Use this URL to update your frontend configuration.