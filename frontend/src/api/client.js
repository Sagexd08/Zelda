import axios from 'axios';

// For Vercel deployment, we need to use the production API URL
const API_BASE_URL = import.meta.env.VITE_API_URL || 
  (window.location.hostname.includes('vercel.app') 
    ? 'https://zelda-facial-auth-api.onrender.com' // Updated with actual Render.com backend URL
    : 'http://localhost:8000');

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  withCredentials: true, // Enable cookies for cross-origin requests if needed
});

// System Info
export const getSystemInfo = async () => {
  const response = await apiClient.get('/api/v1/system/info');
  return response.data;
};

export const getHealth = async () => {
  const response = await apiClient.get('/health');
  return response.data;
};

// Registration
export const registerUser = async (userId, images) => {
  const formData = new FormData();
  images.forEach((image) => {
    formData.append('images', image);
  });

  const response = await apiClient.post(
    `/api/v1/register?user_id=${userId}`,
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }
  );
  return response.data;
};

// Authentication
export const authenticateUser = async (userId, image) => {
  const formData = new FormData();
  formData.append('image', image);

  const response = await apiClient.post(
    `/api/v1/authenticate?user_id=${userId}`,
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }
  );
  return response.data;
};

// Identification
export const identifyUser = async (image, topK = 3) => {
  const formData = new FormData();
  formData.append('image', image);

  const response = await apiClient.post(
    `/api/v1/identify?top_k=${topK}`,
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }
  );
  return response.data;
};

// User Management
export const deleteUser = async (userId) => {
  const response = await apiClient.post(
    `/api/v1/delete_user?user_id=${userId}`
  );
  return response.data;
};

// WebSocket connection
export const createWebSocketConnection = (clientId) => {
  const wsUrl = API_BASE_URL.replace('http', 'ws');
  return new WebSocket(`${wsUrl}/ws/${clientId}`);
};

export default apiClient;


