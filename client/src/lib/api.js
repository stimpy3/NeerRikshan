import axios from 'axios'

export const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
})

// --- Glacial lake monitoring (High Mountain Asia) ---
export const getLakes = (params = {}) =>
  api.get('/lakes', { params }).then((res) => res.data)

export const getLakeStats = () => api.get('/lakes/stats').then((res) => res.data)

export const getLakesGeoJSON = (params = {}) =>
  api.get('/lakes/geojson', { params }).then((res) => res.data)

export const getFlowPaths = (params = {}) =>
  api.get('/lakes/flowpaths', { params }).then((res) => res.data)

export const getLake = (id) => api.get(`/lakes/${id}`).then((res) => res.data)

export const getLakeFrames = (id) =>
  api.get(`/lakes/${id}/frames`).then((res) => res.data)
