import { BrowserRouter, Routes, Route } from 'react-router-dom'
import LakeDetail from './pages/LakeDetail'
import LakeOverview from './pages/LakeOverview'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<LakeOverview />} />
        <Route path="/lakes/:id" element={<LakeDetail />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
