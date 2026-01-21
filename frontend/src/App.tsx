import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import PerformanceV2 from './pages/PerformanceV2'
import Risk from './pages/Risk'
import Holdings from './pages/Holdings'
import ImportData from './pages/ImportData'
import Layout from './components/Layout'

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/performance" element={<PerformanceV2 />} />
          <Route path="/risk" element={<Risk />} />
          <Route path="/holdings" element={<Holdings />} />
          <Route path="/import" element={<ImportData />} />
        </Routes>
      </Layout>
    </Router>
  )
}

export default App
