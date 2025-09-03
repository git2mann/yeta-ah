import React from 'react'
import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import IndividualAnalysis from './pages/IndividualAnalysis'
import BatchAnalysis from './pages/BatchAnalysis'
import ModelInfo from './pages/ModelInfo'
import EthicalGuidelines from './pages/EthicalGuidelines'

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/individual" element={<IndividualAnalysis />} />
        <Route path="/batch" element={<BatchAnalysis />} />
        <Route path="/model" element={<ModelInfo />} />
        <Route path="/ethics" element={<EthicalGuidelines />} />
      </Routes>
    </Layout>
  )
}

export default App