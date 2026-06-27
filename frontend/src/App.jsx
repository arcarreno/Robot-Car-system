import { useState } from 'react'
import { Toaster } from 'sileo'
import ThemeToggle from './components/ThemeToggle.jsx'
import Tabs from './components/Tabs.jsx'
import ControlManual from './components/ControlManual.jsx'
import RutaAutonoma from './components/RutaAutonoma.jsx'
import RutaContinua from './components/RutaContinua.jsx'
import ConnectionPanel from './components/ConnectionPanel.jsx'
import KeyboardToggle from './components/KeyboardToggle.jsx'
import SystemIndicators from './components/SystemIndicators.jsx'
import { useTheme } from './context/ThemeContext.jsx'
import './App.css'

function AppContent() {
  const [activeTab, setActiveTab] = useState('manual')

  return (
    <div className="app">
      <header className="app-header">
        <h1>Robot Car Control</h1>
        <div className="header-right">
          <ConnectionPanel />
          <KeyboardToggle />
          <ThemeToggle />
        </div>
      </header>

      <SystemIndicators />

      <main className="app-main">
        {activeTab === 'manual' && <ControlManual />}
        {activeTab === 'route' && <RutaAutonoma />}
        {activeTab === 'continuous' && <RutaContinua />}
      </main>

      <Tabs activeTab={activeTab} onTabChange={setActiveTab} />
    </div>
  )
}

export default function App() {
  const { theme } = useTheme()

  return (
    <>
      <Toaster
        position="top-center"
        className="sileo-toaster"
        theme={theme}
        options={{
          fill: theme === 'dark' ? '#ffffff' : '#1c1c1e',
        }}
      />
      <AppContent />
    </>
  )
}
