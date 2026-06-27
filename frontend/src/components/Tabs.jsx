import './Tabs.css'

const TABS = [
  { id: 'manual', label: 'Control Manual' },
  { id: 'route', label: 'Ruta Autonoma' },
  { id: 'continuous', label: 'Ruta Continua' },
]

export default function Tabs({ activeTab, onTabChange }) {
  // U1 fix: navegacion con flechas entre tabs (patron WAI-ARIA tabs).
  // Antes no se podia cambiar de tab con teclado, solo con click.
  const handleKeyDown = (e, currentIdx) => {
    let newIdx = null
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
      newIdx = (currentIdx + 1) % TABS.length
    } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
      newIdx = (currentIdx - 1 + TABS.length) % TABS.length
    } else if (e.key === 'Home') {
      newIdx = 0
    } else if (e.key === 'End') {
      newIdx = TABS.length - 1
    }
    if (newIdx !== null) {
      e.preventDefault()
      onTabChange(TABS[newIdx].id)
    }
  }

  return (
    <nav className="tabs-container">
      <div
        className="tabs-glass"
        role="tablist"
        aria-label="Modos de operacion del robot"
      >
        {TABS.map((tab, idx) => (
          <button
            key={tab.id}
            role="tab"
            id={`tab-${tab.id}`}
            aria-selected={activeTab === tab.id}
            aria-controls={`tabpanel-${tab.id}`}
            tabIndex={activeTab === tab.id ? 0 : -1}
            className={`tab-pill ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => onTabChange(tab.id)}
            onKeyDown={(e) => handleKeyDown(e, idx)}
          >
            {tab.label}
          </button>
        ))}
      </div>
    </nav>
  )
}
