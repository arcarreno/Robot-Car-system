import { useRobot } from '../context/RobotContext.jsx'
import './KeyboardToggle.css'

export default function KeyboardToggle() {
  const { keyboardMode, toggleKeyboardMode, esp32Connected } = useRobot()

  return (
    <button
      className={`keyboard-btn ${keyboardMode ? 'active' : ''}`}
      onClick={toggleKeyboardMode}
      disabled={!esp32Connected}
      title={keyboardMode ? 'Desactivar modo teclado' : 'Activar modo teclado'}
    >
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <rect x="2" y="4" width="20" height="16" rx="2" />
        <line x1="6" y1="8" x2="6" y2="8" />
        <line x1="10" y1="8" x2="10" y2="8" />
        <line x1="14" y1="8" x2="14" y2="8" />
        <line x1="18" y1="8" x2="18" y2="8" />
        <line x1="6" y1="12" x2="6" y2="12" />
        <line x1="10" y1="12" x2="10" y2="12" />
        <line x1="14" y1="12" x2="14" y2="12" />
        <line x1="18" y1="12" x2="18" y2="12" />
        <line x1="6" y1="16" x2="18" y2="16" />
      </svg>
      {keyboardMode ? 'Teclado: ON' : 'Modo Teclado'}
    </button>
  )
}
