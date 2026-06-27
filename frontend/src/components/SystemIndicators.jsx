import { useRobot } from '../context/RobotContext.jsx'
import './SystemIndicators.css'

export default function SystemIndicators() {
  const { esp32Connected, backendConnected, keyboardMode } = useRobot()

  return (
    <div className="system-indicators">
      {/* Indicador Backend */}
      <div className={`indicator backend ${backendConnected ? 'online' : ''}`}>
        <span className="indicator-dot" />
        <span className="indicator-label">
          {backendConnected ? 'Backend: ON' : 'Backend: OFF'}
        </span>
      </div>

      {/* Indicador Modo Teclado */}
      {/* U4 fix: antes usaba inline style para el color del dot, los otros
          dos indicadores usan CSS class (.online -> .indicator-dot). Unificado
          a la misma convencion via .indicator.kb.on. */}
      <div className={`indicator kb ${keyboardMode ? 'on' : ''}`}>
        <span className="indicator-dot" />
        <span className="kb-text">
          {keyboardMode ? 'KB: WASD' : 'KB: OFF'}
        </span>
      </div>

      {/* Estado ESP32 */}
      <div className={`indicator esp32 ${esp32Connected ? 'online' : ''}`}>
        <span className="indicator-dot" />
        <span>{esp32Connected ? 'ESP32 OK' : 'ESP32'}</span>
      </div>
    </div>
  )
}
