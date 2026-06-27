import { useState, useCallback, useRef, useEffect } from 'react'
import { sileo } from 'sileo'
import { useRobot, useRobotFrame } from '../context/RobotContext.jsx'
import StatusPanel from './StatusPanel.jsx'
import './RutaContinua.css'

// F1 fix: velocidades anteriores (50/80/128) estaban POR DEBAJO del dead-zone
// de los motores DC LAFVIN 4WD (~30-50/255 = 12-20% PWM no supera la friccion
// estatica). Lenta=50 era 19.6% y Normal=80 era 31.4%, al limite. Subimos todas
// por encima de la zona muerta: Lenta=100 (39%), Normal=140 (55%), Rapida=200 (78%).
const SPEED_MODES = [
  { id: 'low', label: 'Lenta', icon: '▶', speedValue: 100 },
  { id: 'medium', label: 'Normal', icon: '▶▶', speedValue: 140 },
  { id: 'high', label: 'Rapida', icon: '▶▶▶', speedValue: 200 },
]

export default function RutaContinua() {
  const {
    esp32Connected,
    backendConnected,
    machineState,
    detectedArrow,
    continuousActive,
    sendRouteCommand,
  } = useRobot()
  const { lastFrameB64 } = useRobotFrame()

  // U3 fix: default unificado a 'medium' para que coincida con
  // RutaAutonoma. Antes era 'low' y al cambiar de tab el usuario veia
  // un salto en la velocidad. Como ademas F1 subio el minimo de 'Lenta'
  // a 100/255 (39% PWM), el riesgo de que un usuario inexperto arranque
  // en Lenta y se frustre por lentitud ya no aplica.
  const [speedMode, setSpeedMode] = useState('medium')
  // U11 fix: loading state local. Antes al clickear "Iniciar Ruta
  // Continua" el boton seguia mostrando "Iniciar" hasta que el WS
  // confirmara (~1-2s), dando feedback pobre. Ahora: deshabilitamos
  // el boton inmediatamente y mostramos "Iniciando...".
  const [isStarting, setIsStarting] = useState(false)
  const isRunningRef = useRef(false)
  const routeLostTimerRef = useRef(null)

  useEffect(() => {
    if (machineState === 'CONTINUOUS') {
      if (routeLostTimerRef.current) {
        clearTimeout(routeLostTimerRef.current)
        routeLostTimerRef.current = null
      }
      isRunningRef.current = true
    } else if (isRunningRef.current) {
      if (!routeLostTimerRef.current) {
        routeLostTimerRef.current = setTimeout(() => {
          routeLostTimerRef.current = null
          isRunningRef.current = false
        }, 1500)
      }
    }
  }, [machineState])

  useEffect(() => {
    return () => {
      if (routeLostTimerRef.current) clearTimeout(routeLostTimerRef.current)
    }
  }, [])

  const handleStart = useCallback(() => {
    if (!esp32Connected || isRunningRef.current) return

    const speedValue = SPEED_MODES.find(m => m.id === speedMode)?.speedValue || 80
    isRunningRef.current = true
    // U11 fix: feedback visual inmediato antes del round-trip WS.
    setIsStarting(true)
    const ok = sendRouteCommand('iniciar_continua', { speed: speedValue })

    if (!ok) {
      isRunningRef.current = false
      setIsStarting(false)
      sileo.error({
        title: 'Error',
        description: 'No hay conexion WebSocket con el backend',
        position: 'top-center',
      })
    } else {
      sileo.success({
        title: 'Ruta continua iniciada',
        description: 'El robot avanzara siempre, reaccionando a semaforos',
        position: 'top-center',
        duration: 3000,
      })
      // El boton seguira disabled hasta que continuousActive se active
      // (o el WS lo confirme). El loading state se limpia cuando
      // continuousActive pasa a true (efecto abajo).
    }
  }, [esp32Connected, speedMode, sendRouteCommand])

  // U11 fix (parte 2): limpiar el loading state cuando el WS confirma
  // que la ruta continua esta activa.
  useEffect(() => {
    if (continuousActive) {
      setIsStarting(false)
    }
  }, [continuousActive])

  const handleAbort = useCallback(() => {
    isRunningRef.current = false
    sendRouteCommand('detener_continua')

    sileo.info({
      title: 'Ruta continua detenida',
      description: 'Robot detenido por el usuario',
      position: 'top-center',
      duration: 3000,
    })
  }, [sendRouteCommand])

  const arrowLabel = detectedArrow === 'left' ? 'Izquierda'
    : detectedArrow === 'right' ? 'Derecha'
    : 'Avanzar'

  const arrowIcon = detectedArrow === 'left' ? '\u2190'
    : detectedArrow === 'right' ? '\u2192'
    : '\u2191'

  return (
    <div className="ruta-container">
      <div className="ruta-card">
        <h2 className="ruta-title">Ruta Continua</h2>
        <p className="ruta-subtitle">
          El robot avanza siempre. Respeta semaforos y flechas automaticamente.
        </p>

        {/* Indicador de direccion detectada */}
        {continuousActive && (
          <div className="continuous-indicator">
            <span className="continuous-arrow">{arrowIcon}</span>
            <span className="continuous-label">{arrowLabel}</span>
          </div>
        )}

        {/* Selector de velocidad */}
        <div className="ruta-section">
          <label className="ruta-label">Velocidad</label>
          <div className="speed-modes">
            {SPEED_MODES.map((mode) => (
              <button
                key={mode.id}
                className={`speed-btn ${speedMode === mode.id ? 'active' : ''}`}
                onClick={() => setSpeedMode(mode.id)}
                disabled={continuousActive}
              >
                <span className="speed-icon">{mode.icon}</span>
                <span className="speed-label">{mode.label}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="route-buttons">
          <button
            className={`start-btn ${continuousActive ? 'running' : ''} ${!esp32Connected || isStarting ? 'disabled' : ''}`}
            onClick={continuousActive ? handleAbort : handleStart}
            disabled={!esp32Connected || isStarting}
          >
            {continuousActive
              ? 'Detener'
              : isStarting
                ? 'Iniciando...'
                : 'Iniciar Ruta Continua'}
          </button>
        </div>

        {/* Tablero de estados — fuera del modal de video */}
        {continuousActive && (
          <StatusPanel compact embedded />
        )}

        {continuousActive && (
          <div className="route-video-overlay">
            <div className="route-video-modal">
              <div className="route-video-main">
                <span className="route-video-label">
                  {backendConnected ? 'Vision procesada' : 'Esperando backend...'}
                </span>
                {backendConnected && lastFrameB64 ? (
                  <img
                    src={`data:image/jpeg;base64,${lastFrameB64}`}
                    alt="Camera Feed (Procesado)"
                    className="route-video-feed"
                  />
                ) : (
                  // U13 fix: clase compartida (ver App.css).
                  <div className="route-video-feed video-placeholder">
                    {backendConnected ? 'Esperando frames...' : 'Sin conexion al backend'}
                  </div>
                )}
              </div>
              <button
                className="abort-btn-stream"
                onClick={handleAbort}
              >
                Abortar
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
