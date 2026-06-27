import { useState, useCallback, useRef, useEffect } from 'react'
import { sileo } from 'sileo'
import { useRobot, useRobotFrame } from '../context/RobotContext.jsx'
import StatusPanel from './StatusPanel.jsx'
import './RutaAutonoma.css'

// DOC8 fix: magic numbers hoisted a constantes nombradas. Antes habia 5 y
// 1600 sueltos en useState sin explicacion. El nombre ahora deja claro
// que 5 metros es la distancia default y 1600ms el tiempo de giro.
const DEFAULT_DISTANCE_M = 5
const DEFAULT_TURN_MS = 1600
const ROUTE_RESET_DELAY_MS = 5000      // tiempo que la UI muestra "Ruta completada" antes de volver a idle
const ROUTE_LOST_GRACE_MS = 1500       // tiempo que esperamos tras perdida de senal (parpadeo watchdog) antes de cerrar
const ROUTE_COMPLETED_DURATION_MS = 4000  // duracion del toast de sileo al completar

const SPEED_MODES = [
  { id: 'low', label: 'Baja', icon: '▶', speedFactor: 1 },
  { id: 'medium', label: 'Media', icon: '▶▶', speedFactor: 1.5 },
  { id: 'high', label: 'Alta', icon: '▶▶▶', speedFactor: 2 },
]

export default function RutaAutonoma() {
  const {
    esp32Connected,
    backendConnected,
    machineState,
    routeProgress,
    routePhase,
    sendRouteCommand,
  } = useRobot()
  const { lastFrameB64 } = useRobotFrame()

  const [distance, setDistance] = useState(DEFAULT_DISTANCE_M)
  const [speedMode, setSpeedMode] = useState('medium')
  const [turnMs, setTurnMs] = useState(() => {
    const saved = localStorage.getItem('robot-turn-ms')
    return saved ? parseInt(saved, 10) : DEFAULT_TURN_MS
  })
  const [routeState, setRouteState] = useState('idle')  // idle | running | completed
  const [routeMessage, setRouteMessage] = useState('')
  // D1 fix: abortRef eliminado (dead code, nunca se leyo).
  // D2 fix: routeStateRef eliminado (dead code, nunca se leyo).
  const isRunningRef = useRef(false)
  const routeLostTimerRef = useRef(null)
  // F3 fix: handleRouteComplete programaba un setTimeout de 5s para volver a idle
  // sin guardar el handle. Si el usuario iniciaba una nueva ruta antes de los 5s,
  // el timer viejo disparaba setRouteState('idle') sobre el estado 'running' nuevo
  // y corrompia silenciosamente la UI. Tambien causaba setState en componente
  // desmontado. Ahora se trackea en routeResetTimerRef.
  const routeResetTimerRef = useRef(null)

  // DOC9 fix: handleRouteComplete envuelto en useCallback con deps [].
  // DEBE estar ANTES del useEffect que lo referencia (TDZ de const).
  const handleRouteComplete = useCallback(() => {
    if (!isRunningRef.current) return
    isRunningRef.current = false
    setRouteState('completed')
    setRouteMessage('Ruta completada')

    sileo.success({
      title: 'Ruta completada',
      description: 'Robot regreso al punto de inicio',
      position: 'top-center',
      duration: ROUTE_COMPLETED_DURATION_MS,
    })

    if (routeResetTimerRef.current) {
      clearTimeout(routeResetTimerRef.current)
    }
    routeResetTimerRef.current = setTimeout(() => {
      routeResetTimerRef.current = null
      setRouteState('idle')
      setRouteMessage('')
      isRunningRef.current = false
    }, ROUTE_RESET_DELAY_MS)
  }, [])

  // Limpiar timers y resetear estado al desmontar.
  useEffect(() => {
    return () => {
      if (routeLostTimerRef.current) clearTimeout(routeLostTimerRef.current)
      if (routeResetTimerRef.current) clearTimeout(routeResetTimerRef.current)
      isRunningRef.current = false
    }
  }, [])

  // Sincronizar estado local con el contexto del WS
  useEffect(() => {
    if (machineState === 'ROUTE' && routePhase) {
      if (routeLostTimerRef.current) {
        clearTimeout(routeLostTimerRef.current)
        routeLostTimerRef.current = null
      }

      setRouteState('running')

      const phaseMessages = {
        out: 'Avanzando...',
        turn: 'Girando 180 grados',
        back: 'Regresando al inicio...',
        done: 'Ruta completada',
      }
      setRouteMessage(phaseMessages[routePhase] || 'Ejecutando ruta...')
    }

    if (routePhase === 'done') {
      if (routeLostTimerRef.current) {
        clearTimeout(routeLostTimerRef.current)
        routeLostTimerRef.current = null
      }
      handleRouteComplete()
    } else if (machineState !== 'ROUTE' && isRunningRef.current) {
      if (!routeLostTimerRef.current) {
        routeLostTimerRef.current = setTimeout(() => {
          routeLostTimerRef.current = null
          handleRouteComplete()
        }, ROUTE_LOST_GRACE_MS)
      }
    }
  }, [machineState, routePhase, handleRouteComplete])

  // Progress visual: freeze en 100% cuando la ruta esta completada
  const displayProgress = (routePhase === 'done' || routeState === 'completed') ? 100
    : routeProgress != null ? Math.round(routeProgress * 100)
    : 0

  const handleStartRoute = useCallback(() => {
    if (!esp32Connected || isRunningRef.current) return

    // F3 fix: cancelar cualquier timer de reset pendiente de la ruta anterior
    // para que no corrompa el estado de esta nueva ruta cuando dispare.
    if (routeResetTimerRef.current) {
      clearTimeout(routeResetTimerRef.current)
      routeResetTimerRef.current = null
    }

    isRunningRef.current = true
    setRouteState('running')
    setRouteMessage('Iniciando ruta...')

    const ok = sendRouteCommand('iniciar_ruta', {
      distancia: distance,
      velocidad: speedMode,
      giro_ms: turnMs,
    })

    if (!ok) {
      isRunningRef.current = false
      setRouteState('idle')
      sileo.error({
        title: 'Error',
        description: 'No hay conexion WebSocket con el backend',
        position: 'top-center',
      })
    }
  }, [esp32Connected, distance, speedMode, turnMs, sendRouteCommand])

  const handleAbortRoute = useCallback(() => {
    isRunningRef.current = false
    setRouteState('idle')
    sendRouteCommand('detener_ruta')

    sileo.info({
      title: 'Ruta abortada',
      description: 'Robot detenido por el usuario',
      position: 'top-center',
      duration: 3000,
    })
  }, [sendRouteCommand])

  const handleTurnChange = (value) => {
    const ms = Number(value)
    setTurnMs(ms)
    localStorage.setItem('robot-turn-ms', ms)
  }

  return (
    <div className="ruta-container">
      <div className="ruta-card">
        <h2 className="ruta-title">Ruta Autonoma</h2>
        <p className="ruta-subtitle">
          Ruta de ida y vuelta con deteccion de semaforos.
          El robot respeta rojos y amarillos automaticamente.
        </p>

        <div className="ruta-section">
          <label className="ruta-label">
            Distancia: <span className="ruta-value">{distance} metros</span>
          </label>
          <div className="slider-wrapper">
            <input
              type="range"
              min="1"
              max="10"
              step="1"
              value={distance}
              onChange={(e) => setDistance(Number(e.target.value))}
              className="ruta-slider"
              disabled={routeState === 'running'}
            />
            <div className="slider-labels">
              <span>1m</span>
              <span>5m</span>
              <span>10m</span>
            </div>
          </div>
        </div>

        <div className="ruta-section">
          <label className="ruta-label">Velocidad</label>
          <div className="speed-modes">
            {SPEED_MODES.map((mode) => (
              <button
                key={mode.id}
                className={`speed-btn ${speedMode === mode.id ? 'active' : ''}`}
                onClick={() => setSpeedMode(mode.id)}
                disabled={routeState === 'running'}
              >
                <span className="speed-icon">{mode.icon}</span>
                <span className="speed-label">{mode.label}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="ruta-section">
          <label className="ruta-label">
            Giro 180&deg;: <span className="ruta-value">{turnMs}ms</span>
          </label>
          <div className="slider-wrapper">
            <input
              type="range"
              min="200"
              max="6000"
              step="50"
              value={turnMs}
              onChange={(e) => handleTurnChange(e.target.value)}
              className="ruta-slider"
              disabled={routeState === 'running'}
            />
            <div className="slider-labels">
              <span>200ms</span>
              <span>3100ms</span>
              <span>6000ms</span>
            </div>
          </div>
          <p className="ruta-hint">
            Si el robot no gira 180&deg; completo, aumenta este valor.
            Si gira de mas, disminuyalo.
          </p>
        </div>

        <div className="route-buttons">
          <button
            className={`start-btn ${routeState === 'running' ? 'running' : ''} ${!esp32Connected ? 'disabled' : ''}`}
            onClick={handleStartRoute}
            disabled={!esp32Connected || routeState === 'running'}
          >
            {routeState === 'running' ? 'Ejecutando...' : 'Iniciar Ruta'}
          </button>
        </div>

        {(routeState === 'running' || routeState === 'completed') && (
          <div className="route-progress">
            <div className="progress-bar-track">
              <div
                className="progress-bar-fill"
                style={{ width: `${displayProgress}%` }}
              />
            </div>
            <span className="progress-message">{routeMessage}</span>
            <span className="progress-percent">{displayProgress}%</span>
          </div>
        )}

        {routeState === 'running' && (
          <StatusPanel compact embedded />
        )}

        {routeState === 'running' && (
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
                onClick={handleAbortRoute}
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
