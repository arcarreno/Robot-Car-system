import { useState, useEffect, useRef, useCallback } from 'react'
import { sileo } from 'sileo'
import { useRobot } from '../context/RobotContext.jsx'
import EarthLoader from './EarthLoader.jsx'

export default function ConnectionPanel() {
  // U9 fix: connectAll -> connectEsp32 (ver contexto).
  const { connectEsp32, connecting, connectionSteps, esp32Connected } = useRobot()
  const [showPanel, setShowPanel] = useState(false)
  // U2 fix: guard contra doble-click. Si ya hay un connectAll en flight,
  // no disparar otro. Antes el usuario podia abrir el modal y volver a
  // clickear "Conectar" cerrando la modal primero.
  const inFlightRef = useRef(false)
  // Para mover el foco al boton de cerrar al abrir
  const closeBtnRef = useRef(null)

  const handleConnect = useCallback(async () => {
    if (inFlightRef.current) return
    inFlightRef.current = true
    setShowPanel(true)
    try {
      const result = await connectEsp32()
      if (result.esp32) {
        sileo.success({
          title: 'Conectado',
          description: 'ESP32 listo para operar',
          position: 'top-center',
          duration: 4000,
        })
        // U2 fix: auto-cerrar el modal en exito. Antes quedaba abierto
        // hasta que el usuario clickee X o el overlay.
        setShowPanel(false)
      } else {
        sileo.error({
          title: 'Conexion fallida',
          description: 'No se pudo conectar al ESP32. Revisa las instrucciones.',
          position: 'top-center',
          duration: 8000,
        })
      }
    } finally {
      inFlightRef.current = false
    }
  }, [connectEsp32])

  // U2 fix: cerrar el modal con Escape + mover foco al abrir.
  useEffect(() => {
    if (!showPanel) return
    const handleKey = (e) => {
      if (e.key === 'Escape') setShowPanel(false)
    }
    window.addEventListener('keydown', handleKey)
    // Mover foco al boton de cerrar al abrir el modal
    const t = setTimeout(() => closeBtnRef.current?.focus(), 50)
    return () => {
      window.removeEventListener('keydown', handleKey)
      clearTimeout(t)
    }
  }, [showPanel])

  if (esp32Connected) {
    return (
      <div className="connection-badge online">
        <span className="badge-dot" />
        Activo
      </div>
    )
  }

  return (
    <>
      <button className="connect-btn" onClick={handleConnect} disabled={inFlightRef.current}>
        {inFlightRef.current ? 'Conectando...' : 'Conectar'}
      </button>

      {showPanel && (connecting || connectionSteps.length > 0) && (
        <div
          className="connection-modal-overlay"
          onClick={() => setShowPanel(false)}
          role="dialog"
          aria-modal="true"
          aria-labelledby="connection-modal-title"
        >
          <div className="connection-modal" onClick={(e) => e.stopPropagation()}>
            <button
              ref={closeBtnRef}
              className="modal-close"
              onClick={() => setShowPanel(false)}
              aria-label="Cerrar panel de conexion"
            >
              &times;
            </button>
            <h3 id="connection-modal-title" className="modal-title">
              Estado de conexion
            </h3>

            <EarthLoader />

            <div className="steps-list">
              {connectionSteps.map((step, i) => (
                <div key={i} className={`step ${step.status}`}>
                  <span className="step-icon">
                    {step.status === 'pending' && 'Esperando...'}
                    {step.status === 'checking' && 'Verificando...'}
                    {step.status === 'success' && 'Conectado'}
                    {step.status === 'error' && 'Error'}
                  </span>
                  <span className="step-label">{step.label}</span>
                  {step.status === 'error' && step.fix && (
                    <div className="step-fix">
                      <strong>Como arreglarlo:</strong>
                      <code>{step.fix}</code>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </>
  )
}
