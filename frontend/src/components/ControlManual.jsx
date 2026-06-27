import { useRobot, useRobotFrame } from '../context/RobotContext.jsx'
import StatusPanel from './StatusPanel.jsx'
import './ControlManual.css'

// U15 fix: antes ControlManual recibia sendCommandDirect/currentCommand
// del contexto y los pasaba como props a ControlGrid/LightControls, pero
// esos children igual llamaban useRobot() para esp32Connected. Patron
// mixto y prop drilling innecesario. Ahora todo se lee del contexto en
// cada componente.

export default function ControlManual() {
  return (
    <div className="control-manual">
      <div className="video-section-manual">
        <VideoFeed />
      </div>
      <div className="controls-section-manual">
        <StatusDisplay />
        <ControlGrid />
        <LightControls />
        <StatusPanel />
      </div>
    </div>
  )
}

function VideoFeed() {
  const { backendConnected, detectedConfidence } = useRobot()
  const { lastFrameB64 } = useRobotFrame()

  return (
    <div className="video-player">
      <div className="video-header">
        <h3>Video en Vivo</h3>
        {backendConnected && <span className="video-source-badge">Procesado</span>}
      </div>
      <div className="video-container">
        {backendConnected && lastFrameB64 ? (
          <>
            <img
              src={`data:image/jpeg;base64,${lastFrameB64}`}
              alt="Camera Feed (Procesado)"
              className="video-feed"
            />
            {detectedConfidence != null && (
              <span className="confidence-badge">
                {Math.round(detectedConfidence * 100)}%
              </span>
            )}
          </>
        ) : (
          // U13 fix: clase CSS compartida en App.css en vez de inline
          // style repetido. Tambien lo usan RutaAutonoma y RutaContinua.
          <div className="video-feed video-placeholder">
            {backendConnected ? 'Esperando frames del backend...' : 'Sin conexion al backend'}
          </div>
        )}
      </div>
    </div>
  )
}

function StatusDisplay() {
  const { esp32Connected, currentCommand } = useRobot()

  return (
    <div className="status-bar-manual">
      <div className="status-indicator">
        <span className={`status-dot ${esp32Connected ? 'online' : ''}`} />
        <span>{esp32Connected ? 'Conectado' : 'Desconectado'}</span>
      </div>
      <div className="status-info">
        <div className="status-item">
          <span className="status-label">Comando:</span>
          <span className={`status-value command ${currentCommand?.toLowerCase() ?? 'idle'}`}>
            {currentCommand}
          </span>
        </div>
      </div>
    </div>
  )
}

function ControlGrid() {
  const { sendCommandDirect, currentCommand, esp32Connected } = useRobot()
  return (
    <div className="control-grid-manual">
      <div className="control-row">
        <button
          className={`control-btn go ${currentCommand === 'GO' ? 'active' : ''}`}
          onClick={() => sendCommandDirect('go')}
          disabled={!esp32Connected}
        >
          <span className="icon">&#9650;</span>
          <span className="label">Adelante</span>
        </button>
      </div>
      <div className="control-row row-middle">
        <button
          className={`control-btn left ${currentCommand === 'LEFT' ? 'active' : ''}`}
          onClick={() => sendCommandDirect('left')}
          disabled={!esp32Connected}
        >
          <span className="icon">&#9664;</span>
          <span className="label">Izquierda</span>
        </button>
        <button
          className={`control-btn stop ${currentCommand === 'STOP' ? 'active' : ''}`}
          onClick={() => sendCommandDirect('stop')}
          disabled={!esp32Connected}
        >
          <span className="icon">&#9632;</span>
          <span className="label">Parar</span>
        </button>
        <button
          className={`control-btn right ${currentCommand === 'RIGHT' ? 'active' : ''}`}
          onClick={() => sendCommandDirect('right')}
          disabled={!esp32Connected}
        >
          <span className="icon">&#9654;</span>
          <span className="label">Derecha</span>
        </button>
      </div>
      <div className="control-row">
        <button
          className={`control-btn back ${currentCommand === 'BACK' ? 'active' : ''}`}
          onClick={() => sendCommandDirect('back')}
          disabled={!esp32Connected}
        >
          <span className="icon">&#9660;</span>
          <span className="label">Atras</span>
        </button>
      </div>
    </div>
  )
}

function LightControls() {
  const { sendCommandDirect, esp32Connected } = useRobot()
  return (
    <div className="light-controls-manual">
      <button
        className="light-btn on"
        onClick={() => sendCommandDirect('ledon')}
        disabled={!esp32Connected}
      >
        <span className="light-icon">ON</span>
        Encender LED
      </button>
      <button
        className="light-btn off"
        onClick={() => sendCommandDirect('ledoff')}
        disabled={!esp32Connected}
      >
        <span className="light-icon">OFF</span>
        Apagar LED
      </button>
    </div>
  )
}