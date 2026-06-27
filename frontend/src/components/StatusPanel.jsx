/**
 * StatusPanel — Tablero de estados compartido.
 *
 * Muestra deteccion de color, flecha, obstaculo, FPS, estado, etc.
 * Reutilizable en ControlManual, RutaAutonoma y RutaContinua.
 */

import { useRobot } from '../context/RobotContext'

const COLOR_LABELS = {
  red: 'ROJO',
  yellow: 'AMARILLO',
  green: 'VERDE',
}

const COLOR_HEX = {
  red: '#FF0000',
  yellow: '#FFCC00',
  green: '#008000',
}

const ARROW_LABELS = {
  left: 'IZQUIERDA',
  right: 'DERECHA',
}

const ARROW_ICON = {
  left: '\u2190',
  right: '\u2192',
}

const ARROW_COLOR = {
  left: '#2563eb',
  right: '#16a34a',
}

const STATE_LABELS = {
  IDLE: 'ESPERANDO',
  COLOR_CHECK: 'MONITOREANDO',
  MANUAL: 'CONTROL MANUAL',
  ROUTE: 'RUTA AUTONOMA',
  CONTINUOUS: 'RUTA CONTINUA',
}

export default function StatusPanel({ compact = false, embedded = false }) {
  const {
    backendConnected, detectedColor, detectedArrow,
    machineState, processingFps, esp32Speed,
    obstacleDistance, obstacleDetected, obstacleDirection,
  } = useRobot()

  if (compact) {
    // Version compacta para modales de video (una columna)
    return (
      <div className={`status-panel status-panel-compact ${embedded ? 'status-panel-embedded' : ''}`}>
        <div className="status-panel-grid status-panel-grid-compact">
          {/* Color */}
          <div className="status-card status-card-sm">
            <span className="status-card-label">Color</span>
            <div className="status-card-value">
              {backendConnected && detectedColor ? (
                <>
                  <span
                    className="color-swatch"
                    style={{ background: COLOR_HEX[detectedColor] || '#888' }}
                  />
                  <span className="color-name">{COLOR_LABELS[detectedColor] || detectedColor}</span>
                </>
              ) : (
                <span className="color-na">—</span>
              )}
            </div>
          </div>

          {/* Flecha */}
          <div className="status-card status-card-sm">
            <span className="status-card-label">Flecha</span>
            <div className="status-card-value">
              {backendConnected && detectedArrow ? (
                <>
                  <span
                    className="arrow-icon"
                    style={{ color: ARROW_COLOR[detectedArrow] || '#888' }}
                  >
                    {ARROW_ICON[detectedArrow]}
                  </span>
                  <span className="arrow-name">{ARROW_LABELS[detectedArrow] || detectedArrow}</span>
                </>
              ) : (
                <span className="color-na">—</span>
              )}
            </div>
          </div>

          {/* Estado */}
          <div className="status-card status-card-sm">
            <span className="status-card-label">Estado</span>
            <span className="status-card-value">
              {backendConnected
                ? STATE_LABELS[machineState] || machineState
                : 'DESCONECTADO'}
            </span>
          </div>

          {/* FPS */}
          <div className="status-card status-card-sm">
            <span className="status-card-label">FPS</span>
            <span className="status-card-value fps-num">
              {backendConnected ? `${processingFps.toFixed(1)}` : '—'}
            </span>
          </div>

          {/* Obstaculo */}
          <div className={`status-card status-card-sm ${obstacleDetected ? 'obstacle-active' : ''}`}>
            <span className="status-card-label">Obstaculo</span>
            <span className="status-card-value">
              {obstacleDistance != null ? (
                <>
                  <span className={`obstacle-dot ${obstacleDetected ? 'danger' : 'safe'}`} />
                  {obstacleDistance.toFixed(2)}m
                </>
              ) : (
                <span className="color-na">—</span>
              )}
            </span>
          </div>
        </div>
      </div>
    )
  }

  // Version completa para ControlManual
  return (
    <div className="status-panel">
      <h4 className="status-panel-title">Tablero de Estados</h4>
      <div className="status-panel-grid">
        {/* Color detectado */}
        <div className="status-card color-card">
          <span className="status-card-label">Color Detectado</span>
          <div className="status-card-value">
            {backendConnected && detectedColor ? (
              <>
                <span
                  className="color-swatch"
                  style={{ background: COLOR_HEX[detectedColor] || '#888' }}
                />
                <span className="color-name">{COLOR_LABELS[detectedColor] || detectedColor}</span>
              </>
            ) : (
              <span className="color-na">—</span>
            )}
          </div>
        </div>

        {/* Flecha detectada */}
        <div className="status-card arrow-card">
          <span className="status-card-label">Flecha Detectada</span>
          <div className="status-card-value">
            {backendConnected && detectedArrow ? (
              <>
                <span
                  className="arrow-icon"
                  style={{ color: ARROW_COLOR[detectedArrow] || '#888' }}
                >
                  {ARROW_ICON[detectedArrow]}
                </span>
                <span className="arrow-name">{ARROW_LABELS[detectedArrow] || detectedArrow}</span>
              </>
            ) : (
              <span className="color-na">—</span>
            )}
          </div>
        </div>

        {/* Estado */}
        <div className="status-card state-card">
          <span className="status-card-label">Estado</span>
          <span className="status-card-value">
            {backendConnected
              ? STATE_LABELS[machineState] || machineState
              : 'DESCONECTADO'}
          </span>
        </div>

        {/* FPS */}
        <div className="status-card fps-card">
          <span className="status-card-label">FPS</span>
          <span className="status-card-value fps-num">
            {backendConnected ? `${processingFps.toFixed(1)} fps` : '—'}
          </span>
        </div>

        {/* Velocidad ESP32 */}
        <div className="status-card speed-card">
          <span className="status-card-label">Velocidad</span>
          <span className="status-card-value">
            {esp32Speed != null ? `${esp32Speed}` : '—'}
          </span>
        </div>

        {/* Conexion Backend */}
        <div className="status-card conn-card">
          <span className="status-card-label">Backend</span>
          <span className={`status-card-value ${backendConnected ? 'conn-ok' : 'conn-off'}`}>
            {backendConnected ? 'CONECTADO' : 'OFF'}
          </span>
        </div>

        {/* Obstaculo detectado */}
        <div className={`status-card obstacle-card ${obstacleDetected ? 'obstacle-active' : ''}`}>
          <span className="status-card-label">Obstaculo</span>
          <span className="status-card-value">
            {obstacleDistance != null ? (
              <>
                <span className={`obstacle-dot ${obstacleDetected ? 'danger' : 'safe'}`} />
                {obstacleDistance.toFixed(2)}m
                {obstacleDirection && (
                  <span className="obstacle-dir">
                    {obstacleDirection === 'left' ? ' \u2190' : ' \u2192'}
                  </span>
                )}
              </>
            ) : (
              <span className="color-na">\u2014</span>
            )}
          </span>
        </div>
      </div>
    </div>
  )
}
