import { createContext, useContext, useState, useEffect, useRef, useCallback, useMemo } from 'react'
import { sileo } from 'sileo'
import { ESP32_URL, BACKEND_WS } from '../config.js'

// Dos contexts separados para aislar la frecuencia de updates:
// - RobotStateContext: cambia seguido (color, arrow, state, fps). Lo
//   consumen componentes que necesitan re-renderizar rapido (Tablero).
// - RobotFrameContext: cambia lento (lastFrameB64, throttled a 4 FPS en
//   el backend). Lo consume solo el <VideoFeed>. Asi el Tablero no
//   re-renderiza cada vez que llega un JPEG.
const RobotContext = createContext(null)
const RobotFrameContext = createContext(null)

const ALLOWED_COMMANDS = ['go', 'back', 'left', 'right', 'stop', 'ledon', 'ledoff']
const KEY_COMMANDS = {
  'w': 'go', 'ArrowUp': 'go',
  's': 'back', 'ArrowDown': 'back',
  'a': 'left', 'ArrowLeft': 'left',
  'd': 'right', 'ArrowRight': 'right',
  ' ': 'stop',
}

export function RobotProvider({ children }) {
  const [esp32Connected, setEsp32Connected] = useState(false)
  const [currentCommand, setCurrentCommand] = useState('IDLE')
  const [connecting, setConnecting] = useState(false)
  const [connectionSteps, setConnectionSteps] = useState([])
  const [keyboardMode, setKeyboardMode] = useState(false)
  const [backendConnected, setBackendConnected] = useState(false)
  const [lastFrameB64, setLastFrameB64] = useState(null)
  const [detectedColor, setDetectedColor] = useState(null)
  const [machineState, setMachineState] = useState('IDLE')
  const [processingFps, setProcessingFps] = useState(0)
  const [routeProgress, setRouteProgress] = useState(null)
  const [routePhase, setRoutePhase] = useState(null)
  const [detectedArrow, setDetectedArrow] = useState(null)
  const [continuousActive, setContinuousActive] = useState(false)
  const [esp32Speed, setEsp32Speed] = useState(null)
  const [detectedConfidence, setDetectedConfidence] = useState(null)
  const [obstacleDistance, setObstacleDistance] = useState(null)
  const [obstacleDetected, setObstacleDetected] = useState(false)
  const [obstacleDirection, setObstacleDirection] = useState(null)
  const commandTimerRef = useRef(null)
  const lastSendTimeRef = useRef(0)
  const wsRef = useRef(null)
  const wsReconnectRef = useRef(null)
  const wsReconnectingRef = useRef(false)
  // U6 fix: guard explicito para evitar doble-schedule de reconexion.
  // Antes: onerror -> ws.close() -> onclose, ambos handlers
  // programaban setTimeout(connect) si llegaban a correr en ventanas
  // distintas. wsReconnectingRef solo protege la fase de "abriendo
  // conexion", no la fase de "esperando timer". Este nuevo ref cubre
  // la fase del timer.
  const reconnectScheduledRef = useRef(false)
  const reconnectAttemptRef = useRef(0)

  useEffect(() => {
    return () => {
      if (commandTimerRef.current) clearTimeout(commandTimerRef.current)
      if (wsReconnectRef.current) clearTimeout(wsReconnectRef.current)
      // D6 fix: eliminado el wsRef.current.close() de aca. El cleanup del
      // efecto de WebSocket (lineas ~78-90) ya cierra el socket y es la unica
      // fuente de verdad. Dejarlo en dos lugares es un footgun: el primero
      // corre antes y setea wsRef.current = null, haciendo que el segundo sea
      // un no-op silencioso.
    }
  }, [])

  // --- Ping ESP32 ---
  useEffect(() => {
    let cancelled = false
    let retryCount = 0

    const ping = async () => {
      try {
        const res = await fetch(`${ESP32_URL}/status`, { signal: AbortSignal.timeout(3000) })
        if (!cancelled && res.ok) {
          setEsp32Connected(true)
          retryCount = 0
        }
      } catch {
        if (!cancelled) {
          retryCount++
          // U10 fix: marcar offline en el primer ping fallido si estabamos
          // online. Antes: requeria 2 fallos consecutivos, dando 5-10s
          // de estado stale (usuario ve "online" aunque el ESP32 murio).
          if (retryCount >= 1) {
            setEsp32Connected(false)
          }
        }
      }
    }

    const interval = setInterval(ping, 5000)
    return () => { cancelled = true; clearInterval(interval) }
  }, [])

  // --- WebSocket con el backend ---
  useEffect(() => {
    let cancelled = false

    const connect = () => {
      if (wsReconnectingRef.current) return
      wsReconnectingRef.current = true

      if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
        wsReconnectingRef.current = false
        return
      }

      try {
        const ws = new WebSocket(BACKEND_WS)
        wsRef.current = ws

        ws.onopen = () => {
          if (!cancelled) {
            setBackendConnected(true)
            reconnectAttemptRef.current = 0
          }
          wsReconnectingRef.current = false
        }

        ws.onmessage = (event) => {
          if (cancelled) return
          try {
            const data = JSON.parse(event.data)

            // Mensajes de tipo especial
            if (data.type === 'heartbeat') return
            if (data.type === 'route_status') {
              if (data.status === 'started') {
                setRouteProgress(0)
                setRoutePhase('out')
              } else if (data.status === 'stopped') {
                setRouteProgress(null)
                setRoutePhase(null)
              }
              return
            }
            if (data.type === 'continuous_status') {
              setContinuousActive(data.status === 'started')
              return
            }

            // Frame procesado
            if (data.frame_b64) setLastFrameB64(data.frame_b64)
            // B7 fix: usar != null para permitir reset a null cuando no hay color.
            // Antes: if (data.detected_color) ignoraba null y el color viejo
            // permanecía en el Tablero de Estados indefinidamente.
            if (data.detected_color !== undefined) setDetectedColor(data.detected_color)
            if (data.detected_arrow !== undefined) setDetectedArrow(data.detected_arrow)
            // DOC10 fix: usar != null en vez de truthy. Antes '' (string
            // vacio) era falsy y se ignoraba. Poco probable hoy (backend
            // siempre manda 'IDLE' no ''), pero el check era fragil.
            if (data.state != null) setMachineState(data.state)
            if (data.fps != null) setProcessingFps(data.fps)
            // Datos de ruta (vienen en frames cuando state=ROUTE)
            if (data.route_progress != null) setRouteProgress(data.route_progress)
            if (data.route_phase != null) setRoutePhase(data.route_phase)
            // FASE F: velocidad ESP32 y confianza YOLO
            if (data.speed != null) setEsp32Speed(data.speed)
            if (data.detected_confidence != null) setDetectedConfidence(data.detected_confidence)
            // Deteccion de obstaculos (MiDaS depth)
            if (data.obstacle_distance !== undefined) setObstacleDistance(data.obstacle_distance)
            if (data.obstacle_detected !== undefined) setObstacleDetected(data.obstacle_detected)
            if (data.obstacle_direction !== undefined) setObstacleDirection(data.obstacle_direction)
          } catch (e) {
            console.warn('[WS] Error parsing message:', e)
          }
        }

        ws.onclose = () => {
          if (!cancelled) setBackendConnected(false)
          wsReconnectingRef.current = false
          if (cancelled) return
          if (reconnectScheduledRef.current) return  // U6 fix: ya hay un timer programado
          reconnectAttemptRef.current += 1
          const delay = Math.min(1000 * Math.pow(2, reconnectAttemptRef.current), 30000)
          reconnectScheduledRef.current = true  // U6 fix: marcar que ya hay timer
          wsReconnectRef.current = setTimeout(() => {
            reconnectScheduledRef.current = false  // U6 fix: limpiar al disparar
            connect()
          }, delay)
        }

        ws.onerror = () => {
          wsReconnectingRef.current = false
          ws.close()  // onclose manejara el reconnect
        }
      } catch {
        wsReconnectingRef.current = false
        if (cancelled) return
        if (reconnectScheduledRef.current) return  // U6 fix
        reconnectAttemptRef.current += 1
        const delay = Math.min(1000 * Math.pow(2, reconnectAttemptRef.current), 30000)
        reconnectScheduledRef.current = true  // U6 fix
        wsReconnectRef.current = setTimeout(() => {
          reconnectScheduledRef.current = false  // U6 fix
          connect()
        }, delay)
      }
    }

    connect()
    return () => {
      cancelled = true
      if (wsRef.current) {
        wsRef.current.close()
        wsRef.current = null
      }
    }
  }, [])

  // --- Keyboard handler (solo cuando keyboardMode activo) ---
  useEffect(() => {
    if (!keyboardMode) return

    const sendViaWs = (cmd) => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ action: 'comando', comando: cmd }))
        return true
      }
      return false
    }

    // F2 fix: guard de target. Antes el listener capturaba W/A/S/D/flechas/Space
    // y disparaba comandos del robot incluso cuando el usuario estaba tipeando
    // en un <input>/<textarea> o ajustando un slider con flechas. Resultado:
    // - Inputs de texto se volvieran inutilizables con keyboardMode activo.
    // - Los sliders de distancia y turnMs de RutaAutonoma movian el robot Y
    //   cambiaban el valor del slider al mismo tiempo.
    const isEditableTarget = (target) => {
      if (!target) return false
      const tag = target.tagName
      return (
        tag === 'INPUT' ||
        tag === 'TEXTAREA' ||
        tag === 'SELECT' ||
        target.isContentEditable
      )
    }

    const handleKeyDown = (e) => {
      if (e.repeat) return
      if (isEditableTarget(e.target)) return
      const cmd = KEY_COMMANDS[e.key]
      if (cmd) {
        e.preventDefault()
        if (!sendViaWs(cmd)) {
          sendCommandDirect(cmd)
        }
      }
    }

    const handleKeyUp = (e) => {
      if (isEditableTarget(e.target)) return
      if (KEY_COMMANDS[e.key]) {
        e.preventDefault()
        const cmd = KEY_COMMANDS[e.key]
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send(JSON.stringify({ action: 'tecla_up', comando: cmd }))
        } else {
          sendCommandDirect('stop')
        }
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    window.addEventListener('keyup', handleKeyUp)
    return () => {
      window.removeEventListener('keydown', handleKeyDown)
      window.removeEventListener('keyup', handleKeyUp)
    }
  }, [keyboardMode]) // eslint-disable-line react-hooks/exhaustive-deps

  // U9 fix: renombrado connectAll -> connectEsp32. La funcion solo prueba
  // ESP32 (un ping HTTP), no el backend (que ya tiene su propio WebSocket).
  // El nombre anterior implicaba que probaba ambos, lo cual era confuso.
  // Tambien: el toast en ConnectionPanel.jsx dice "ESP32 listo para
  // operar" que es coherente con este alcance.
  const connectEsp32 = useCallback(async () => {
    setConnecting(true)
    setConnectionSteps([])
    setEsp32Connected(false)

    const steps = []

    steps.push({ label: 'Conectando al ESP32...', status: 'checking' })
    setConnectionSteps([...steps])

    try {
      const response = await fetch(`${ESP32_URL}/status`, {
        method: 'GET',
        signal: AbortSignal.timeout(5000)
      })
      if (response.ok) {
        steps[0] = { label: 'ESP32 conectado', status: 'success' }
        setEsp32Connected(true)
      } else {
        steps[0] = {
          label: 'ESP32 respondio pero con error',
          status: 'error',
          fix: 'Revisa que el ESP32 este correctamente encendido y flasheado'
        }
      }
    } catch (err) {
      if (err.name === 'AbortError') {
        steps[0] = {
          label: 'ESP32 no responde (timeout)',
          status: 'error',
          fix: '1. Conectate al WiFi del ESP32 (ESP32-CAM Robot)\n2. Verifica que tengas IP estatica 192.168.4.2\n3. El ESP32 debe estar encendido'
        }
      } else {
        steps[0] = {
          label: 'Sin conexion al ESP32',
          status: 'error',
          fix: '1. Conecta tu laptop al WiFi del ESP32\n2. Verifica que la IP estatica sea 192.168.4.2\n3. Abre http://192.168.4.1/go en el navegador'
        }
      }
    }
    setConnectionSteps([...steps])
    setConnecting(false)
    return { esp32: steps[0]?.status === 'success' }
  }, [])

  const sendCommandDirect = useCallback(async (command) => {
    if (!ALLOWED_COMMANDS.includes(command)) return false

    const now = Date.now()
    if (now - lastSendTimeRef.current < 100) return false
    lastSendTimeRef.current = now

    // A4 fix: el frontend SOLO notifica al backend via WS.
    // El backend es la unica autoridad para enviar comandos al ESP32.
    // Antes: el frontend tambien enviaba HTTP directo al ESP32, creando
    // una "doble via" donde el ESP32 recibia comandos por dos canales
    // sin coordinacion (race condition).
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'comando', comando: command }))
      setCurrentCommand(command.toUpperCase())
      if (commandTimerRef.current) clearTimeout(commandTimerRef.current)
      commandTimerRef.current = setTimeout(() => setCurrentCommand('IDLE'), 2000)
      return true
    }

    // Sin conexion WS: notificar al usuario
    sileo.error({
      title: 'Sin conexion con backend',
      description: `No se pudo enviar ${command}: sin WebSocket`,
      position: 'top-center',
      duration: 3000,
    })
    setCurrentCommand('IDLE')
    return false
  }, [])

  const toggleKeyboardMode = useCallback(() => {
    setKeyboardMode(prev => !prev)
  }, [])

  const sendRouteCommand = useCallback((action, params = {}) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        action,
        ...params,
      }))
      return true
    }
    return false
  }, [])

  return (
    <RobotContext.Provider
      value={{
        esp32Connected,
        currentCommand,
        connecting,
        connectionSteps,
        connectEsp32,
        sendCommandDirect,
        keyboardMode,
        toggleKeyboardMode,
        // D4 fix: eliminado setKeyboardMode del value. Era dead code, ningun
        // consumidor lo usaba (solo KeyboardToggle usa toggleKeyboardMode).
        backendConnected,
        detectedColor,
        machineState,
        processingFps,
        routeProgress,
        routePhase,
        detectedArrow,
        continuousActive,
        sendRouteCommand,
        esp32Speed,
        detectedConfidence,
        obstacleDistance,
        obstacleDetected,
        obstacleDirection,
      }}
    >
      <RobotFrameContext.Provider value={{ lastFrameB64 }}>
        {children}
      </RobotFrameContext.Provider>
    </RobotContext.Provider>
  )
}

export function useRobot() {
  const ctx = useContext(RobotContext)
  if (!ctx) throw new Error('useRobot debe usarse dentro de RobotProvider')
  return ctx
}

// Hook para componentes que SOLO necesitan el frame (VideoFeed en
// ControlManual, RutaAutonoma, RutaContinua). Se re-renderiza solo
// cuando llega un JPEG nuevo (throttled a 4 FPS en el backend).
export function useRobotFrame() {
  const ctx = useContext(RobotFrameContext)
  if (!ctx) throw new Error('useRobotFrame debe usarse dentro de RobotProvider')
  return ctx
}