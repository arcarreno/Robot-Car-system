import { useState, useEffect, useCallback, useRef, useMemo } from 'react'

type LightColor = 'red' | 'yellow' | 'green'
type Direction = 'left' | 'right' | null

const DURATION = 10000

export default function Semaforo() {
  const [activeLight, setActiveLight] = useState<LightColor>('red')
  const [direction, setDirection] = useState<Direction>(null)
  const [waitingForInput, setWaitingForInput] = useState(true)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Flecha visible mientras el verde este activo y haya direccion
  const showArrow = useMemo(
    () => activeLight === 'green' && direction !== null,
    [activeLight, direction]
  )

  const clearTimers = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current)
    timerRef.current = null
  }, [])

  const goToRed = useCallback(() => {
    clearTimers()
    setActiveLight('red')
    setDirection(null)
    setWaitingForInput(true)
  }, [clearTimers])

  const startGreenCycle = useCallback((dir: Direction) => {
    clearTimers()
    setActiveLight('green')
    setDirection(dir)
    setWaitingForInput(false)

    timerRef.current = setTimeout(() => {
      setActiveLight('yellow')
      timerRef.current = setTimeout(() => {
        goToRed()
      }, DURATION)
    }, DURATION)
  }, [clearTimers, goToRed])

  const startYellowCycle = useCallback(() => {
    clearTimers()
    setActiveLight('yellow')
    setDirection(null)
    setWaitingForInput(false)

    timerRef.current = setTimeout(() => {
      goToRed()
    }, DURATION)
  }, [clearTimers, goToRed])

  useEffect(() => {
    return () => clearTimers()
  }, [clearTimers])

  const handleScreenClick = useCallback((e: React.MouseEvent) => {
    if (!waitingForInput) return

    const w = window.innerWidth
    const x = e.clientX
    const third = w / 3

    const dir: Direction = x < third ? 'left' : x > third * 2 ? 'right' : null
    startGreenCycle(dir)
  }, [waitingForInput, startGreenCycle])

  const handleColorClick = useCallback((color: LightColor, e: React.MouseEvent) => {
    e.stopPropagation()
    clearTimers()

    if (color === 'red') {
      goToRed()
    } else if (color === 'green') {
      setActiveLight('green')
      setDirection(null)
      setWaitingForInput(false)

      timerRef.current = setTimeout(() => {
        setActiveLight('yellow')
        timerRef.current = setTimeout(() => {
          goToRed()
        }, DURATION)
      }, DURATION)
    } else {
      startYellowCycle()
    }
  }, [clearTimers, goToRed, startYellowCycle])

  const lightConfig: Record<LightColor, { bg: string; glow: string; off: string }> = {
    red: {
      bg: 'bg-red-600',
      glow: 'shadow-[0_0_30px_12px_rgba(220,38,38,0.8)]',
      off: 'bg-red-950',
    },
    yellow: {
      bg: 'bg-yellow-400',
      glow: 'shadow-[0_0_30px_12px_rgba(250,204,21,0.8)]',
      off: 'bg-yellow-950',
    },
    green: {
      bg: 'bg-green-500',
      glow: 'shadow-[0_0_30px_12px_rgba(34,197,94,0.8)]',
      off: 'bg-green-950',
    },
  }

  return (
    <div
      className="w-full h-full flex flex-col items-center justify-center bg-white select-none"
      onClick={handleScreenClick}
    >
      {waitingForInput && (
        <p className="text-gray-400 text-lg mb-6 animate-pulse">
          Toca la izquierda, centro o derecha para elegir dirección
        </p>
      )}

      <div className="relative">
        <div className="bg-gray-800 rounded-3xl p-8 sm:p-10 md:p-6 lg:p-8 flex flex-col items-center gap-8 sm:gap-10 md:gap-6 lg:gap-8 border-4 border-gray-700 shadow-2xl">
          {(['red', 'yellow', 'green'] as LightColor[]).map((color) => {
            const isActive = activeLight === color
            const cfg = lightConfig[color]
            return (
              <button
                key={color}
                onClick={(e) => handleColorClick(color, e)}
                className={`
                  w-40 h-40 sm:w-56 sm:h-56 md:w-48 md:h-48 lg:w-56 lg:h-56 rounded-full transition-all duration-300 cursor-pointer
                  border-2 border-gray-600
                  ${isActive ? `${cfg.bg} ${cfg.glow}` : cfg.off}
                  hover:scale-110 active:scale-95
                `}
                aria-label={`Semáforo ${color}`}
              />
            )
          })}
        </div>

        {showArrow && direction && (
          <div
            className={`
              absolute top-1/2 -translate-y-1/2
              ${direction === 'left' ? '-left-28 sm:-left-36 md:-left-28 lg:-left-32' : '-right-28 sm:-right-36 md:-right-28 lg:-right-32'}
            `}
          >
            {/* Circulo verde oscuro (diferente del verde de la luz principal) */}
            <div className="w-20 h-20 sm:w-28 sm:h-28 md:w-20 md:h-20 lg:w-24 lg:h-24 rounded-full bg-green-700 flex items-center justify-center shadow-[0_0_20px_8px_rgba(21,128,61,0.8)]">
              <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="white"
                strokeWidth="3"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="w-12 h-12 sm:w-16 sm:h-16 md:w-12 md:h-12 lg:w-14 lg:h-14"
                style={{ transform: direction === 'left' ? 'rotate(180deg)' : 'rotate(0deg)' }}
              >
                <path d="M5 12h14M12 5l7 7-7 7" />
              </svg>
            </div>
          </div>
        )}
      </div>

      <div className="mt-8 text-gray-500 text-sm sm:text-base">
        {waitingForInput && (
          <span>Esperando dirección...</span>
        )}
        {activeLight === 'green' && (
          <span className="text-green-400">
            Verde {direction === 'left' ? '← Izquierda' : direction === 'right' ? 'Derecha →' : ''}
          </span>
        )}
        {activeLight === 'yellow' && (
          <span className="text-yellow-400">Amarillo — Precaución</span>
        )}
        {activeLight === 'red' && !waitingForInput && (
          <span className="text-red-400">Rojo — Alto</span>
        )}
      </div>
    </div>
  )
}
