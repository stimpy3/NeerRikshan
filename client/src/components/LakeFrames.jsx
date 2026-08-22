import { useCallback, useEffect, useRef, useState } from 'react'
import { api, getLakeFrames } from '../lib/api'

const SPEEDS = [
  { label: '0.5×', ms: 800 },
  { label: '1×', ms: 400 },
  { label: '2×', ms: 200 },
  { label: '4×', ms: 100 },
]

/**
 * Year-by-year imagery for one lake.
 *
 * A trend line asserts that a lake grew; this lets you check it. The overlay
 * toggle draws the model's water outline on the same frame, so the question
 * "did it really grow, or did the model drift?" is answerable by eye rather
 * than on trust.
 *
 * Scrubbing matters more than playback here. Growth between adjacent years is
 * often smaller than the frame-to-frame flicker of snow and cloud, so the
 * useful gesture is holding on one year and stepping deliberately - hence
 * arrows, keyboard, wheel, and a year-labelled scrubber, not just Play.
 *
 * Frames only exist for lakes that have been rendered (the watchlist by
 * default), so an empty result is normal and says so rather than looking broken.
 */
export default function LakeFrames({ lakeId, series }) {
  const [frames, setFrames] = useState([])
  const [i, setI] = useState(0)
  const [overlay, setOverlay] = useState(true)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1) // index into SPEEDS
  const [loading, setLoading] = useState(true)
  const timer = useRef(null)
  const box = useRef(null)

  useEffect(() => {
    let alive = true
    setLoading(true)
    getLakeFrames(lakeId)
      .then((d) => {
        if (!alive) return
        setFrames(d.frames || [])
        setI(0)
        setLoading(false)
      })
      .catch(() => alive && (setFrames([]), setLoading(false)))
    return () => {
      alive = false
    }
  }, [lakeId])

  const step = useCallback(
    (delta) => {
      setPlaying(false)
      setI((k) => Math.min(Math.max(k + delta, 0), Math.max(frames.length - 1, 0)))
    },
    [frames.length],
  )

  useEffect(() => {
    clearInterval(timer.current)
    if (playing && frames.length > 1) {
      timer.current = setInterval(
        () => setI((k) => (k + 1) % frames.length),
        SPEEDS[speed].ms,
      )
    }
    return () => clearInterval(timer.current)
  }, [playing, frames.length, speed])

  useEffect(() => () => clearInterval(timer.current), [])

  // Arrow keys step years while the player has focus. Scoped to the element
  // rather than window so it cannot fight the page's own navigation.
  useEffect(() => {
    const el = box.current
    if (!el) return
    const onKey = (e) => {
      if (e.key === 'ArrowLeft') { e.preventDefault(); step(-1) }
      if (e.key === 'ArrowRight') { e.preventDefault(); step(1) }
      if (e.key === ' ') { e.preventDefault(); setPlaying((p) => !p) }
    }
    el.addEventListener('keydown', onKey)
    return () => el.removeEventListener('keydown', onKey)
  }, [step])

  if (loading) {
    return <div className="py-2 font-sans text-[11px] text-muted">Loading imagery…</div>
  }
  if (!frames.length) {
    return (
      <div className="border border-dashed border-line px-2 py-3 font-sans text-[11px] text-muted">
        No imagery rendered for this lake yet — thumbnails are generated for the
        watchlist first.
      </div>
    )
  }

  const idx = Math.min(i, frames.length - 1)
  const f = frames[idx]
  const areaFor = (yr) => (series && series[String(yr)]) || null
  const area = areaFor(f.year)
  const firstArea = areaFor(frames[0].year)
  const delta =
    area != null && firstArea ? ((area - firstArea) / firstArea) * 100 : null

  return (
    <div
      ref={box}
      tabIndex={0}
      className="outline-none focus:ring-1 focus:ring-[#b4442e]/40"
      onWheel={(e) => {
        if (Math.abs(e.deltaY) < 4) return
        e.preventDefault()
        step(e.deltaY > 0 ? 1 : -1)
      }}
    >
      <div className="mb-1 flex flex-wrap items-center gap-1.5 font-sans text-[11px]">
        <span className="font-condensed text-[17px] leading-none text-ink">{f.year}</span>
        {area != null && (
          <span className="text-muted">
            {area.toFixed(1)} ha
            {delta != null && Math.abs(delta) >= 0.1 && (
              <span className={delta > 0 ? 'text-[#b4442e]' : 'text-[#2e6fb4]'}>
                {' '}
                {delta > 0 ? '+' : ''}
                {delta.toFixed(1)}%
              </span>
            )}
          </span>
        )}
        <span className="text-muted">
          · {idx + 1}/{frames.length}
        </span>

        <div className="ml-auto flex items-center gap-1">
          <button
            onClick={() => step(-1)}
            disabled={idx === 0}
            title="Previous year (←)"
            className="border border-line px-1.5 py-0.5 text-ink hover:bg-paper disabled:opacity-30"
          >
            ‹
          </button>
          <button
            onClick={() => setPlaying((p) => !p)}
            title="Play / pause (space)"
            className="border border-line px-1.5 py-0.5 text-ink hover:bg-paper"
          >
            {playing ? '❚❚' : '▶'}
          </button>
          <button
            onClick={() => step(1)}
            disabled={idx === frames.length - 1}
            title="Next year (→)"
            className="border border-line px-1.5 py-0.5 text-ink hover:bg-paper disabled:opacity-30"
          >
            ›
          </button>
          <select
            value={speed}
            onChange={(e) => setSpeed(Number(e.target.value))}
            title="Playback speed"
            className="border border-line bg-transparent px-1 py-0.5 text-ink"
          >
            {SPEEDS.map((s, k) => (
              <option key={s.label} value={k}>
                {s.label}
              </option>
            ))}
          </select>
          <label className="ml-1 flex items-center gap-1 text-muted">
            <input
              type="checkbox"
              checked={overlay}
              onChange={(e) => setOverlay(e.target.checked)}
            />
            outline
          </label>
        </div>
      </div>

      <div className="relative w-full overflow-hidden border border-line bg-ink/5">
        {/* fixed aspect box so swapping years never reflows the panel */}
        <div style={{ paddingTop: '100%' }} />
        {frames.map((fr, k) => (
          <img
            key={fr.year}
            src={`${api.defaults.baseURL}${overlay ? fr.overlay : fr.rgb}`}
            alt={`${lakeId} ${fr.year}`}
            loading="eager"
            draggable={false}
            className="absolute inset-0 h-full w-full object-cover"
            style={{ opacity: k === idx ? 1 : 0, transition: 'opacity 90ms' }}
          />
        ))}
        <span className="absolute bottom-1 left-1 bg-ink/70 px-1.5 py-0.5 font-sans text-[10px] text-white">
          Sentinel-2 true colour · melt season {f.year}
        </span>
      </div>

      {/* Year-labelled scrubber. Clicking a year jumps straight to it, which is
          the gesture people actually reach for when comparing two dates. */}
      <input
        type="range"
        min={0}
        max={frames.length - 1}
        value={idx}
        onChange={(e) => {
          setPlaying(false)
          setI(Number(e.target.value))
        }}
        className="mt-1.5 w-full accent-[#b4442e]"
      />
      <div className="flex justify-between font-sans text-[10px]">
        {frames.map((fr, k) => (
          <button
            key={fr.year}
            onClick={() => {
              setPlaying(false)
              setI(k)
            }}
            className={
              k === idx
                ? 'font-condensed text-[12px] text-ink'
                : 'text-muted hover:text-ink'
            }
          >
            {fr.year}
          </button>
        ))}
      </div>
      <p className="mt-1 font-sans text-[10px] text-muted">
        Scroll, drag, or ← → to step years · space to play
      </p>
    </div>
  )
}
