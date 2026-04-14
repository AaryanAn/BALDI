import { useEffect, useMemo, useRef, useState } from 'react'
import {
  DrawingSession,
  dtwDistance,
  normalizeTrajectory,
  scoreFromDistance,
  type TemplateStore,
  type Trajectory,
} from './lib/gesture'
import { loadHandTracker, type HandTracker } from './lib/handTracker'
import { loadPythonTemplates, loadTemplates, mergeTemplateStores, saveTemplates } from './lib/templates'
import './App.css'

type TopItem = { label: string; distance: number; score: number }

function clampLabel(raw: string): string {
  const s = raw.trim().toUpperCase()
  if (!s) return ''
  if (s.length !== 1) return ''
  const c = s.charCodeAt(0)
  if (c < 65 || c > 90) return ''
  return s
}

export default function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const overlayRef = useRef<HTMLCanvasElement | null>(null)
  const rafRef = useRef<number | null>(null)

  const [handTracker, setHandTracker] = useState<HandTracker | null>(null)
  const [cameraError, setCameraError] = useState<string | null>(null)
  const [status, setStatus] = useState<string>('Loading hand tracker…')
  const [labelRaw, setLabelRaw] = useState<string>('')
  const label = useMemo(() => clampLabel(labelRaw), [labelRaw])

  const [templates, setTemplates] = useState<TemplateStore>(() => loadTemplates())
  const sessionRef = useRef<DrawingSession>(new DrawingSession())

  const [predText, setPredText] = useState<string>('')
  const [topText, setTopText] = useState<string>('')
  const [scoreText, setScoreText] = useState<string>('')

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const tracker = await loadHandTracker()
        if (cancelled) return
        setHandTracker(tracker)
        setStatus('Starting camera…')
      } catch (e) {
        setStatus('Failed to load hand tracker')
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      const py = await loadPythonTemplates()
      if (cancelled) return
      if (Object.keys(py).length > 0) {
        setTemplates((cur) => mergeTemplateStores(py, cur))
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (!handTracker) return

    let stream: MediaStream | null = null
    let stopped = false

    const start = async () => {
      try {
        const s = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        })
        if (stopped) {
          s.getTracks().forEach((t) => t.stop())
          return
        }
        stream = s
        const v = videoRef.current
        if (!v) return
        v.srcObject = s
        await v.play()
        setStatus('Ready')
        loop()
      } catch (e) {
        setCameraError('Camera permission denied or unavailable.')
        setStatus('Camera error')
      }
    }

    const drawOverlay = (paths: Trajectory[], isDrawing: boolean) => {
      const canvas = overlayRef.current
      const video = videoRef.current
      if (!canvas || !video) return

      const w = video.videoWidth || 1280
      const h = video.videoHeight || 720
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w
        canvas.height = h
      }
      const ctx = canvas.getContext('2d')
      if (!ctx) return
      ctx.clearRect(0, 0, w, h)

      // stroke
      const anyStroke = paths.some((p) => p.length >= 2)
      if (anyStroke) {
        ctx.save()
        ctx.lineWidth = 4
        ctx.lineCap = 'round'
        ctx.lineJoin = 'round'
        ctx.strokeStyle = '#ffe066'
        ctx.shadowColor = 'rgba(0,0,0,0.25)'
        ctx.shadowBlur = 6

        for (const traj of paths) {
          if (traj.length < 2) continue
          ctx.beginPath()
          ctx.moveTo(traj[0].x, traj[0].y)
          for (let i = 1; i < traj.length; i++) ctx.lineTo(traj[i].x, traj[i].y)
          ctx.stroke()
        }
        ctx.restore()
      }

      // dot
      const tip = sessionRef.current.lastPoint()
      if (tip) {
        ctx.save()
        ctx.fillStyle = isDrawing ? '#ff3b30' : '#34c759'
        ctx.beginPath()
        ctx.arc(tip.x, tip.y, 8, 0, Math.PI * 2)
        ctx.fill()
        ctx.restore()
      }
    }

    const loop = () => {
      if (stopped) return
      const video = videoRef.current
      if (!video || video.readyState < 2) {
        rafRef.current = requestAnimationFrame(loop)
        return
      }

      const res = handTracker.detect(video)
      sessionRef.current.updateFromHandResult(res, video.videoWidth, video.videoHeight)
      drawOverlay(sessionRef.current.allPaths(), sessionRef.current.isDrawing())

      rafRef.current = requestAnimationFrame(loop)
    }

    start()

    return () => {
      stopped = true
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      if (stream) stream.getTracks().forEach((t) => t.stop())
    }
  }, [handTracker])

  const clearDrawing = () => {
    sessionRef.current.clear()
    setPredText('')
    setTopText('')
    setScoreText('')
  }

  const saveAsTemplate = () => {
    const lbl = label
    if (!lbl) {
      setScoreText('Enter a single uppercase letter A–Z.')
      return
    }
    const traj = normalizeTrajectory(sessionRef.current.flattenedPath())
    if (traj.length < 5) {
      setScoreText('No stroke captured yet.')
      return
    }
    const next: TemplateStore = { ...templates }
    const list = next[lbl] ? [...next[lbl]!] : []
    list.push(traj)
    next[lbl] = list
    setTemplates(next)
    saveTemplates(next)
    setScoreText(`Saved template for “${lbl}”.`)
  }

  const evaluate = () => {
    const traj = normalizeTrajectory(sessionRef.current.flattenedPath())
    if (traj.length < 5) {
      setPredText('No path to evaluate.')
      setTopText('')
      setScoreText('')
      return
    }

    const top: TopItem[] = []
    for (const [lbl, tpls] of Object.entries(templates)) {
      if (!tpls || tpls.length === 0) continue
      let best = Number.POSITIVE_INFINITY
      for (const t of tpls) {
        const d = dtwDistance(traj, t)
        if (d < best) best = d
      }
      top.push({ label: lbl, distance: best, score: scoreFromDistance(best) })
    }
    top.sort((a, b) => a.distance - b.distance)

    const best = top[0]
    if (!best) {
      setPredText('No templates saved yet.')
      setTopText('')
      setScoreText('')
      return
    }

    const bestPct = Math.round(best.score * 100)
    setPredText(`Best guess: “${best.label}” (${bestPct}% template match)`)
    setTopText(
      top
        .slice(0, 3)
        .map((t, i) => `${i + 1}. “${t.label}”: ${Math.round(t.score * 100)}%`)
        .join('\n'),
    )

    const lbl = label
    if (!lbl) {
      setScoreText('')
      return
    }
    const tpls = templates[lbl]
    if (!tpls || tpls.length === 0) {
      setScoreText(`No template for “${lbl}”. Use uppercase A–Z, or save your drawing as a template.`)
      return
    }
    let bestDist = Number.POSITIVE_INFINITY
    for (const t of tpls) bestDist = Math.min(bestDist, dtwDistance(traj, t))
    setScoreText(
      `Your label “${lbl}”: ${Math.round(scoreFromDistance(bestDist) * 100)}% template match (${tpls.length} saved).`,
    )
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header__inner">
          <div>
            <div className="brand">BALDI</div>
            <div className="subtitle">JavaScript prototype (browser-only)</div>
          </div>
          <div className="status">
            <span className={`dot ${status === 'Ready' ? 'dot--ok' : 'dot--warn'}`} />
            {status}
          </div>
        </div>
      </header>

      <main className="shell">
        <section className="left">
          <div className="card">
            <div className="card__head">
              <div>
                <div className="card__title">Camera preview</div>
                <div className="pill">Hand (RGB)</div>
              </div>
            </div>

            <div className="tip">
              <div className="tip__title">Best distance for hand tracking</div>
              <div className="tip__text">
                Hold your hand far enough from the webcam that your full hand fits comfortably in the frame — about
                arm’s length usually works best.
              </div>
            </div>

            {cameraError ? <div className="error">{cameraError}</div> : null}

            <div className="videoWrap">
              <video ref={videoRef} className="video" playsInline muted />
              <canvas ref={overlayRef} className="overlay" />
            </div>

            <div className="caption">Pinch thumb + index to draw. Release to stop.</div>

            <div className="row row--end">
              <button className="btn btn--ghost" onClick={clearDrawing}>
                Clear
              </button>
            </div>
          </div>
        </section>

        <section className="right">
          <div className="card card--accent">
            <div className="card__head">
              <div className="card__title">Recognition</div>
              <div className="pill">DTW</div>
            </div>

            <div className="help">
              Templates use uppercase English letters A–Z only. Save your own strokes to improve matching.
            </div>

            <div className="sectionTitle">How to draw</div>
            <div className="help">
              Other fingers in a loose fist; only thumb and index move. Step back so your whole hand stays in frame.
            </div>

            <div className="sectionTitle">Language</div>
            <div className="help">English</div>

            <label className="field">
              <div className="field__label">Letter label (optional)</div>
              <input
                value={labelRaw}
                onChange={(e) => setLabelRaw(e.target.value)}
                placeholder="A"
                className="input"
              />
              <div className="field__hint">Uppercase A–Z only (auto-uppercased).</div>
            </label>

            <div className="mono block">{predText}</div>
            <div className="mono block">{topText}</div>
            <div className="block">{scoreText}</div>

            <div className="col">
              <button className="btn" onClick={saveAsTemplate}>
                Save as template
              </button>
              <button className="btn" onClick={evaluate}>
                Evaluate
              </button>
            </div>
          </div>
        </section>
      </main>
    </div>
  )
}
