export type Point = { x: number; y: number }
export type Trajectory = Point[]

export type TemplateStore = Record<string, Trajectory[] | undefined>

export function scoreFromDistance(dist: number): number {
  // Simple monotonic mapping (0..inf) -> (0..1].
  // Tuned to feel similar to "percentage match".
  const d = Math.max(0, dist)
  return 1 / (1 + 6 * d)
}

export function resampleByArcLength(points: Trajectory, numPoints: number): Trajectory {
  if (points.length === 0) return []
  if (points.length === 1) {
    const p = points[0]!
    return Array.from({ length: numPoints }, () => ({ x: p.x, y: p.y }))
  }
  if (numPoints <= 0) return []

  // cumulative distance along polyline
  const cum: number[] = [0]
  for (let i = 1; i < points.length; i++) {
    const a = points[i - 1]!
    const b = points[i]!
    cum.push(cum[i - 1]! + Math.hypot(b.x - a.x, b.y - a.y))
  }
  const total = cum[cum.length - 1]!
  if (total === 0) {
    const p = points[0]!
    return Array.from({ length: numPoints }, () => ({ x: p.x, y: p.y }))
  }

  const out: Trajectory = []
  let j = 0
  for (let i = 0; i < numPoints; i++) {
    const t = (i * total) / (numPoints - 1)
    while (j + 1 < cum.length && cum[j + 1]! < t) j++
    if (j + 1 >= cum.length) {
      const p = points[points.length - 1]!
      out.push({ x: p.x, y: p.y })
      continue
    }
    const t0 = cum[j]!
    const t1 = cum[j + 1]!
    const alpha = t1 === t0 ? 0 : (t - t0) / (t1 - t0)
    const a = points[j]!
    const b = points[j + 1]!
    out.push({ x: (1 - alpha) * a.x + alpha * b.x, y: (1 - alpha) * a.y + alpha * b.y })
  }
  return out
}

export function normalizeTrajectory(points: Trajectory): Trajectory {
  if (points.length === 0) return points

  // Match Python pipeline: resample to fixed length, then normalize + axis align.
  const resampled = resampleByArcLength(points, 100)
  if (resampled.length === 0) return resampled

  const mean = resampled.reduce(
    (acc, p) => ({ x: acc.x + p.x, y: acc.y + p.y }),
    { x: 0, y: 0 },
  )
  const cx = mean.x / resampled.length
  const cy = mean.y / resampled.length

  const centered = resampled.map((p) => ({ x: p.x - cx, y: p.y - cy }))
  let maxNorm = 0
  for (const p of centered) {
    const n = Math.hypot(p.x, p.y)
    if (n > maxNorm) maxNorm = n
  }
  if (maxNorm <= 0) return centered
  const scaled = centered.map((p) => ({ x: p.x / maxNorm, y: p.y / maxNorm }))

  // Align main axis (match Python `trajectory.normalization.normalize`).
  // For 2D, compute principal direction from covariance analytically.
  let sxx = 0
  let syy = 0
  let sxy = 0
  for (const p of scaled) {
    sxx += p.x * p.x
    syy += p.y * p.y
    sxy += p.x * p.y
  }
  const n = scaled.length
  if (n >= 2) {
    const inv = 1 / (n - 1)
    const a = sxx * inv
    const d = syy * inv
    const b = sxy * inv

    // Eigenvector for largest eigenvalue of [[a,b],[b,d]]
    // Use angle formula: theta = 0.5 * atan2(2b, a-d)
    const theta = 0.5 * Math.atan2(2 * b, a - d)
    const c = Math.cos(-theta)
    const s = Math.sin(-theta)
    return scaled.map((p) => ({ x: c * p.x - s * p.y, y: s * p.x + c * p.y }))
  }

  return scaled
}

export function capPoints(points: Trajectory, maxPoints: number): Trajectory {
  if (!maxPoints || maxPoints <= 0) return points
  const n = points.length
  if (n <= maxPoints) return points
  const out: Trajectory = []
  for (let i = 0; i < maxPoints; i++) {
    const t = (i * (n - 1)) / (maxPoints - 1)
    const idx = Math.floor(t)
    out.push(points[idx]!)
  }
  return out
}

export function dtwDistance(a: Trajectory, b: Trajectory): number {
  const n = a.length
  const m = b.length
  if (n === 0 || m === 0) return Number.POSITIVE_INFINITY

  // O(nm) DP with rolling rows.
  const prev = new Float32Array(m + 1)
  const curr = new Float32Array(m + 1)
  for (let j = 0; j <= m; j++) prev[j] = Number.POSITIVE_INFINITY
  prev[0] = 0

  for (let i = 1; i <= n; i++) {
    curr[0] = Number.POSITIVE_INFINITY
    const ai = a[i - 1]!
    for (let j = 1; j <= m; j++) {
      const bj = b[j - 1]!
      const d = Math.hypot(ai.x - bj.x, ai.y - bj.y)
      const bestPrev = Math.min(prev[j], curr[j - 1], prev[j - 1])
      curr[j] = d + bestPrev
    }
    prev.set(curr)
  }

  const dist = prev[m]!
  const norm = n + m
  return norm > 0 ? dist / norm : dist
}

export type HandResult = {
  // normalized landmarks in [0..1] relative to image size
  landmarks?: Array<{ x: number; y: number }>
}

export class DrawingSession {
  private wasPinching = false
  private drawing = false
  private prevPoint: Point | null = null
  private smoothed: Point | null = null
  private paths: Trajectory[] = []
  private currentPath: Trajectory | null = null

  // Tunables similar to the Python version
  private readonly pinchEnter = 0.042
  private readonly pinchExit = 0.062
  private readonly smoothingAlpha = 0.25
  private readonly smoothingDeadzonePx = 4
  private readonly sampleMinPx = 6

  clear(): void {
    this.wasPinching = false
    this.drawing = false
    this.prevPoint = null
    this.smoothed = null
    this.paths = []
    this.currentPath = null
  }

  isDrawing(): boolean {
    return this.drawing
  }

  lastPoint(): Point | null {
    return this.smoothed
  }

  flattenedPath(): Trajectory {
    const out: Trajectory = []
    for (const stroke of this.paths) out.push(...stroke)
    return out
  }

  allPaths(): Trajectory[] {
    return this.paths
  }

  updateFromHandResult(res: HandResult | null, width: number, height: number): void {
    const lm = res?.landmarks
    if (!lm || lm.length < 9) {
      this.prevPoint = null
      this.smoothed = null
      if (this.wasPinching) {
        this.drawing = false
        this.currentPath = null
      }
      this.wasPinching = false
      return
    }

    const thumb = lm[4]!
    const tip = lm[8]!
    const m = Math.min(width, height)
    const pinchDist = Math.hypot((thumb.x - tip.x) * width, (thumb.y - tip.y) * height) / m

    const pinchActive = nextPinchState(this.wasPinching, pinchDist, this.pinchEnter, this.pinchExit)

    // Mirror x to match the mirrored camera preview (and the Python pipeline,
    // which flips the frame before running MediaPipe).
    const raw: Point = { x: Math.round(width - tip.x * width), y: Math.round(tip.y * height) }

    // smoothing (EMA with deadzone)
    if (!this.smoothed) {
      this.smoothed = raw
    } else {
      const ax = this.smoothed.x
      const ay = this.smoothed.y
      const dx = raw.x - ax
      const dy = raw.y - ay
      const dist = Math.hypot(dx, dy)
      if (dist >= this.smoothingDeadzonePx) {
        const sx = Math.round(ax + this.smoothingAlpha * dx)
        const sy = Math.round(ay + this.smoothingAlpha * dy)
        this.smoothed = { x: sx, y: sy }
      }
    }

    this.updatePath(this.smoothed, pinchActive)
  }

  private updatePath(point: Point, pinchActive: boolean): void {
    if (pinchActive && !this.wasPinching) {
      this.drawing = true
      this.currentPath = []
      this.paths.push(this.currentPath)
      this.currentPath.push(point)
    } else if (!pinchActive && this.wasPinching) {
      this.drawing = false
      this.currentPath = null
    }
    this.wasPinching = pinchActive

    if (!this.prevPoint) {
      this.prevPoint = point
      return
    }

    if (this.drawing && pinchActive && this.currentPath) {
      const dx = point.x - this.prevPoint.x
      const dy = point.y - this.prevPoint.y
      const dist = Math.hypot(dx, dy)
      if (dist >= this.sampleMinPx) {
        const steps = Math.floor(dist / this.sampleMinPx)
        if (steps <= 1) {
          this.currentPath.push(point)
        } else {
          const x0 = this.prevPoint.x
          const y0 = this.prevPoint.y
          const x1 = point.x
          const y1 = point.y
          for (let k = 1; k <= steps; k++) {
            const t = k / steps
            this.currentPath.push({
              x: Math.round(x0 + t * (x1 - x0)),
              y: Math.round(y0 + t * (y1 - y0)),
            })
          }
        }
      }
    }

    this.prevPoint = point
  }
}

export function nextPinchState(
  wasPinching: boolean,
  dist: number,
  enter: number,
  exit: number,
): boolean {
  if (wasPinching) return dist <= exit
  return dist <= enter
}

