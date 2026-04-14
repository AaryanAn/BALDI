import type { TemplateStore, Trajectory } from './gesture'

const KEY = 'baldi.templates.v1'
const PY_TEMPLATES_URL = '/templates.json'

export function loadTemplates(): TemplateStore {
  try {
    const raw = localStorage.getItem(KEY)
    if (!raw) return {}
    const parsed = JSON.parse(raw)
    if (!parsed || typeof parsed !== 'object') return {}
    return parsed as TemplateStore
  } catch {
    return {}
  }
}

export async function loadPythonTemplates(): Promise<TemplateStore> {
  try {
    const res = await fetch(PY_TEMPLATES_URL, { cache: 'no-store' })
    if (!res.ok) return {}
    const parsed = await res.json()
    if (!parsed || typeof parsed !== 'object') return {}
    return coerceTemplateStore(parsed)
  } catch {
    return {}
  }
}

function coerceTrajectory(maybe: unknown): Trajectory {
  if (!Array.isArray(maybe)) return []
  // Accept either [{x,y}, ...] or [[x,y], ...]
  const out: Trajectory = []
  for (const p of maybe) {
    if (p && typeof p === 'object' && 'x' in p && 'y' in p) {
      const x = Number((p as any).x)
      const y = Number((p as any).y)
      if (Number.isFinite(x) && Number.isFinite(y)) out.push({ x, y })
      continue
    }
    if (Array.isArray(p) && p.length >= 2) {
      const x = Number(p[0])
      const y = Number(p[1])
      if (Number.isFinite(x) && Number.isFinite(y)) out.push({ x, y })
    }
  }
  return out
}

function coerceTemplateStore(maybe: unknown): TemplateStore {
  if (!maybe || typeof maybe !== 'object') return {}
  const out: TemplateStore = {}
  for (const [k, v] of Object.entries(maybe as Record<string, unknown>)) {
    if (!Array.isArray(v) || v.length === 0) continue
    const key = k.toUpperCase()
    const templates: Trajectory[] = []
    for (const t of v) {
      const traj = coerceTrajectory(t)
      if (traj.length > 0) templates.push(traj)
    }
    if (templates.length > 0) out[key] = templates
  }
  return out
}

export function mergeTemplateStores(a: TemplateStore, b: TemplateStore): TemplateStore {
  const out: TemplateStore = { ...a }
  for (const [k, v] of Object.entries(b)) {
    if (!v || v.length === 0) continue
    const key = k.toUpperCase()
    const existing = out[key] ? [...out[key]!] : []
    out[key] = existing.concat(v)
  }
  return out
}

export function saveTemplates(store: TemplateStore): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(store))
  } catch {
    // ignore quota / private mode errors
  }
}

