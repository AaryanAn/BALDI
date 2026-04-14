import { mkdir, readFile, readdir, writeFile } from 'node:fs/promises'
import path from 'node:path'
import process from 'node:process'
import Npyjs from 'npyjs'

const repoRoot = path.resolve(process.cwd(), '..')
const templatesRoot = path.join(repoRoot, 'src', 'templates')
const outDir = path.join(process.cwd(), 'public')
const outFile = path.join(outDir, 'templates.json')

async function listDirs(dir) {
  const entries = await readdir(dir, { withFileTypes: true })
  return entries.filter((e) => e.isDirectory()).map((e) => path.join(dir, e.name))
}

async function listFiles(dir) {
  const entries = await readdir(dir, { withFileTypes: true })
  return entries.filter((e) => e.isFile()).map((e) => path.join(dir, e.name))
}

function toPoints(parsed) {
  const { data, shape } = parsed
  if (!data || !shape || shape.length !== 2 || shape[1] !== 2) return null
  const n = shape[0]
  const out = new Array(n)
  for (let i = 0; i < n; i++) {
    out[i] = [Number(data[i * 2]), Number(data[i * 2 + 1])]
  }
  return out
}

async function main() {
  const npy = new Npyjs()

  const labelDirs = await listDirs(templatesRoot)
  const store = {}

  for (const dir of labelDirs) {
    const label = path.basename(dir)
    const files = await listFiles(dir)
    const npys = files.filter((f) => f.toLowerCase().endsWith('.npy'))
    if (npys.length === 0) continue

    const templates = []
    for (const f of npys) {
      try {
        const buf = await readFile(f)
        const parsed = await npy.load(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength))
        const pts = toPoints(parsed)
        if (pts && pts.length) templates.push(pts)
      } catch {
        // ignore unreadable templates
      }
    }
    if (templates.length) store[label.toUpperCase()] = templates
  }

  await mkdir(outDir, { recursive: true })
  await writeFile(outFile, JSON.stringify(store))
  // eslint-disable-next-line no-console
  console.log(`Wrote ${Object.keys(store).length} labels to ${outFile}`)
}

main().catch((e) => {
  // eslint-disable-next-line no-console
  console.error(e)
  process.exit(1)
})

