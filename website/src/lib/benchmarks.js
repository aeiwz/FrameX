import fs from 'fs'
import path from 'path'

const resultsDir = path.join(process.cwd(), '../benchmarks/results')
const resultsJsonPath = path.join(resultsDir, 'benchmark_results.json')
const reportMdPath = path.join(resultsDir, 'benchmark_report.md')

function readJson(pathname) {
  if (!fs.existsSync(pathname)) return null
  return JSON.parse(fs.readFileSync(pathname, 'utf8'))
}

function readGeneratedAt() {
  if (!fs.existsSync(reportMdPath)) return null
  const content = fs.readFileSync(reportMdPath, 'utf8')
  const match = content.match(/Generated:\s+(.+)/)
  return match ? match[1].trim() : null
}

function prettifyScenario(name) {
  return name
    .replaceAll('_', ' ')
    .replace(/\b\w/g, (ch) => ch.toUpperCase())
}

function toNumber(value) {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function buildComparisons(rows, category) {
  const pair =
    category === 'c_backend'
      ? ['python_backend', 'c_backend']
      : ['native', 'framex']

  const grouped = new Map()
  for (const row of rows) {
    if (row.category !== category) continue
    const key = `${row.scenario}::${row.workers}`
    if (!grouped.has(key)) {
      grouped.set(key, { scenario: row.scenario, workers: row.workers })
    }
    grouped.get(key)[row.engine] = row
  }

  const output = []
  for (const group of grouped.values()) {
    const left = group[pair[0]]
    const right = group[pair[1]]
    if (!left || !right) continue

    const leftSeconds = toNumber(left.seconds)
    const rightSeconds = toNumber(right.seconds)
    const speedup = leftSeconds && rightSeconds ? leftSeconds / rightSeconds : null
    const winner =
      speedup == null ? null : speedup >= 1 ? pair[1] : pair[0]

    output.push({
      scenario: group.scenario,
      scenarioLabel: prettifyScenario(group.scenario),
      workers: group.workers,
      leftEngine: pair[0],
      rightEngine: pair[1],
      leftSeconds,
      rightSeconds,
      speedup,
      winner,
      leftPeakRssMb: toNumber(left.peak_rss_mb),
      rightPeakRssMb: toNumber(right.peak_rss_mb),
    })
  }

  output.sort((a, b) => {
    if (a.scenario === b.scenario) return a.workers - b.workers
    return a.scenario.localeCompare(b.scenario)
  })
  return output
}

export function getBenchmarkReportData() {
  const rows = readJson(resultsJsonPath)
  if (!Array.isArray(rows)) {
    return {
      available: false,
      generatedAt: null,
      categories: {},
      highlights: null,
      rowCount: 0,
    }
  }

  const categories = {
    performance: buildComparisons(rows, 'performance'),
    parallel_processing: buildComparisons(rows, 'parallel_processing'),
    single_core: buildComparisons(rows, 'single_core'),
    multiprocessing: buildComparisons(rows, 'multiprocessing'),
    memory: buildComparisons(rows, 'memory'),
    c_backend: buildComparisons(rows, 'c_backend'),
  }

  const nonC = Object.values(categories)
    .filter((value, idx) => idx < 5)
    .flat()
    .filter((item) => item.speedup != null)

  const cRows = categories.c_backend.filter((item) => item.speedup != null)

  const bestFramex = nonC.reduce(
    (best, row) => (!best || row.speedup > best.speedup ? row : best),
    null
  )
  const slowestFramex = nonC.reduce(
    (worst, row) => (!worst || row.speedup < worst.speedup ? row : worst),
    null
  )
  const bestCBackend = cRows.reduce(
    (best, row) => (!best || row.speedup > best.speedup ? row : best),
    null
  )

  return {
    available: true,
    generatedAt: readGeneratedAt(),
    categories,
    rowCount: rows.length,
    highlights: {
      bestFramex,
      slowestFramex,
      bestCBackend,
    },
  }
}

