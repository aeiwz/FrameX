import Link from 'next/link'
import { getBenchmarkReportData } from '@/lib/benchmarks'

export const metadata = {
  title: 'Benchmark Report | FrameX',
  description: 'Native vs FrameX benchmark comparison across performance, parallel processing, memory, and C backend.',
}

const categoryMeta = [
  { key: 'performance', title: 'Performance', subtitle: 'Native vs FrameX operation timings.' },
  { key: 'parallel_processing', title: 'Parallel Processing', subtitle: 'Threaded kernel scaling by worker count.' },
  { key: 'single_core', title: 'Single Core', subtitle: 'Single-worker baseline overhead and throughput.' },
  { key: 'multiprocessing', title: 'Multiprocessing', subtitle: 'Process-based object-heavy workload scaling.' },
  { key: 'memory', title: 'Memory', subtitle: 'Runtime and peak RSS comparisons.' },
  { key: 'c_backend', title: 'C Backend', subtitle: 'FrameX Python backend vs C backend kernels.' },
]

function formatSeconds(value) {
  if (value == null) return '—'
  return value < 0.001 ? `${(value * 1_000_000).toFixed(1)} µs` : `${(value * 1000).toFixed(3)} ms`
}

function formatSpeedup(value) {
  if (value == null) return '—'
  return `${value.toFixed(3)}x`
}

function formatRss(value) {
  if (value == null) return '—'
  return `${value.toFixed(2)} MB`
}

function formatMillis(seconds) {
  if (seconds == null || Number.isNaN(seconds)) return '—'
  return `${(seconds * 1000).toFixed(2)} ms`
}

function winnerLabel(row) {
  if (!row.winner) return '—'
  return row.winner === row.rightEngine ? 'FrameX Side' : 'Baseline Side'
}

function statusLabel(status) {
  if (status === 'pass') return 'PASS'
  if (status === 'partial') return 'PARTIAL'
  if (status === 'not_applicable') return 'N/A'
  return 'FAIL'
}

function HighlightCard({ title, row }) {
  if (!row) return null
  return (
    <article className="benchmark-highlight-card">
      <h3>{title}</h3>
      <p className="benchmark-highlight-scenario">
        {row.scenarioLabel} ({row.workers} worker{row.workers > 1 ? 's' : ''})
      </p>
      <p className="benchmark-highlight-metric">
        <strong>{formatSpeedup(row.speedup)}</strong>
        <span>baseline / FrameX</span>
      </p>
    </article>
  )
}

export default function BenchmarksPage() {
  const data = getBenchmarkReportData()

  if (!data.available) {
    return (
      <section className="benchmark-page">
        <header className="benchmark-header">
          <p className="benchmark-eyebrow">Benchmark Report</p>
          <h1>No Benchmark Data Found</h1>
          <p>Run the benchmark suite first, then refresh this page.</p>
          <pre><code>python3 -m benchmarks.benchmark_suite --rows 300000 --repeats 3 --warmups 1 --workers 1,2,4,8</code></pre>
        </header>
      </section>
    )
  }

  return (
    <section className="benchmark-page">
      <header className="benchmark-header">
        <p className="benchmark-eyebrow">Benchmark Report</p>
        <h1>FrameX vs Native: Performance Dashboard</h1>
        <p>
          Snapshot from the local benchmark suite. Use this report to track wins, regressions,
          and where optimization work should focus next.
        </p>
        <div className="benchmark-run-block">
          <p><strong>Run Performance Test</strong></p>
          <pre><code>python3 -m benchmarks.benchmark_suite</code></pre>
        </div>
        <div className="benchmark-meta">
          <span><strong>Rows Captured:</strong> {data.rowCount}</span>
          <span><strong>Generated:</strong> {data.generatedAt || 'Unknown'}</span>
          <Link href="/docs/benchmark-results" className="button-outline">Read Full Narrative</Link>
          <Link href="/docs/performance_test" className="button-outline">Performance Test Guide</Link>
        </div>
      </header>

      <section className="benchmark-highlights" aria-labelledby="benchmark-highlights-title">
        <h2 id="benchmark-highlights-title">Highlights</h2>
        <div className="benchmark-highlight-grid">
          <HighlightCard title="Best FrameX Speedup" row={data.highlights?.bestFramex} />
          <HighlightCard title="Slowest FrameX Case" row={data.highlights?.slowestFramex} />
          <HighlightCard title="Best C Backend Case" row={data.highlights?.bestCBackend} />
        </div>
      </section>

      <section className="benchmark-sections" aria-label="Benchmark comparison tables">
        {categoryMeta.map((meta) => {
          const rows = data.categories[meta.key] || []
          if (!rows.length) return null

          return (
            <article key={meta.key} className="benchmark-section-card">
              <div className="benchmark-section-head">
                <h2>{meta.title}</h2>
                <p>{meta.subtitle}</p>
              </div>

              <div className="benchmark-table-wrap">
                <table className="benchmark-table">
                  <thead>
                    <tr>
                      <th scope="col">Scenario</th>
                      <th scope="col">Workers</th>
                      <th scope="col">{rows[0].leftEngine}</th>
                      <th scope="col">{rows[0].rightEngine}</th>
                      <th scope="col">Speedup</th>
                      <th scope="col">Winner</th>
                      <th scope="col">RSS ({rows[0].leftEngine}/{rows[0].rightEngine})</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((row) => (
                      <tr key={`${meta.key}-${row.scenario}-${row.workers}`}>
                        <th scope="row">{row.scenarioLabel}</th>
                        <td>{row.workers}</td>
                        <td>{formatSeconds(row.leftSeconds)}</td>
                        <td>{formatSeconds(row.rightSeconds)}</td>
                        <td>{formatSpeedup(row.speedup)}</td>
                        <td>{winnerLabel(row)}</td>
                        <td>{formatRss(row.leftPeakRssMb)} / {formatRss(row.rightPeakRssMb)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </article>
          )
        })}
      </section>

      {data.workloadChecks?.length ? (
        <section className="benchmark-workloads" aria-labelledby="workload-matrix-title">
          <div className="benchmark-section-head">
            <h2 id="workload-matrix-title">Workload Capability Matrix Check</h2>
            <p>Automated runtime checks from <code>benchmarks.check_framex_workloads</code>.</p>
          </div>

          <div className="benchmark-table-wrap">
            <table className="benchmark-table">
              <thead>
                <tr>
                  <th scope="col">Workload</th>
                  <th scope="col">Status</th>
                  <th scope="col">Runtime</th>
                  <th scope="col">Detail</th>
                </tr>
              </thead>
              <tbody>
                {data.workloadChecks.map((row) => (
                  <tr key={row.workload}>
                    <th scope="row">{row.workload}</th>
                    <td>
                      <span className={`status-pill status-${row.status}`}>{statusLabel(row.status)}</span>
                    </td>
                    <td>{formatMillis(row.seconds)}</td>
                    <td>{row.detail}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}
    </section>
  )
}
