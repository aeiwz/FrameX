import Link from 'next/link'

export default function Home() {
  return (
    <div className="home-page">
      <section className="hero" aria-labelledby="hero-title">
        <p className="hero-eyebrow">Single-Machine Analytics Engine</p>
        <h1 id="hero-title">Parallel Dataframes. Purposeful UX. Zero-Nonsense Workflows.</h1>
        <p>
          FrameX combines Pandas-like tables, NumPy-compatible arrays, and Arrow-native memory
          so teams can process medium-to-large datasets without jumping straight to a cluster.
        </p>
        <div className="hero-actions">
          <Link href="/docs/getting_started" className="button">Get Started</Link>
          <Link href="/docs/tutorial_etl_pipeline" className="button-outline">Run ETL Tutorial</Link>
        </div>
        <ul className="hero-metrics">
          <li><strong>Arrow-first</strong><span>Columnar internals</span></li>
          <li><strong>Eager + Lazy</strong><span>Choose execution style</span></li>
          <li><strong>NDArray Interop</strong><span>NumPy protocol support</span></li>
        </ul>
        <div className="hero-install" aria-label="Install and import">
          <pre><code>pip install pyframe-xpy</code></pre>
          <pre><code>import framex as fx</code></pre>
        </div>
      </section>

      <section className="grid" aria-label="Platform capabilities">
        <div className="card">
          <h3>Arrow-Native Core</h3>
          <p>
            Tables are backed by Arrow for efficient columnar operations,
            cleaner interoperability, and better memory behavior.
          </p>
        </div>

        <div className="card">
          <h3>DataFrame + NDArray</h3>
          <p>
            Use `DataFrame`, `Series`, and chunked `NDArray` in one workflow,
            with NumPy protocol support included.
          </p>
        </div>

        <div className="card">
          <h3>Eager and Lazy Modes</h3>
          <p>
            Keep day-to-day work intuitive with eager execution, then switch to
            lazy pipelines when transformations become complex.
          </p>
        </div>

        <div className="card">
          <h3>Real-World File IO</h3>
          <p>
            Ship data across parquet, ORC, SQLite, JSON, CSV, fixed-width text,
            and export-ready HTML/XML with one `read_file` / `write_file` surface.
          </p>
        </div>

        <div className="card">
          <h3>Guides That Ship</h3>
          <p>
            Full docs now include onboarding, practical tutorials, use cases,
            architecture details, and API reference material.
          </p>
        </div>
      </section>

      <section className="quick-links" aria-labelledby="quick-links-title">
        <h2 id="quick-links-title">Start Here</h2>
        <div className="quick-links-grid">
          <Link href="/docs/features" className="quick-link">Features</Link>
          <Link href="/docs/getting_started" className="quick-link">Getting Started</Link>
          <Link href="/docs/tutorial_etl_pipeline" className="quick-link">Tutorial: ETL Pipeline</Link>
          <Link href="/docs/tutorial_numpy_array" className="quick-link">Tutorial: NumPy NDArray</Link>
          <Link href="/docs/use_cases" className="quick-link">Use Cases</Link>
          <Link href="/docs/configuration_guide" className="quick-link">Configuration Guide</Link>
          <Link href="/docs/sqlite_guide" className="quick-link">SQLite Guide</Link>
          <Link href="/docs/performance_test" className="quick-link">Performance Test</Link>
          <Link href="/benchmarks" className="quick-link">Benchmark Report</Link>
          <Link href="/docs/architecture" className="quick-link">Architecture</Link>
          <Link href="/docs/api_reference" className="quick-link">API Reference</Link>
        </div>
      </section>

      <section className="learning-path" aria-labelledby="learning-path-title">
        <h2 id="learning-path-title">Learning Path</h2>
        <div className="learning-path-grid">
          <Link href="/docs/features" className="path-step">
            <span className="path-order">1</span>
            <strong>Feature</strong>
            <p>Understand FrameX capabilities and architecture fit.</p>
          </Link>
          <Link href="/docs/tutorial_etl_pipeline" className="path-step">
            <span className="path-order">2</span>
            <strong>Tutorial</strong>
            <p>Run practical pipelines and learn API patterns quickly.</p>
          </Link>
          <Link href="/docs/use_cases" className="path-step">
            <span className="path-order">3</span>
            <strong>Use Case</strong>
            <p>Map FrameX to ETL, analytics, and ML preprocessing workloads.</p>
          </Link>
        </div>
      </section>
    </div>
  )
}
