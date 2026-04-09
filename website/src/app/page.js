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
          <Link href="/docs/getting_started" className="quick-link">Getting Started</Link>
          <Link href="/docs/tutorial_etl_pipeline" className="quick-link">Tutorial: ETL Pipeline</Link>
          <Link href="/docs/tutorial_numpy_array" className="quick-link">Tutorial: NumPy NDArray</Link>
          <Link href="/docs/use_cases" className="quick-link">Use Cases</Link>
          <Link href="/docs/architecture" className="quick-link">Architecture</Link>
          <Link href="/docs/api_reference" className="quick-link">API Reference</Link>
        </div>
      </section>
    </div>
  )
}
