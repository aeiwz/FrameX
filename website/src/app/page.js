import Link from 'next/link'

export default function Home() {
  return (
    <div>
      <section className="hero">
        <h1>High-Performance Dataframes.</h1>
        <p>Combining Pandas and NumPy semantics with an Arrow-native multiprocessing runtime for local data processing.</p>
        <div style={{ marginTop: '2rem' }}>
          <Link href="/docs/getting_started" className="button">Get Started</Link>
          <a href="https://github.com/aeiwz/FrameX" className="button-outline" target="_blank" rel="noopener noreferrer">View on GitHub</a>
        </div>
      </section>

      <section className="grid">
        <div className="card">
          <h3>Arrow-Native Zero-Copy</h3>
          <p>Built directly on PyArrow's in-memory format, avoiding slow object serialization and providing frictionless interoperability across tools.</p>
        </div>
        <div className="card">
          <h3>Pandas & NumPy Semantics</h3>
          <p>Offers <code>DataFrame</code>, <code>Series</code>, and chunked <code>NDArray</code> abstractions with methods you already know, dropping right into native workflows.</p>
        </div>
        <div className="card">
          <h3>Smart Parallel Executor</h3>
          <p>Automatically heuristically selects between multi-threading (for GIL-releasing numeric logic) and multi-processing (for Python object ops).</p>
        </div>
        <div className="card">
          <h3>Eager & Lazy execution</h3>
          <p>Operations are eager by default for easy debugging, with an opt-in <code>.lazy()</code> interface to construct performant query graphs.</p>
        </div>
      </section>
    </div>
  )
}
