import './globals.css'
import Link from 'next/link'

export const metadata = {
  title: 'FrameX',
  description: 'High-performance Python library for parallel dataframe and array processing',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        <a href="#main-content" className="skip-link">Skip to content</a>
        <div className="container">
          <nav className="topnav" aria-label="Primary">
            <Link href="/" className="logo">FrameX</Link>
            <div className="links" role="list">
              <Link href="/docs">Docs</Link>
              <Link href="/benchmarks">Benchmarks</Link>
              <Link href="/docs/tutorial_etl_pipeline">Tutorials</Link>
              <Link href="/docs/use_cases">Use Cases</Link>
              <a href="https://github.com/aeiwz/FrameX" target="_blank" rel="noopener noreferrer" aria-label="FrameX on GitHub (opens in new tab)">GitHub</a>
            </div>
          </nav>
          <main id="main-content">
            {children}
          </main>
        </div>
      </body>
    </html>
  )
}
