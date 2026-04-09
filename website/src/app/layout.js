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
        <div className="container">
          <nav>
            <Link href="/" className="logo">FrameX</Link>
            <div className="links">
              <Link href="/docs/getting_started">Docs</Link>
              <a href="https://github.com/aeiwz/FrameX" target="_blank" rel="noopener noreferrer">GitHub</a>
            </div>
          </nav>
          <main>
            {children}
          </main>
        </div>
      </body>
    </html>
  )
}
