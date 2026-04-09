import Link from 'next/link'
import { getDocsNav } from '@/lib/docs'

export const metadata = {
  title: 'Documentation | FrameX',
  description: 'FrameX documentation, tutorials, and use cases.',
}

export default function DocsIndexPage() {
  const nav = getDocsNav()

  return (
    <section className="docs-shell">
      <header className="docs-topbar" aria-label="Docs header">
        <div>
          <p className="docs-topbar-eyebrow">FrameX Documentation</p>
          <h1>Documentation</h1>
        </div>
        <nav className="docs-topbar-links" aria-label="Docs quick links">
          <Link href="/docs/features">Features</Link>
          <Link href="/docs/tutorial_etl_pipeline">Tutorial</Link>
          <Link href="/docs/use_cases">Use Case</Link>
          <Link href="/docs/configuration_guide">Configuration</Link>
          <Link href="/docs/performance_test">Performance</Link>
        </nav>
      </header>

      <div className="docs-index-header">
        <p>
          Learn FrameX step-by-step with practical tutorials, architecture notes,
          and a complete API reference.
        </p>
      </div>

      <div className="docs-path-strip" aria-label="Recommended reading order">
        <Link href="/docs/features" className="docs-path-link">1. Feature</Link>
        <Link href="/docs/tutorial_etl_pipeline" className="docs-path-link">2. Tutorial</Link>
        <Link href="/docs/use_cases" className="docs-path-link">3. Use Case</Link>
        <Link href="/docs/configuration_guide" className="docs-path-link">Configuration Guide</Link>
        <Link href="/docs/performance_test" className="docs-path-link">Performance Test</Link>
      </div>

      <div className="docs-index-grid">
        {nav.map((group) => (
          <div className="docs-index-card" key={group.section}>
            <h3>{group.section}</h3>
            <ul>
              {group.items.map((item) => (
                <li key={item.slug}>
                  <Link href={`/docs/${item.slug}`}>{item.title}</Link>
                  {item.description ? <p>{item.description}</p> : null}
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </section>
  )
}
