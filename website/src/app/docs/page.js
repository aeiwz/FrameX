import Link from 'next/link'
import { getDocsNav } from '@/lib/docs'

export const metadata = {
  title: 'Documentation | FrameX',
  description: 'FrameX documentation, tutorials, and use cases.',
}

export default function DocsIndexPage() {
  const nav = getDocsNav()

  return (
    <section>
      <div className="docs-index-header">
        <h1>Documentation</h1>
        <p>
          Learn FrameX step-by-step with practical tutorials, architecture notes,
          and a complete API reference.
        </p>
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
