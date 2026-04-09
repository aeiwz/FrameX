import { getDocBySlug, getDocSlugs, getDocsNav } from '@/lib/docs'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import Link from 'next/link'
import Mermaid from '@/components/Mermaid'

export function generateStaticParams() {
  const files = getDocSlugs()
  return files.map((file) => ({
    slug: file.replace(/\.md$/, ''),
  }))
}

export async function generateMetadata({ params }) {
  const { slug } = await params
  const doc = getDocBySlug(slug)
  if (!doc) {
    return {
      title: 'Doc Not Found | FrameX',
    }
  }

  return {
    title: `${doc.meta.title} | FrameX Docs`,
    description: doc.meta.description,
  }
}

export default async function DocPage({ params }) {
  const { slug } = await params
  const doc = getDocBySlug(slug)
  const nav = getDocsNav()

  if (!doc) {
    return <div>Doc not found</div>
  }

  return (
    <div className="docs-shell">
      <header className="docs-topbar" aria-label="Docs header">
        <div>
          <p className="docs-topbar-eyebrow">FrameX Documentation</p>
          <h1>{doc.meta.title}</h1>
        </div>
        <nav className="docs-topbar-links" aria-label="Docs quick links">
          <Link href="/docs/features">Features</Link>
          <Link href="/docs/tutorial_etl_pipeline">Tutorial</Link>
          <Link href="/docs/use_cases">Use Case</Link>
          <Link href="/docs/configuration_guide">Configuration</Link>
          <Link href="/docs/performance_test">Performance</Link>
        </nav>
      </header>

      <div className="docs-layout">
        <aside className="sidebar">
          {nav.map((group) => (
            <div className="sidebar-group" key={group.section}>
              <h4>{group.section}</h4>
              <ul>
                {group.items.map((item) => (
                  <li key={item.slug}>
                    <Link
                      href={`/docs/${item.slug}`}
                      className={item.slug === slug ? 'active' : undefined}
                    >
                      {item.title}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </aside>

        <article className="prose">
          {doc.meta?.description ? <p className="doc-description">{doc.meta.description}</p> : null}
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              code(props) {
                const { children, className, node, ...rest } = props
                const match = /language-(\w+)/.exec(className || '')
                if (match && match[1] === 'mermaid') {
                  return <Mermaid chart={String(children).replace(/\n$/, '')} />
                }
                return (
                  <code {...rest} className={className}>
                    {children}
                  </code>
                )
              },
            }}
          >
            {doc.content}
          </ReactMarkdown>
        </article>
      </div>
    </div>
  )
}
