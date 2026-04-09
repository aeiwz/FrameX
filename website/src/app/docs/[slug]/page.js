import { getDocBySlug, getDocSlugs } from '@/lib/docs'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import Link from 'next/link'

export function generateStaticParams() {
  const files = getDocSlugs()
  return files.map((file) => ({
    slug: file.replace(/\.md$/, ''),
  }))
}

export default function DocPage({ params }) {
  const doc = getDocBySlug(params.slug)

  if (!doc) {
    return <div>Doc not found</div>
  }

  return (
    <div className="docs-layout">
      <aside className="sidebar">
        <ul>
          <li><Link href="/docs/getting_started">Getting Started</Link></li>
          <li><Link href="/docs/architecture">Architecture</Link></li>
          <li><Link href="/docs/api_reference">API Reference</Link></li>
        </ul>
      </aside>
      <article className="prose">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{doc.content}</ReactMarkdown>
      </article>
    </div>
  )
}
