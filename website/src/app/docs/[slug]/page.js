import { getDocBySlug, getDocSlugs, getDocsNav } from '@/lib/docs'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import Link from 'next/link'

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
    <div className="docs-layout">
      <aside className="sidebar" aria-label="Documentation navigation">
        {nav.map((group) => (
          <div key={group.section} className="sidebar-group">
            <h4>{group.section}</h4>
            <ul>
              {group.items.map((item) => (
                <li key={item.slug}>
                  <Link
                    href={`/docs/${item.slug}`}
                    className={item.slug === doc.slug ? 'active' : ''}
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
        {doc.meta.description ? <p className="doc-description">{doc.meta.description}</p> : null}
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{doc.content}</ReactMarkdown>
      </article>
    </div>
  )
}
