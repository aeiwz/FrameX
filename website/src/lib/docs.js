import fs from 'fs'
import path from 'path'
import matter from 'gray-matter'

const docsDirectory = path.join(process.cwd(), '../docs/documents')
const extraDocFiles = [
  path.join(process.cwd(), '../docs/benchmark-results.md'),
]

function inferTitleFromMarkdown(content, fallback) {
  const match = content.match(/^#\s+(.+)$/m)
  return match ? match[1].trim() : fallback
}

function readDocFileFromPath(fullPath) {
  const slug = path.basename(fullPath).replace(/\.md$/, '')
  const fileContents = fs.readFileSync(fullPath, 'utf8')
  const { data, content } = matter(fileContents)
  const fallbackTitle = slug
    .split(/[-_]/)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ')

  return {
    slug,
    meta: {
      title: data.title || inferTitleFromMarkdown(content, fallbackTitle),
      description: data.description || '',
      order: Number.isFinite(data.order) ? data.order : 999,
      section: data.section || 'Reference',
    },
    content,
  }
}

export function getDocSlugs() {
  const slugs = []
  if (fs.existsSync(docsDirectory)) {
    for (const file of fs.readdirSync(docsDirectory)) {
      if (file.endsWith('.md')) slugs.push(file)
    }
  }

  for (const fullPath of extraDocFiles) {
    if (fs.existsSync(fullPath)) {
      slugs.push(path.basename(fullPath))
    }
  }

  return slugs
}

export function getAllDocs() {
  const docs = []

  if (fs.existsSync(docsDirectory)) {
    const files = fs.readdirSync(docsDirectory).filter((f) => f.endsWith('.md'))
    for (const file of files) {
      docs.push(readDocFileFromPath(path.join(docsDirectory, file)))
    }
  }

  for (const fullPath of extraDocFiles) {
    if (fs.existsSync(fullPath)) {
      docs.push(readDocFileFromPath(fullPath))
    }
  }

  docs.sort((a, b) => {
    if (a.meta.order !== b.meta.order) return a.meta.order - b.meta.order
    return a.meta.title.localeCompare(b.meta.title)
  })
  return docs
}

export function getDocsNav() {
  const docs = getAllDocs()
  const groups = new Map()

  for (const doc of docs) {
    if (!groups.has(doc.meta.section)) {
      groups.set(doc.meta.section, [])
    }
    groups.get(doc.meta.section).push({
      slug: doc.slug,
      title: doc.meta.title,
      description: doc.meta.description,
      order: doc.meta.order,
    })
  }

  return Array.from(groups.entries()).map(([section, items]) => ({
    section,
    items,
  }))
}

export function getDocBySlug(slug) {
  const realSlug = slug.replace(/\.md$/, '')
  const fullPath = path.join(docsDirectory, `${realSlug}.md`)
  if (fs.existsSync(fullPath)) return readDocFileFromPath(fullPath)

  for (const extraPath of extraDocFiles) {
    if (path.basename(extraPath, '.md') === realSlug && fs.existsSync(extraPath)) {
      return readDocFileFromPath(extraPath)
    }
  }

  return null
}
