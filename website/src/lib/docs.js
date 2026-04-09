import fs from 'fs'
import path from 'path'
import matter from 'gray-matter'

const docsDirectory = path.join(process.cwd(), '../docs/documents')

function readDocFile(file) {
  const slug = file.replace(/\.md$/, '')
  const fullPath = path.join(docsDirectory, file)
  const fileContents = fs.readFileSync(fullPath, 'utf8')
  const { data, content } = matter(fileContents)

  return {
    slug,
    meta: {
      title: data.title || slug,
      description: data.description || '',
      order: Number.isFinite(data.order) ? data.order : 999,
      section: data.section || 'Docs',
    },
    content,
  }
}

export function getDocSlugs() {
  if (!fs.existsSync(docsDirectory)) return []
  return fs.readdirSync(docsDirectory).filter((f) => f.endsWith('.md'))
}

export function getAllDocs() {
  const files = getDocSlugs()
  const docs = files.map(readDocFile)
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
  if (!fs.existsSync(fullPath)) return null
  return readDocFile(`${realSlug}.md`)
}
