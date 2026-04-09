import fs from 'fs'
import path from 'path'
import matter from 'gray-matter'

const docsDirectory = path.join(process.cwd(), '../docs/documents')

export function getDocSlugs() {
  if (!fs.existsSync(docsDirectory)) return []
  return fs.readdirSync(docsDirectory).filter(f => f.endsWith('.md'))
}

export function getDocBySlug(slug) {
  const realSlug = slug.replace(/\.md$/, '')
  const fullPath = path.join(docsDirectory, `${realSlug}.md`)
  if (!fs.existsSync(fullPath)) {
    return null;
  }
  const fileContents = fs.readFileSync(fullPath, 'utf8')
  
  const { data, content } = matter(fileContents)

  return {
    slug: realSlug,
    meta: data,
    content,
  }
}
