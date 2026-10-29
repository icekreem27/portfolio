import type { APIRoute } from 'astro';
import { projects } from '../data/projects';

export const GET: APIRoute = ({ site }) => {
  const base = site!.href.replace(/\/$/, '');
  const urls = ['/', ...projects.map((p) => `/projects/${p.slug}/`)];
  const xml = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${urls.map((u) => `  <url><loc>${base}${u}</loc></url>`).join('\n')}
</urlset>
`;
  return new Response(xml, { headers: { 'Content-Type': 'application/xml' } });
};
