# Karim Abbas · portfolio

Static site built with [Astro](https://astro.build). Deploys to GitHub Pages at karimabbas.com.

## Develop

```bash
npm install
npm run dev      # http://localhost:4321
npm run build    # static output in dist/
```

## Where things live

- `src/data/site.ts`: name, links, hero, about, publications
- `src/data/projects.ts`: every project write-up (content only, no markup)
- `src/pages/`: home, project template, 404, sitemap, redirects for the old `/project-pages/*.html` URLs
- `src/styles/global.css`: all styles, with light and dark themes
- `public/`: icons, share image (`og.png`), font, `CNAME`, `robots.txt`

The CV lives at `public/Karim_Abbas_CV.pdf`. Replace the file (same name) to update it.
