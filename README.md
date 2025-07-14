# Applied AI Research Blog

A Jekyll-based blog for documenting experiences learning and building Applied AI projects.

## Features

- **Clean, minimalist design** optimized for technical content
- **Syntax highlighting** for code blocks with copy-to-clipboard functionality
- **MathJax support** for mathematical equations
- **Responsive design** that works on all devices
- **Academic citations** and references support
- **Tag and category organization**
- **Reading time estimation**
- **Table of contents** for long posts
- **SEO optimization** with jekyll-seo-tag

## Quick Start

1. **Install dependencies:**
   ```bash
   bundle install
   ```

2. **Run the development server:**
   ```bash
   bundle exec jekyll serve
   ```

3. **View your blog:**
   Open http://localhost:4000 in your browser

## Writing Posts

Posts are written in Markdown and stored in the `_posts` directory. Use the following frontmatter template:

```yaml
---
layout: post
title: "Your Post Title"
date: 2024-01-15 10:00:00 -0500
categories: [category1, category2]
tags: [tag1, tag2, tag3]
author: "Your Name"
math: true  # Enable MathJax for this post
toc: true   # Generate table of contents
reading_time: true
references:
  - title: "Paper Title"
    authors: "Author Name"
    year: 2023
    journal: "Journal Name"
    url: "https://example.com"
---
```

## Configuration

Edit `_config.yml` to customize:
- Site title and description
- Author information
- Social media links
- Domain configuration for deployment

## Deployment

This blog is configured for deployment on AWS using S3 + CloudFront. You can also deploy to:
- GitHub Pages
- Netlify
- Vercel
- Any static hosting provider

## Structure

```
├── _config.yml          # Site configuration
├── _layouts/            # Page templates
├── _includes/           # Reusable components
├── _posts/              # Blog posts
├── _sass/               # Stylesheets
├── assets/              # Static assets
├── about.md             # About page
├── tags.html            # Tags index
├── categories.html      # Categories index
├── archive.html         # Post archive
└── index.html           # Homepage
```

## Customization

- **Styling**: Modify files in `_sass/` directory
- **Layout**: Edit templates in `_layouts/` and `_includes/`
- **JavaScript**: Add custom scripts to `assets/js/main.js`
- **Colors**: Update color variables in `_sass/main.scss`

## License

This blog template is open source and available under the MIT License.