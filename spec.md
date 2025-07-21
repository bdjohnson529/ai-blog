# Research Blog Specification

## 1. Project Overview & Goals

**Purpose:** Technical blog for documenting experiences learning and building Applied AI projects.

**Target Audience:** 
- Applied AI practitioners
- Machine learning researchers
- Potential collaborators
- Potential employers

## 2. User Stories

As a publisher:
- I want to write posts in Markdown and save them in a structured way.
- I want to be able to navigate quickly through my posts and find relevant content.
- I want to be able to include mathematical equations to demonstrate ML concepts.
- I want to include code snippets in my posts with syntax highlighting and copy functionality.
- I want to include citations and references to academic papers and resources.

As a designer:
- I want the blog to have a clean, professional look.
- I want the blog to be responsive and accessible on all devices.
- I want the blog to have a minimalist design that focuses on content.

As a maintainer:
- I want to be able to deploy the blog easily.
- I want to ensure the blog is performant and SEO optimized.

As a reader:
- I want to be able to copy code snippets so that I can test them in my own environment.
- I want to easily navigate through blog posts by category and tags.


## 3. Functional Requirements

### Functional Requirements

The blog must support:
- writing posts in Markdown format.
- syntax highlighting for code snippets.
- rendering mathematical equations using LaTeX or MathJax.
- including citations and references to academic papers.
- a clean, professional design that is responsive.

### Non-Functional Requirements

1. The blog must load within 3 seconds on a standard broadband connection.
2. The blog must be optimized for SEO.
3. The blog must be deployable to AWS using S3 and CloudFront.


### Out of Scope

- User authentication and management
- Commenting system
- Analytics tracking
- Video content
- Animations

## 4. Technical Requirements

**Static Site Generator:** Jekyll 4.x

**Hosting Platform:** AWS
- Primary: S3 + CloudFront + Route 53

**Deployment Method:**
- CI/CD pipeline using GitHub Actions

**Domain:** [YOUR_DOMAIN_HERE.com]

**Metadata Requirements:**
- Publication date and last updated
- Reading time estimation
- Tag system for topics
- Category classification

## 5. Design & UX Specifications

**Theme Approach:** 
- Clean, minimalist academic design
- Inspired by: [SPECIFY REFERENCE SITES/THEMES]
- Custom theme or base: [Minimal Mistakes / Academic Pages / Custom]

**Color Scheme:**
- Primary: [SPECIFY PRIMARY COLOR]
- Secondary: [SPECIFY SECONDARY COLOR]
- Background: Clean white/light gray
- Code blocks: Dark theme with syntax highlighting
- Links: Distinctive but professional

**Typography:**
- Body text: [SPECIFY FONT - e.g., Source Sans Pro]
- Headings: [SPECIFY FONT - e.g., Roboto Slab]
- Code: Fira Code or JetBrains Mono
- Math: Computer Modern (LaTeX default)

**Navigation:**
- Fixed header with main navigation
- Sidebar with recent posts, tags, and search
- Breadcrumb navigation for deep pages
- "Back to top" functionality on long posts

**Features:**
- Dark mode toggle (optional)
- Print-friendly CSS
- Social sharing buttons
- Table of contents for long posts


## 6. Project Management
[X] Initial version
[X] CI/CD pipeline setup
[X] Deployment

[ ] Styling