# hyper³labs demos

Interactive model demos at **[demos.hyper3labs.com](https://demos.hyper3labs.com/)**.

Each demo lives in a top-level folder. A shared build discovers `*/demo.json`, creates the index, and mounts each demo's `site/` directory under its folder name. Source code, documentation and regeneration tools outside `site/` are not included in the deployed website.

```text
haystack-hyper3-clip-v1/
  demo.json                 # Index title and description
  site/                     # Editable HTML, JS, images and recorded results
  tools/                    # Optional inference and result regeneration
shared/                     # Demo index template, branding and HTTP rules
tools/                      # Shared build and checks
wrangler.jsonc              # One Cloudflare deployment for all demos
dist/                       # Generated; not committed
```

## Develop and deploy

Requires Python 3.12+ and Node.js 20+.

```sh
npm ci
npm run build
npm run check
npm run dev
```

For deployment, authenticate with `npx wrangler login`, then run `npm run deploy`. The existing Cloudflare service is named `hyper3-haystack-demo`; it hosts the entire collection and owns `demos.hyper3labs.com`. Keeping its name preserves the existing domain and deployment history. There is no Docker, container or inference server in the hosted site.

GitHub Actions validates builds and recorded data on pushes and pull requests. Production deployment is manual through Wrangler; no Cloudflare credentials are committed or configured in GitHub.

## Add a demo

1. Create an adjacent folder such as `another-demo/`.
2. Add `demo.json` with a `title` and `description`.
3. Put the public website in `site/index.html`, with assets beneath `site/`. Use links and fetch URLs rooted at `/another-demo/` (or safe relative paths).
4. Run the shared build and checks, then deploy. The index entry is generated automatically.

Keep private notes, credentials, model weights and assets without suitable publication rights out of the repository. Only put files intended for the website in `site/`.

## Available demos

- **[Hyper3-CLIP v1 × Haystack](https://demos.hyper3labs.com/haystack-hyper3-clip-v1/)** — fixed-direction radial image retrieval with a static CLIP comparison. [Source and reproduction](haystack-hyper3-clip-v1/README.md).

Photographs retain their individual Creative Commons licenses and attribution, listed in each demo's credits. The hyper³labs name and logo identify the publisher; their inclusion does not imply endorsement of derivative projects.
