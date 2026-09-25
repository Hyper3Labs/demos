# Demo repository conventions

Write the company name as hyper³labs. Keep each demo in its own top-level folder with `demo.json` and `site/`; adjacent demos must not overwrite one another. The shared build discovers descriptors automatically.

Only `site/` contents are published to the website. Keep credentials, local histories, model weights and unlicensed assets out of Git. Preserve image attribution and individual licenses. Do not describe precomputed results as live inference or rank movement as validated model superiority.

After changes, run `npm run build` and `npm run check`. Deploy using the root Wrangler configuration, preserving existing URLs and other demos. Copy-only changes do not require model inference or regeneration of result data.
