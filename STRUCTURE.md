# GGUF Compass — Project Structure

gguf-compass/
├── index.html              # Main SPA (single-file, self-contained)
├── README.md               # Epic readme
├── LICENSE                 # MIT
├── CONTRIBUTING.md         # How to contribute
├── .gitignore
├── assets/
│   ├── og-image.png        # Open Graph image (to be generated)
│   └── favicon.svg
├── data/
│   ├── hardware-db.json    # GPU/CPU/RAM tiers database
│   ├── models-db.json      # Curated model catalog
│   └── rules.json          # Fit rules engine
├── docs/
│   ├── knowledge-base.md   # Extracted Codacus + internal knowledge
│   └── presets.md          # Curated presets documentation
└── tools/
    └── validate.py         # Static regression test
