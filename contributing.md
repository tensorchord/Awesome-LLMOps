# Contribution Guidelines

Please ensure your pull request adheres to the following guidelines:

- New categories or improvements to the existing categorization are welcome.
- Search previous suggestions before making a new one, as yours may be a duplicate.
- Make an individual pull request for each suggestion.
    - Run `./scripts/generate-star-badges.py` to generate Github star badges if needed.
    - Run `./scripts/github-markdown-toc ./README.md` to generate ToC if needed.
    - Run `./scripts/check-activity.py` to check repository activity status (requires `GITHUB_TOKEN` env var for full results).
- Order link titles alphabetically within each category.
- Mark archived repositories with *(Archived)* in the description.
- Remove repositories that no longer exist (404 errors).

Thank you for your suggestions!
