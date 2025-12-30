#!/usr/bin/env python3
"""
Script to check GitHub repository activity and identify stale projects.
Addresses issue #133: Sort by Activity

This script extracts all GitHub repositories from README.md and checks their
last update date using the GitHub API. It generates a report of inactive
projects that haven't been updated in the specified threshold.
"""

import re
import sys
import os
import json
from datetime import datetime, timezone
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
from collections import defaultdict

README_FILE = "README.md"
INACTIVE_MONTHS_THRESHOLD = 12  # Projects inactive for more than 12 months

def extract_github_repos(readme_path: str) -> list[tuple[str, str, str]]:
    """
    Extract GitHub repository URLs from README.md.
    Returns list of tuples: (project_name, owner, repo)
    """
    repos = []
    # Match GitHub URLs in markdown links or table cells
    # Patterns like: [name](https://github.com/owner/repo) or https://github.com/owner/repo
    github_pattern = re.compile(
        r'https://github\.com/([a-zA-Z0-9_.-]+)/([a-zA-Z0-9_.-]+)'
    )
    
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    seen = set()
    for match in github_pattern.finditer(content):
        owner, repo = match.groups()
        # Clean up repo name (remove trailing characters like ) or ])
        repo = repo.rstrip(')\]"\'')
        key = f"{owner}/{repo}".lower()
        if key not in seen:
            seen.add(key)
            repos.append((f"{owner}/{repo}", owner, repo))
    
    return repos


def get_repo_info(owner: str, repo: str, token: str = None) -> dict | None:
    """
    Get repository information from GitHub API.
    Returns dict with pushed_at, updated_at, archived status, etc.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}"
    headers = {
        'Accept': 'application/vnd.github.v3+json',
        'User-Agent': 'Awesome-LLMOps-Activity-Checker'
    }
    if token:
        headers['Authorization'] = f'token {token}'
    
    try:
        req = Request(url, headers=headers)
        with urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode('utf-8'))
    except HTTPError as e:
        if e.code == 404:
            return {'error': 'not_found', 'message': 'Repository not found'}
        elif e.code == 403:
            return {'error': 'rate_limited', 'message': 'API rate limit exceeded'}
        else:
            return {'error': 'http_error', 'message': str(e)}
    except URLError as e:
        return {'error': 'url_error', 'message': str(e)}
    except Exception as e:
        return {'error': 'unknown', 'message': str(e)}


def calculate_months_since(date_str: str) -> int:
    """Calculate months since the given ISO date string."""
    if not date_str:
        return -1
    try:
        date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)
        diff = now - date
        return diff.days // 30
    except:
        return -1


def main():
    # Check for GitHub token
    token = os.environ.get('GITHUB_TOKEN')
    if not token:
        print("Note: GITHUB_TOKEN not set. API rate limits apply (60 requests/hour).")
        print("Set GITHUB_TOKEN environment variable for higher limits.\n")
    
    # Find README.md
    readme_path = README_FILE
    if not os.path.exists(readme_path):
        readme_path = os.path.join(os.path.dirname(__file__), '..', README_FILE)
    
    if not os.path.exists(readme_path):
        print(f"Error: {README_FILE} not found")
        return 1
    
    print(f"Extracting GitHub repositories from {readme_path}...")
    repos = extract_github_repos(readme_path)
    print(f"Found {len(repos)} unique GitHub repositories.\n")
    
    # Collect activity data
    inactive_repos = []
    active_repos = []
    archived_repos = []
    error_repos = []
    
    for i, (full_name, owner, repo) in enumerate(repos):
        print(f"[{i+1}/{len(repos)}] Checking {full_name}...", end=' ', flush=True)
        
        info = get_repo_info(owner, repo, token)
        
        if info is None:
            print("Error: No response")
            error_repos.append((full_name, "No response"))
            continue
        
        if 'error' in info:
            print(f"Error: {info['message']}")
            error_repos.append((full_name, info['message']))
            if info['error'] == 'rate_limited':
                print("\nRate limit exceeded. Set GITHUB_TOKEN to continue.")
                break
            continue
        
        # Check if archived
        if info.get('archived', False):
            print("Archived")
            archived_repos.append((full_name, info.get('pushed_at', 'unknown')))
            continue
        
        # Check last push date
        pushed_at = info.get('pushed_at')
        months_inactive = calculate_months_since(pushed_at)
        
        if months_inactive >= INACTIVE_MONTHS_THRESHOLD:
            print(f"Inactive ({months_inactive} months)")
            inactive_repos.append((full_name, pushed_at, months_inactive, info.get('stargazers_count', 0)))
        else:
            print(f"Active ({months_inactive} months since last push)")
            active_repos.append((full_name, pushed_at, months_inactive, info.get('stargazers_count', 0)))
    
    # Generate report
    print("\n" + "=" * 60)
    print("ACTIVITY REPORT")
    print("=" * 60)
    
    print(f"\n📊 Summary:")
    print(f"   Total repositories checked: {len(repos)}")
    print(f"   ✅ Active (< {INACTIVE_MONTHS_THRESHOLD} months): {len(active_repos)}")
    print(f"   ⚠️  Inactive (≥ {INACTIVE_MONTHS_THRESHOLD} months): {len(inactive_repos)}")
    print(f"   📦 Archived: {len(archived_repos)}")
    print(f"   ❌ Errors: {len(error_repos)}")
    
    if inactive_repos:
        print(f"\n⚠️  INACTIVE PROJECTS (no updates in {INACTIVE_MONTHS_THRESHOLD}+ months):")
        print("-" * 60)
        # Sort by months inactive, descending
        inactive_repos.sort(key=lambda x: x[2], reverse=True)
        for full_name, pushed_at, months, stars in inactive_repos:
            date_str = pushed_at[:10] if pushed_at else 'unknown'
            print(f"   {full_name}")
            print(f"      Last push: {date_str} ({months} months ago) | ⭐ {stars}")
    
    if archived_repos:
        print(f"\n📦 ARCHIVED PROJECTS:")
        print("-" * 60)
        for full_name, pushed_at in archived_repos:
            date_str = pushed_at[:10] if pushed_at else 'unknown'
            print(f"   {full_name} (last push: {date_str})")
    
    if error_repos:
        print(f"\n❌ PROJECTS WITH ERRORS:")
        print("-" * 60)
        for full_name, error in error_repos:
            print(f"   {full_name}: {error}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
