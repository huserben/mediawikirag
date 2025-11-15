# MediaWiki API fetcher for Wiki RAG

import requests


class MediaWikiFetcher:

    def __init__(self, api_url):
        self.api_url = api_url


    def fetch_all_pages(self, max_retries=3, backoff=2, batch_size=50):
        """
        Fetches all pages from the MediaWiki API using pagination (gapcontinue).
        Returns a list of page dicts with title, content, revision id, and timestamp.
        Retries on network/API errors.
        """
        import time
        S = requests.Session()
        params = {
            'action': 'query',
            'format': 'json',
            'generator': 'allpages',
            'gaplimit': batch_size,
            'prop': 'revisions',
            'rvprop': 'ids|timestamp|content',
        }
        all_results = []
        gapcontinue = None
        while True:
            if gapcontinue:
                params['gapcontinue'] = gapcontinue
            else:
                params.pop('gapcontinue', None)
            attempt = 0
            while attempt < max_retries:
                try:
                    response = S.get(self.api_url, params=params, timeout=10)
                    response.raise_for_status()
                    data = response.json()
                    
                    pages = data.get('query', {}).get('pages', {})
                    for page in pages.values():
                        title = page.get('title')
                        revisions = page.get('revisions', [{}])
                        content = revisions[0].get('*', '') if revisions else ''
                        # Clean wiki markup: remove section headings, image/file
                        # links and reference tags so downstream chunking and
                        # embedding only use plain article text.
                        content = self.clean_content(content)
                        revid = revisions[0].get('revid', None)
                        timestamp = revisions[0].get('timestamp', None)
                        all_results.append({
                            'title': title,
                            'content': content,
                            'revid': revid,
                            'timestamp': timestamp
                        })
                    break
                except Exception:
                    attempt += 1
                    if attempt < max_retries:
                        time.sleep(backoff * attempt)
                    else:
                        raise
            if 'continue' in data and 'gapcontinue' in data['continue']:
                gapcontinue = data['continue']['gapcontinue']
            else:
                break
        return all_results

    def clean_content(self, content: str) -> str:
        """
        Remove headings (== ... ==), image/file links ([[File:...]]),
        <ref>...</ref> tags, <references/> and basic HTML tags from the
        raw wikitext/content returned by the MediaWiki API. Returns a
        cleaned plain-text string.
        """
        import re

        if not content:
            return content

        # Remove section headings like == Heading == or === Sub ===
        content = re.sub(r"^={2,}.*={2,}\s*$", "", content, flags=re.M)

        # Remove File/Image links like [[File:Example.jpg|...]]
        pattern_file = r"\[\[(?:File|Image):[^\[\]]*\]\]"
        content = re.sub(pattern_file, "", content, flags=re.I)

        # Remove <ref>...</ref> and self-closing <ref /> tags
        content = re.sub(
            r"<ref[^>]*>.*?</ref>",
            "",
            content,
            flags=re.S | re.I,
        )
        content = re.sub(r"<ref[^>]*/>", "", content, flags=re.I)

        # Remove <references/> and <references>...</references>
        content = re.sub(
            r"<references[^>]*>.*?</references>",
            "",
            content,
            flags=re.S | re.I,
        )
        content = re.sub(r"<references[^>]*/>", "", content, flags=re.I)

        # Remove any remaining HTML tags
        content = re.sub(r"<[^>]+>", "", content)

        # Remove common image/file templates like {{Infobox ...}} minimally
        # (keep simple: remove {{File:...}} variants)
        content = re.sub(r"\{\{ ?[Ff]ile:[^}]*\}\}", "", content)

        # Collapse multiple blank lines and trim
        content = re.sub(r"\n{2,}", "\n\n", content).strip()

        return content
