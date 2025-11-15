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
