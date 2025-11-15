from fetcher import MediaWikiFetcher

api_url = "https://simpsons.fandom.com/de/api.php"
fetcher = MediaWikiFetcher(api_url)
pages = fetcher.fetch_all_pages(limit=2)  # Fetch 2 pages for a quick test

for page in pages:
    print(f"Title: {page['title']}")
    print(f"Content: {page['content'][:200]}...")  # Print first 200 chars
    print("-" * 40)
