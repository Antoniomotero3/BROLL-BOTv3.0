import requests
import time

def duckduckgo_image_search(query, max_results=10):
    search_url = "https://duckduckgo.com/"
    params = {"q": query}

    # First request to get the token
    res = requests.post(search_url, data=params)
    if res.status_code != 200:
        print(f"❌ Failed to get DuckDuckGo token for query: '{query}'")
        return []

    # Try to extract the vqd token from the response
    try:
        vqd_token = res.text.split("vqd=\"", 1)[1].split("\"", 1)[0]
    except IndexError:
        print(f"❌ DuckDuckGo token not found in response for: '{query}'")
        return []

    # Now use the token to search for images
    image_url = "https://duckduckgo.com/i.js"
    image_params = {"q": query, "vqd": vqd_token, "o": "json", "l": "us-en", "p": "1", "s": "0"}

    try:
        image_res = requests.get(image_url, headers={"User-Agent": "Mozilla/5.0"}, params=image_params)
        image_res.raise_for_status()
        image_data = image_res.json()
        return [result['image'] for result in image_data.get("results", [])[:max_results]]
    except Exception as e:
        print(f"❌ DuckDuckGo search failed for '{query}': {e}")
        return []

def wikimedia_image_search(query, max_results=10):
    search_url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "imageinfo",
        "generator": "search",
        "gsrsearch": query,
        "gsrlimit": max_results,
        "iiprop": "url"
    }

    try:
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        return [v["imageinfo"][0]["url"] for v in pages.values() if "imageinfo" in v]
    except Exception as e:
        print(f"❌ Wikimedia search failed for '{query}': {e}")
        return []
