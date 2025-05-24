import os
from yt_dlp import YoutubeDL

# Define search keywords for each swimming stroke.
# These keywords will be used to query YouTube for relevant videos.
stroke_keywords = {
    "freestyle": [
        "freestyle swimming slow motion",
        "front crawl slow motion",
        "freestyle stroke slow motion",
        "freestyle swimming technique"
    ],
    "butterfly": [
        "butterfly stroke slow motion",
        "butterfly swimming slow motion",
        "butterfly swimming technique"
    ],
    "backstroke": [
        "backstroke swimming slow motion",
        "backstroke stroke slow motion",
        "backstroke swimming technique"
    ],
    "breaststroke": [
        "breaststroke stroke slow motion",
        "breaststroke swimming slow motion",
        "breaststroke swimming technique"
    ],
}

# Directory where all downloaded videos will be stored, organized by stroke.
download_dir = "videos"
os.makedirs(download_dir, exist_ok=True)

def search_youtube_videos(query, max_results=3):
    """
    Search YouTube for videos matching the query and return a list of video URLs.

    Args:
        query (str): The search query string.
        max_results (int): Number of top results to return.

    Returns:
        list of str: List of YouTube video URLs.
    """
    ydl_opts = {
        'quiet': True,             # Suppress verbose output
        'skip_download': True,     # Only search, do not download
        'extract_flat': True,      # Do not resolve playlists
        'forcejson': True,
        'dump_single_json': True,
        'default_search': 'ytsearch',
    }
    with YoutubeDL(ydl_opts) as ydl:
        # ytsearchN:<query> fetches N results for the query
        search_query = f"ytsearch{max_results}:{query}"
        result = ydl.extract_info(search_query, download=False)
        # Extract video URLs from the search results
        video_urls = [entry['url'] for entry in result['entries']]
        return video_urls

def download_video(url, output_path):
    """
    Download a single YouTube video in mp4 format to the specified directory.

    Args:
        url (str): YouTube video URL.
        output_path (str): Directory to save the downloaded video.
    """
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',  # Download best quality mp4
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),  # Save as <title>.mp4
        'quiet': True,
        'no_warnings': True
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def main():
    # Loop through each stroke and its associated search queries
    for style, queries in stroke_keywords.items():
        # Create a subdirectory for each stroke
        style_dir = os.path.join(download_dir, style)
        os.makedirs(style_dir, exist_ok=True)

        for query in queries:
            print(f"[🔍] Searching: {query}")
            try:
                # Search for video URLs using the current query
                video_urls = search_youtube_videos(query)
                for url in video_urls:
                    # Some results may be video IDs, not full URLs
                    if url.startswith("http"):
                        video_url = url
                    else:
                        video_url = f"https://www.youtube.com/watch?v={url}"

                    print(f"[⬇️ ] Downloading: {video_url}")
                    download_video(video_url, style_dir)
            except Exception as e:
                print(f"[!] Error: {e}")

if __name__ == "__main__":
    main()
