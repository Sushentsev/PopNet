import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from typing import List, Tuple

host = "https://text-you.ru/"
path = "rus_text_pesni/"


def save_lyrics(path: str, author_lyrics_name: str, lyrics: List[str]):
    with open(path + author_lyrics_name + ".txt", "w") as file:
        for line in lyrics:
            file.write(line + "\n")


def parse_lyrics_info(url: str) -> Tuple[str, List[str]]:
    page = requests.get(url)
    soup = BeautifulSoup(page.text, "html.parser")

    author_lyrics_name = soup.title.contents[0][12:]
    lyrics = [str(line).strip() for line in soup.pre.contents if str(line) != "<br/>"]
    return author_lyrics_name, lyrics


def remove_duplicated_links(links: List[str]) -> List[str]:
    filtered_links = []

    for i, link in enumerate(links):
        if (i == 0) or (links[i] != links[i - 1]):
            filtered_links.append(link)

    return filtered_links


def parse_lyrics_links(url: str) -> List[str]:
    page = requests.get(url)
    soup = BeautifulSoup(page.text, "html.parser")
    paths = []

    for link in soup.find_all("a", href=True):
        href = link["href"]
        if href.startswith("/rus_text_pesni/") and href.endswith(".html"):
            paths.append(href)

    paths = remove_duplicated_links(paths)
    links = [host + lyr_path[1:] for lyr_path in paths]
    return links


# Lyrics collecting from text-you.ru/rus_text_pesni
# Page = 40 lyrics.
def collect_lyrics(save_path: str, pages: int = None):
    page_numbers = range(1, min(300, pages + 1))

    for page_number in tqdm(page_numbers):
        page_link = host + path + "page/" + str(page_number)
        lyrics_links = parse_lyrics_links(page_link)

        for lyrics_link in lyrics_links:
            author_lyrics_name, lyrics = parse_lyrics_info(lyrics_link)
            save_lyrics(save_path, author_lyrics_name, lyrics)


def main():
    collect_lyrics("/home/denis/Study/PopNet/data/examples/", pages=1)


if __name__ == '__main__':
    main()
