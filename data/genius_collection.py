import lyricsgenius
import time
import os
import json

from tqdm import tqdm

save_path = "./all_lyrics"


def save_lyrics(path: str, author_lyrics_name: str, lyrics: str):
    try:
        with open(os.path.join(save_path, author_lyrics_name + ".txt"), "w+") as file:
            file.write(lyrics + "\n")
    except Exception as e:
        print(e)


# https://github.com/johnwmillr/LyricsGenius
def collect_lyrics(CONFIG, save_path: str):
    genius = lyricsgenius.Genius(CONFIG["access_token"])
    genius.verbose = CONFIG["verbose"]
    genius.remove_section_headers = CONFIG["remove_section_headers"]
    genius.skip_non_songs = CONFIG["skip_non_songs"]
    genius.excluded_terms = CONFIG["excluded_terms"]
    genius.timeout = CONFIG["timeout"]
    artist_names = CONFIG["artist_names"]

    for artist_name in artist_names:
        artist = genius.search_artist(artist_name, max_songs=100, sort="popularity")

        if artist is not None:
            for song in artist.songs:
                save_lyrics(save_path, artist.__name + " - " + song.title, song.lyrics)


def main():
    with open("./genious_config.json") as file:
        CONFIG = json.load(file)

    collect_lyrics(CONFIG, save_path)


if __name__ == "__main__":
    main()
