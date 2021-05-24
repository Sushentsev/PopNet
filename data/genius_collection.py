import lyricsgenius
import time

from tqdm import tqdm

save_path = "/home/denis/Study/PopNet/data/genius_examples/"
access_token = "jvi-7jH-bkBdiSeZXv6KLSPA-UTZsa9ZnoglH_VY8D6lXdpjD6flzuCQX_0By-IR"

artist_names = ["Дима Билан", "Филипп Киркоров", "Полина Гагарина",
                "Сергей Лазарев", "Алла Пугачева", "Николай Басков (Nikolay Baskov)"]


def save_lyrics(path: str, author_lyrics_name: str, lyrics: str):
    with open(path + author_lyrics_name + ".txt", "w") as file:
        file.write(lyrics + "\n")


# https://github.com/johnwmillr/LyricsGenius
def collect_lyrics(save_path: str):
    genius = lyricsgenius.Genius(access_token)
    genius.verbose = True  # Turn off status messages
    genius.remove_section_headers = True  # Remove section headers (e.g. [Chorus]) from lyrics when searching
    genius.skip_non_songs = True  # Include hits thought to be non-songs (e.g. track lists)
    genius.excluded_terms = ["(Remix)", "(Live)"]  # Exclude songs with these words in their title
    genius.timeout = 20

    for artist_name in artist_names:
        artist = genius.search_artist(artist_name, max_songs=10, sort="popularity")

        for song in artist.songs:
            save_lyrics(save_path, artist_name + " - " + song.title, song.lyrics)


def main():
    collect_lyrics(save_path)


if __name__ == "__main__":
    main()
