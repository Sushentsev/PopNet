import lyricsgenius
import time
import os
import json

from tqdm import tqdm

save_path = "./genius_examples"

# artist_names = ["Баста (Basta)", "Руки Вверх (Ruki Vverh)", "Артур Пирожков (Arthur Pirozhkov)",
#                 "Макс Корж (Max Korzh)", "Тима Белорусских (Tima Belorusskih)",
#                 "PHARAOH", "Loboda", "Ёлка (Yolka)", "Гражданская оборона (Civil Defense)",
#                 "Noize MC", "Монеточка (Monetochka)", "Кино (Kino)", "MiyaGi & Эндшпиль",
#                 "MORGENSHTERN", "Элджей (Eldzhey)", "Feduk", "MiyaGi", "NILETTO",
#                 "SONIC DEATH", "Арсений Креститель (Arseniy Krestitel’)",
#                 "Увула (Uvula)", "Padla Bear Outfit", "Самое большое простое число (SBPCH)",
#                 "источник (istochnik)", "Botanichesky sad", "Oxxxymiron",
#                 "Электрофорез (Electroforez)", "Буерак (Buerak)",
#                 "Пасош (Pasosh)", "Спасибо (Spasibo)", "прыгай киска (kiska)",
#                 "Kedr Livanskiy", "Забыл повзрослеть (Zabyl Povzroslet)",
#                 "Ушко (Ushko)", "Би-2 (Bi-2)", "Аквариум (Aquarium)", "Аффинаж (Affinage Band)",
#                 "Дайте танк (!) [Daite Tank (!)]", "Комсомольск (Komsomolsk)",
#                 "Молчат Дома (Molchat Doma)", "Клава Кока (Klava Koka)", 
#                 "имя твоей бывшей (imja tvoej byvshej)", "МЭЙБИ БЭЙБИ (MAYBE BABY)",
#                 "дора (mentaldora)"
# ]


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
            for song in tqdm(artist.songs):
                save_lyrics(save_path, artist_name + " - " + song.title, song.lyrics)


def main():
    with open("./genious_config.json") as file:
        CONFIG = json.load(file)

    collect_lyrics(CONFIG, save_path)


if __name__ == "__main__":
    main()
