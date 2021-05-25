import lyricsgenius
import time
import os

from tqdm import tqdm

save_path = "/content/PopNet/data/genius_examples/"
access_token = "jvi-7jH-bkBdiSeZXv6KLSPA-UTZsa9ZnoglH_VY8D6lXdpjD6flzuCQX_0By-IR"

# artist_names = ["Дима Билан", "Филипп Киркоров", "Полина Гагарина",
#                 "Сергей Лазарев", "Алла Пугачева", "Николай Басков (Nikolay Baskov)"]

artist_names = ["Баста (Basta)", "Руки Вверх (Ruki Vverh)", "Артур Пирожков (Arthur Pirozhkov)",
                "Макс Корж (Max Korzh)", "Тима Белорусских (Tima Belorusskih)",
                "PHARAOH", "Loboda", "Ёлка (Yolka)", "Гражданская оборона (Civil Defense)",
                "Noize MC", "Монеточка (Monetochka)", "Кино (Kino)", "MiyaGi & Эндшпиль",
                "MORGENSHTERN", "Элджей (Eldzhey)", "Feduk", "MiyaGi", "NILETTO",
                "SONIC DEATH", "Арсений Креститель (Arseniy Krestitel’)",
                "Увула (Uvula)", "Padla Bear Outfit", "Самое большое простое число (SBPCH)",
                "источник (istochnik)", "Botanichesky sad", "Oxxxymiron",
                "Электрофорез (Electroforez)", "Буерак (Buerak)",
                "Пасош (Pasosh)", "Спасибо (Spasibo)", "прыгай киска (kiska)",
                "Kedr Livanskiy", "Забыл повзрослеть (Zabyl Povzroslet)",
                "Ушко (Ushko)", "Би-2 (Bi-2)", "Аквариум (Aquarium)", "Аффинаж (Affinage Band)",
                "Дайте танк (!) [Daite Tank (!)]", "Комсомольск (Komsomolsk)",
                "Молчат Дома (Molchat Doma)", "Клава Кока (Klava Koka)", 
                "имя твоей бывшей (imja tvoej byvshej)", "МЭЙБИ БЭЙБИ (MAYBE BABY)",
                "дора (mentaldora)"
]

def save_lyrics(path: str, author_lyrics_name: str, lyrics: str):
    try:
        with open(os.path.join(save_path, author_lyrics_name + ".txt"), "w+") as file:
            file.write(lyrics + "\n")
    except Exception as e:
        print(e)


# https://github.com/johnwmillr/LyricsGenius
def collect_lyrics(save_path: str):
    genius = lyricsgenius.Genius(access_token)
    genius.verbose = True  # Turn off status messages
    genius.remove_section_headers = True  # Remove section headers (e.g. [Chorus]) from lyrics when searching
    genius.skip_non_songs = True  # Include hits thought to be non-songs (e.g. track lists)
    genius.excluded_terms = ["(Remix)", "(Live)"]  # Exclude songs with these words in their title
    genius.timeout = 20

    for artist_name in artist_names:
        artist = genius.search_artist(artist_name, max_songs=100, sort="popularity")

        for song in tqdm(artist.songs):
            save_lyrics(save_path, artist_name + " - " + song.title, song.lyrics)


def main():
    collect_lyrics(save_path)


if __name__ == "__main__":
    main()
