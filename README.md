# PopNet
Generating pop lyrics from song titles

# Data

## Data Collection
Data was collected from [Genius](https://genius.com/). Genius is one of the largest song lyrics databases. Right now there are 4638 songs collected.
We predominantly chose pop and rock artists, because they have similar song structures, so algorithms can learn better.
To collect lyrics we used [lyricsgenius](https://github.com/johnwmillr/lyricsgenius) package

Data saved in `/data/` folder. 
```
data
├───examples                # songs parsed from random sites
├───genius examples         # songs parsed from genius
|   genious_config.json     # info for parsing like artist names, lyricsgenius package info
|   genius_collection.py    # script for parsing songs from genius
|   gensongs.zip            # zip file with 4638 songs parsed
|   lyrics_collection.py    # script for parsing songs from random sites
```
