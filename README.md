# PopNet
Generating pop lyrics from song titles

# Data Collection

Data was collected from [Genius](https://genius.com/). Genius is one of the largest song lyrics databases. Right now there are 4638 songs collected.
We predominantly chose pop and rock artists, because they have similar song structures, so algorithms can learn better.
To collect lyrics we used [lyricsgenius](https://github.com/johnwmillr/lyricsgenius) package

Data saved in `/data/` folder. 
```
data
|   genious_config.json     # info for parsing like artist names, lyricsgenius package info
|   genius_collection.py    # script for parsing songs from genius
|   gensongs.zip            # zip file with 4638 songs parsed
|   lyrics_collection.py    # script for parsing songs from random sites
├─── examples               # songs parsed from random sites
├─── genius examples        # songs parsed from genius
```

# ru-GPT 3 Small

First model we trained was ru-GPT 3 Small. We successfully finetuned this model on Colab.
To train model we used folowing `<tags>`:
`<startsong>` - the begining of the song
`<songname>` - song name
`<songlyrics>` - the beginning of the lyrics
`<endsong>` - end of the song

To run model you should create `train.txt` and `valid.txt` files with songs with tags.
To train model you should run folowing script:
```
!export PYTHONPATH=${PYTHONPATH}:/ru-gpts/
!CUDA_VISIBLE_DEVICES=0 python ru-gpts/pretrain_transformers.py \
    --output_dir=models/popnet \
    --model_type=gpt2 \
    --model_name_or_path=sberbank-ai/rugpt3small_based_on_gpt2 \
    --do_train \
    --train_data_file=train.txt \
    --do_eval \
    --eval_data_file=valid.txt \
    --per_gpu_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 5 \
    --block_size 2048 \
    --overwrite_output_dir
```

You can find an example of using the model in `/examples/` folder.

Lyrics example:

### **Дом (Home)**
```
Я хочу тебе рассказать, 
Как я жил без тебя 
И не знал о том что такое больная любовь. 
Я хотел бы чтобы ты была рядом со мной... 
Но только в моем сердце остались те чувства,
Которые мы с тобой потеряли навсегда! 
А так хочется чтобы кто-то пришел к нам жить,
А может быть это будет наш первый секс после долгой разлуки!!! 
Так хотелось забыть обо всем на свете,
Оставив все позади нас двоих!! 
И вот уже который год подряд мне снится 
Моя новая жизнь, где каждый день новый мой друг живет у меня за спиной.. 
Мы живем словно во сне… 
Мне снились твои руки под моим окном, 
Когда он спит ночью, просыпаясь от моих криков: 
"Сука!" 
Ты слышишь мои крики - они звучат для всех одинаково
Это дом нашей мечты! 
Мой любимый человек теперь твой самый лучший враг номер один из тысячи людей,
Которым можно доверять по жизни....
```

![Home](https://previews.123rf.com/images/fizkes/fizkes1902/fizkes190200753/116534838-desperate-upset-teen-girl-victim-crying-alone-at-home-sad-abused-young-woman-in-tears-feeling-depres.jpg)
