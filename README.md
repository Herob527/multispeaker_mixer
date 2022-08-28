# Rust Mixer
Basically, it's based on Olusiowiec's DatasetMixer

This program basically takes every dataset in "datasets" folder and mixes them to create a dataset for training multispeaker models like Flowtron, RADTTS etc.
It doesn't use RADTTS's capability of having name of datasets as strings tho

Each dataset must have "wavs" directory, "list_train.txt" and "list_val.txt"

Program copies files to "mixed_wavs" directory and creates lists in "mixed_lists" directory (if these aren't present, this program can handle it) 

It's strongly recommended to process you audio files using FFmpeg especially if you unpacked wav file from game 
