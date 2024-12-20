![bd13c6f5-7a3f-4262-9891-4259f17abbe0](https://github.com/user-attachments/assets/3147c3f9-a886-4bf9-b1cf-ce4781c82601)

![piporin](https://github.com/user-attachments/assets/693ea0aa-44d3-42a0-b4b4-95ec37af8879)

A fast, local neural text to speech system that sounds great and is re-worked and optimized to execute training on Jetson AGX Orin Platforms using the latest cuda enabled torch container.

![image](https://github.com/user-attachments/assets/4e8c566e-3120-48cb-9fd2-9e14f5b823b4)


## PRIMARY CHANGES
### 1. UPDATED [__main__.py](https://github.com/robit-man/piper-train-jetson/blob/master/src/python/piper_train/__main__.py) in piper/src/python/piper_train
### 2. UPDATED [lightening.py](https://github.com/robit-man/piper-train-jetson/blob/master/src/python/piper_train/vits/lightning.py) in piper/src/python/piper_train/vits/
### 3. UPDATED [commons.py](https://github.com/robit-man/piper-train-jetson/blob/master/src/python/piper_train/vits/commons.py) in piper/src/python/piper_train/vits/
### 4. ADDED [setup.sh](https://github.com/robit-man/piper-train-jetson/blob/master/setup.sh) to be run next to /datasets/dataset/wav + metadata.csv

## Use setup.sh to train a model, and run the [PIPER-TTS Jetson Container](https://github.com/dusty-nv/jetson-containers/tree/master/packages/speech/piper-tts) to use the model
1. Create a folder for the project
2. CD into said folder
3. Create a folder called 'datasets', then create a sub-folder called 'dataset'. This is where you will put your 'wav' folder and 'metadata.csv' containing the dataset you wish to use
4. run ```curl -fsSL https://raw.githubusercontent.com/robit-man/piper-train-jetson/master/setup.sh -o setup.sh && chmod +x setup.sh && ./setup.sh``` to pull and execute the setup file in the base folder you created first
5. Run [jtop](https://pypi.org/project/jetson-stats/) to monitor system utilization
6. Increase 'segment_size' on line 51 of lightening.py if you have a 64GB orin to ```segment_size: int = 8192```, or leave as ```segment_size: int = 4096``` if using a 32GB orin

## Use [this repo](https://github.com/robit-man/synthetic_speech_dataset_generator) to generate a synthetic dataset using elevenlabs

### Below is the remaining original content in the [PIPER](https://github.com/rhasspy/piper/tree/master) readme

Piper is used in a [variety of projects](#people-using-piper).

``` sh
echo 'Welcome to the world of speech synthesis!' | \
  ./piper --model en_US-lessac-medium.onnx --output_file welcome.wav
```

[Listen to voice samples](https://rhasspy.github.io/piper-samples) and check out a [video tutorial by Thorsten Müller](https://youtu.be/rjq5eZoWWSo)

Voices are trained with [VITS](https://github.com/jaywalnut310/vits/) and exported to the [onnxruntime](https://onnxruntime.ai/).

This is a project of the [Open Home Foundation](https://www.openhomefoundation.org/).

## Voices

Our goal is to support Home Assistant and the [Year of Voice](https://www.home-assistant.io/blog/2022/12/20/year-of-voice/).

[Download voices](VOICES.md) for the supported languages:

* Arabic (ar_JO)
* Catalan (ca_ES)
* Czech (cs_CZ)
* Welsh (cy_GB)
* Danish (da_DK)
* German (de_DE)
* Greek (el_GR)
* English (en_GB, en_US)
* Spanish (es_ES, es_MX)
* Finnish (fi_FI)
* French (fr_FR)
* Hungarian (hu_HU)
* Icelandic (is_IS)
* Italian (it_IT)
* Georgian (ka_GE)
* Kazakh (kk_KZ)
* Luxembourgish (lb_LU)
* Nepali (ne_NP)
* Dutch (nl_BE, nl_NL)
* Norwegian (no_NO)
* Polish (pl_PL)
* Portuguese (pt_BR, pt_PT)
* Romanian (ro_RO)
* Russian (ru_RU)
* Serbian (sr_RS)
* Swedish (sv_SE)
* Swahili (sw_CD)
* Turkish (tr_TR)
* Ukrainian (uk_UA)
* Vietnamese (vi_VN)
* Chinese (zh_CN)

You will need two files per voice:

1. A `.onnx` model file, such as [`en_US-lessac-medium.onnx`](https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx)
2. A `.onnx.json` config file, such as [`en_US-lessac-medium.onnx.json`](https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json)

The `MODEL_CARD` file for each voice contains important licensing information. Piper is intended for text to speech research, and does not impose any additional restrictions on voice models. Some voices may have restrictive licenses, however, so please review them carefully!


## Installation

You can [run Piper with Python](#running-in-python) or download a binary release:

* [amd64](https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_amd64.tar.gz) (64-bit desktop Linux)
* [arm64](https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_arm64.tar.gz) (64-bit Raspberry Pi 4)
* [armv7](https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_armv7.tar.gz) (32-bit Raspberry Pi 3/4)

If you want to build from source, see the [Makefile](Makefile) and [C++ source](src/cpp).
You must download and extract [piper-phonemize](https://github.com/rhasspy/piper-phonemize) to `lib/Linux-$(uname -m)/piper_phonemize` before building.
For example, `lib/Linux-x86_64/piper_phonemize/lib/libpiper_phonemize.so` should exist for AMD/Intel machines (as well as everything else from `libpiper_phonemize-amd64.tar.gz`).


## Usage

1. [Download a voice](#voices) and extract the `.onnx` and `.onnx.json` files
2. Run the `piper` binary with text on standard input, `--model /path/to/your-voice.onnx`, and `--output_file output.wav`

For example:

``` sh
echo 'Welcome to the world of speech synthesis!' | \
  ./piper --model en_US-lessac-medium.onnx --output_file welcome.wav
```

For multi-speaker models, use `--speaker <number>` to change speakers (default: 0).

See `piper --help` for more options.

### Streaming Audio

Piper can stream raw audio to stdout as its produced:

``` sh
echo 'This sentence is spoken first. This sentence is synthesized while the first sentence is spoken.' | \
  ./piper --model en_US-lessac-medium.onnx --output-raw | \
  aplay -r 22050 -f S16_LE -t raw -
```

This is **raw** audio and not a WAV file, so make sure your audio player is set to play 16-bit mono PCM samples at the correct sample rate for the voice.

### JSON Input

The `piper` executable can accept JSON input when using the `--json-input` flag. Each line of input must be a JSON object with `text` field. For example:

``` json
{ "text": "First sentence to speak." }
{ "text": "Second sentence to speak." }
```

Optional fields include:

* `speaker` - string
    * Name of the speaker to use from `speaker_id_map` in config (multi-speaker voices only)
* `speaker_id` - number
    * Id of speaker to use from 0 to number of speakers - 1 (multi-speaker voices only, overrides "speaker")
* `output_file` - string
    * Path to output WAV file
    
The following example writes two sentences with different speakers to different files:

``` json
{ "text": "First speaker.", "speaker_id": 0, "output_file": "/tmp/speaker_0.wav" }
{ "text": "Second speaker.", "speaker_id": 1, "output_file": "/tmp/speaker_1.wav" }
```


## People using Piper

Piper has been used in the following projects/papers:

* [Home Assistant](https://github.com/home-assistant/addons/blob/master/piper/README.md)
* [Rhasspy 3](https://github.com/rhasspy/rhasspy3/)
* [NVDA - NonVisual Desktop Access](https://www.nvaccess.org/post/in-process-8th-may-2023/#voices)
* [Image Captioning for the Visually Impaired and Blind: A Recipe for Low-Resource Languages](https://www.techrxiv.org/articles/preprint/Image_Captioning_for_the_Visually_Impaired_and_Blind_A_Recipe_for_Low-Resource_Languages/22133894)
* [Open Voice Operating System](https://github.com/OpenVoiceOS/ovos-tts-plugin-piper)
* [JetsonGPT](https://github.com/shahizat/jetsonGPT)
* [LocalAI](https://github.com/go-skynet/LocalAI)
* [Lernstick EDU / EXAM: reading clipboard content aloud with language detection](https://lernstick.ch/)
* [Natural Speech - A plugin for Runelite, an OSRS Client](https://github.com/phyce/rl-natural-speech)
* [mintPiper](https://github.com/evuraan/mintPiper)
* [Vim-Piper](https://github.com/wolandark/vim-piper)

## Training

See the [training guide](TRAINING.md) and the [source code](src/python).

Pretrained checkpoints are available on [Hugging Face](https://huggingface.co/datasets/rhasspy/piper-checkpoints/tree/main)


## Running in Python

See [src/python_run](src/python_run)

Install with `pip`:

``` sh
pip install piper-tts
```

and then run:

``` sh
echo 'Welcome to the world of speech synthesis!' | piper \
  --model en_US-lessac-medium \
  --output_file welcome.wav
```

This will automatically download [voice files](https://huggingface.co/rhasspy/piper-voices/tree/v1.0.0) the first time they're used. Use `--data-dir` and `--download-dir` to adjust where voices are found/downloaded.

If you'd like to use a GPU, install the `onnxruntime-gpu` package:


``` sh
.venv/bin/pip3 install onnxruntime-gpu
```

and then run `piper` with the `--cuda` argument. You will need to have a functioning CUDA environment, such as what's available in [NVIDIA's PyTorch containers](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch).

