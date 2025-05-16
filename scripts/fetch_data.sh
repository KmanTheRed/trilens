
set -e                      # exit on first error

mkdir -p data

wget -qO data/raid.zip https://github.com/liamdugan/raid/archive/refs/heads/main.zip

unzip -q data/raid.zip -d data && rm data/raid.zip

pip install -q kaggle

kaggle datasets download -d shanegerami/ai-vs-human-text -p data

unzip -q data/ai-vs-human-text.zip -d data && rm data/ai-vs-human-text.zip

