PY=python

setup:
pip install -r requirements.txt

train:
$(PY) -m src.train

eval:
$(PY) -m src.evaluate

score:
$(PY) -m src.score

all: train eval
