install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black library/python/*.py

lint:
	pylint --disable=R,C library/python/*.py

all: install lint