.DEFAULT: help
help:
	@echo "make install"
	@echo "       prepare environment and download dependencies, use only once"
	@echo "make run"
	@echo "       run the program"

DIR=$(shell pwd)

install: update install-gym install-pygame

update:
	sudo apt-get update

install-pygame: install-pip
	python3 -m pip install -U pygame --user


install-gym: install-pip
	pip install gym


install-pip:
	sudo apt-get install python-pip
        
run:
	python pong.py

