#!/usr/bin/env bash


conda env export > excut_env.yml

conda list --explicit > conda_excut_list.txt

pip freeze > pip_excut_requirements.txt