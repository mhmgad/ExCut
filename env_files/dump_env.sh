#!/usr/bin/env bash


conda env export > ExDEC_env.yml

conda list --explicit > conda_ExDEC_list.txt

pip freeze > pip_ExDEC_requirements.txt