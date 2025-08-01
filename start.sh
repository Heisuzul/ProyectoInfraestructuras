#!/bin/bash

# Inicia Ray como nodo principal en segundo plano
ray start --head 

# Ejecuta la app
python main.py