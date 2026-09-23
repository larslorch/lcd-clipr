#!/bin/bash

python lcd_clipr/manager.py linear-er/linear-100-p=30 "$@"
python lcd_clipr/manager.py linear-er/linear-100-p=50 "$@"
python lcd_clipr/manager.py linear-er/linear-100-p=70 "$@"
python lcd_clipr/manager.py linear-er/linear-100-p=100 "$@"

python lcd_clipr/manager.py linear-er/linear-300-p=90 "$@"
python lcd_clipr/manager.py linear-er/linear-300-p=150 "$@"
python lcd_clipr/manager.py linear-er/linear-300-p=200 "$@"
python lcd_clipr/manager.py linear-er/linear-300-p=300 "$@"

python lcd_clipr/manager.py linear-er/linear-1000-p=300 "$@"
python lcd_clipr/manager.py linear-er/linear-1000-p=500 "$@"
python lcd_clipr/manager.py linear-er/linear-1000-p=700 "$@"
python lcd_clipr/manager.py linear-er/linear-1000-p=1000 "$@"

