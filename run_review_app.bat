@echo off
title SWSTP Active Learning Review App
cd /d "%~dp0swstp_edge\active_learning"
echo Starting Active Learning Review App on "images for retrainning"...
python review_app.py --images-dir "images for retrainning"
pause

