@echo off

rem =前后不能有空格

set snr_array=0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0 6.5 7.0 7.5 8.0 8.5 9.0 9.5 10.0 10.5 11.0 11.5 12.0
set "a=./dataset/length/conv/dataset-length-conv-"
set "b=db.csv"

for %%c in (%snr_array%) do (
    python main.py -a eca_resnet34 --ksize 3557 --epochs 100 %a%%%c%b%
)

set snr_array=0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0 6.5 7.0 7.5 8.0 8.5 9.0 9.5 10.0 10.5 11.0 11.5 12.0
set "a=./dataset/length/ldpc/dataset-length-ldpc-"
set "b=db.csv"

for %%c in (%snr_array%) do (
    python main.py -a eca_resnet34 --ksize 3557 --epochs 100 %a%%%c%b%
)

set snr_array=0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0 6.5 7.0 7.5 8.0 8.5 9.0 9.5 10.0 10.5 11.0 11.5 12.0
set "a=./dataset/rate/conv/dataset-rate-conv-"
set "b=db.csv"

for %%c in (%snr_array%) do (
    python main.py -a eca_resnet34 --ksize 3557 --epochs 100 %a%%%c%b%
)

set snr_array=0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0 6.5 7.0 7.5 8.0 8.5 9.0 9.5 10.0 10.5 11.0 11.5 12.0
set "a=./dataset/rate/ldpc/dataset-rate-ldpc-"
set "b=db.csv"

for %%c in (%snr_array%) do (
    python main.py -a eca_resnet34 --ksize 3557 --epochs 100 %a%%%c%b%
)