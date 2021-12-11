@echo off

clang aesenc.c -o aesenc.exe -march=znver2
cls

aesenc.exe 1
echo ""
aesenc.exe 2
echo ""
aesenc.exe 4
echo ""
aesenc.exe 8
echo ""
aesenc.exe F
