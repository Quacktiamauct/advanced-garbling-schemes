clang aesenc.c -o aesenc -march=alderlake
clear

./aesenc 1
echo ""
./aesenc 2
echo ""
./aesenc 4
echo ""
./aesenc 8
echo ""
./aesenc F
