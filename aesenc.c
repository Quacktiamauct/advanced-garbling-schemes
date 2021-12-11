#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <immintrin.h>

#define printfn() printf("\n")

void aes_1wide();
void aes_2wide();
void aes_4wide();
void aes_8wide();
void aes_16wide();

#define ITERATIONS 10000000

int
main(int argc, char **argv) {
	if (argc != 2) {
		printf("Please provide 1 argument, either: 1, 4, 8, F\n");
		return 1;
	}

	switch (*argv[1]) {
		case '1':
			aes_1wide();
			break;
		case '2':
			aes_2wide();
			break;
		case '4':
			aes_4wide();
			break;
		case '8':
			aes_8wide();
			break;
		case 'F':
			aes_16wide();
			break;
	}

	return 0;
}

void
aes_1wide() {
	__m128i RK1[11];
	for (uint32_t i = 0; i < 11; ++i)
	{
		RK1[i] = _mm_set1_epi8(i * 2);
	}

	__m128i X = _mm_set1_epi8(42);

	printf("Benchmarking AES encrypting interleaved 1 block(s)\n");
	uint64_t start = __rdtsc();

	for (uint32_t a = 0; a < ITERATIONS; ++a)
	{
		X = _mm_xor_si128(X, RK1[0]);

		for (uint32_t i = 1; i <= 9; ++i) {
			X = _mm_aesenc_si128(X, RK1[i]);
		}

		X = _mm_aesenclast_si128(X, RK1[10]);
	}

	uint64_t elapsed = __rdtsc() - start;
	double average = (double) elapsed / (double) ITERATIONS;
	double averagePerBlock = average / 1;

	printf("X %016llx\n", _mm_extract_epi64(X, 0));
	printfn();

	printf("Total cycles:             %lu\n", elapsed);
	printf("Cycles / iteration:       %.2f\n", average);
	printf("Average cycles per block  %.2f\n", averagePerBlock);
}

void
aes_2wide() {
	__m128i RK1[11];
	__m128i RK2[11];
	for (uint32_t i = 0; i < 11; ++i)
	{
		RK1[i] = _mm_set1_epi8(i * 2);
		RK2[i] = _mm_set1_epi8(i * 5);
	}

	__m128i X = _mm_set1_epi8(42);
	__m128i Y = _mm_set1_epi8(14);

	printf("Benchmarking AES encrypting interleaved 2 block(s)\n");
	uint64_t start = __rdtsc();

	for (uint32_t a = 0; a < ITERATIONS; ++a)
	{
		X = _mm_xor_si128(X, RK1[0]);
		Y = _mm_xor_si128(Y, RK2[0]);

		for (uint32_t i = 1; i <= 9; ++i) {
			X = _mm_aesenc_si128(X, RK1[i]);
			Y = _mm_aesenc_si128(Y, RK2[i]);
		}

		X = _mm_aesenclast_si128(X, RK1[10]);
		Y = _mm_aesenclast_si128(Y, RK2[10]);
	}

	uint64_t elapsed = __rdtsc() - start;
	double average = (double) elapsed / (double) ITERATIONS;
	double averagePerBlock = average / 2;

	printf("X %016llx, Y %016llx\n", _mm_extract_epi64(X, 0), _mm_extract_epi64(Y, 0));
	printfn();

	printf("Total cycles:             %lu\n", elapsed);
	printf("Cycles / iteration:       %.2f\n", average);
	printf("Average cycles per block  %.2f\n", averagePerBlock);
}

void
aes_4wide() {
	__m128i RK1[11];
	__m128i RK2[11];
	__m128i RK3[11];
	__m128i RK4[11];
	for (uint32_t i = 0; i < 11; ++i)
	{
		RK1[i] = _mm_set1_epi8(i * 2);
		RK2[i] = _mm_set1_epi8(i * 5);
		RK3[i] = _mm_set1_epi8(i * 7);
		RK4[i] = _mm_set1_epi8(i * 11);
	}

	__m128i X = _mm_set1_epi8(42);
	__m128i Y = _mm_set1_epi8(14);
	__m128i Z = _mm_set1_epi8(13);
	__m128i W = _mm_set1_epi8(9);

	printf("Benchmarking AES encrypting interleaved 4 block(s)\n");
	uint64_t start = __rdtsc();

	for (uint32_t a = 0; a < ITERATIONS; ++a)
	{
		X = _mm_xor_si128(X, RK1[0]);
		Y = _mm_xor_si128(Y, RK2[0]);
		Z = _mm_xor_si128(Z, RK3[0]);
		W = _mm_xor_si128(W, RK4[0]);

		for (uint32_t i = 1; i <= 9; ++i) {
			X = _mm_aesenc_si128(X, RK1[i]);
			Y = _mm_aesenc_si128(Y, RK2[i]);
			Z = _mm_aesenc_si128(Z, RK3[i]);
			W = _mm_aesenc_si128(W, RK4[i]);
		}

		X = _mm_aesenclast_si128(X, RK1[10]);
		Y = _mm_aesenclast_si128(Y, RK2[10]);
		Z = _mm_aesenclast_si128(Z, RK3[10]);
		W = _mm_aesenclast_si128(W, RK4[10]);
	}

	uint64_t elapsed = __rdtsc() - start;
	double average = (double) elapsed / (double) ITERATIONS;
	double averagePerBlock = average / 4;

	printf("X %016llx, Y %016llx\n", _mm_extract_epi64(X, 0), _mm_extract_epi64(Y, 0));
	printf("Z %016llx, W %016llx\n", _mm_extract_epi64(Z, 0), _mm_extract_epi64(W, 0));
	printfn();

	printf("Total cycles:             %lu\n", elapsed);
	printf("Cycles / iteration:       %.2f\n", average);
	printf("Average cycles per block  %.2f\n", averagePerBlock);
}

void
aes_8wide() {
	__m128i RK1[11];
	__m128i RK2[11];
	__m128i RK3[11];
	__m128i RK4[11];
	__m128i RK5[11];
	__m128i RK6[11];
	__m128i RK7[11];
	__m128i RK8[11];
	for (uint32_t i = 0; i < 11; ++i)
	{
		RK1[i] = _mm_set1_epi8(i * 2);
		RK2[i] = _mm_set1_epi8(i * 5);
		RK3[i] = _mm_set1_epi8(i * 7);
		RK4[i] = _mm_set1_epi8(i * 11);
		RK5[i] = _mm_set1_epi8(i * 13);
		RK6[i] = _mm_set1_epi8(i * 17);
		RK7[i] = _mm_set1_epi8(i * 19);
		RK8[i] = _mm_set1_epi8(i * 23);
	}

	__m128i X = _mm_set1_epi8(42);
	__m128i Y = _mm_set1_epi8(14);
	__m128i Z = _mm_set1_epi8(13);
	__m128i W = _mm_set1_epi8(9);
	__m128i A = _mm_set1_epi8(42 * 3);
	__m128i B = _mm_set1_epi8(14 * 6);
	__m128i C = _mm_set1_epi8(13 * 5);
	__m128i D = _mm_set1_epi8(9 * 9);

	printf("Benchmarking AES encrypting interleaved 8 block(s)\n");
	uint64_t start = __rdtsc();

	for (uint32_t a = 0; a < ITERATIONS; ++a)
	{
		X = _mm_xor_si128(X, RK1[0]);
		Y = _mm_xor_si128(Y, RK2[0]);
		Z = _mm_xor_si128(Z, RK3[0]);
		W = _mm_xor_si128(W, RK4[0]);
		A = _mm_xor_si128(A, RK5[0]);
		B = _mm_xor_si128(B, RK6[0]);
		C = _mm_xor_si128(C, RK7[0]);
		D = _mm_xor_si128(D, RK8[0]);

		for (uint32_t i = 1; i <= 9; ++i) {
			X = _mm_aesenc_si128(X, RK1[i]);
			Y = _mm_aesenc_si128(Y, RK2[i]);
			Z = _mm_aesenc_si128(Z, RK3[i]);
			W = _mm_aesenc_si128(W, RK4[i]);
			A = _mm_aesenc_si128(A, RK5[i]);
			B = _mm_aesenc_si128(B, RK6[i]);
			C = _mm_aesenc_si128(C, RK7[i]);
			D = _mm_aesenc_si128(D, RK8[i]);
		}

		X = _mm_aesenclast_si128(X, RK1[10]);
		Y = _mm_aesenclast_si128(Y, RK2[10]);
		Z = _mm_aesenclast_si128(Z, RK3[10]);
		W = _mm_aesenclast_si128(W, RK4[10]);
		A = _mm_aesenclast_si128(A, RK5[10]);
		B = _mm_aesenclast_si128(B, RK6[10]);
		C = _mm_aesenclast_si128(C, RK7[10]);
		D = _mm_aesenclast_si128(D, RK8[10]);
	}

	uint64_t elapsed = __rdtsc() - start;
	double average = (double) elapsed / (double) ITERATIONS;
	double averagePerBlock = average / 8;

	printf("X %016llx, Y %016llx\n", _mm_extract_epi64(X, 0), _mm_extract_epi64(Y, 0));
	printf("Z %016llx, W %016llx\n", _mm_extract_epi64(Z, 0), _mm_extract_epi64(W, 0));
	printf("A %016llx, B %016llx\n", _mm_extract_epi64(A, 0), _mm_extract_epi64(B, 0));
	printf("C %016llx, D %016llx\n", _mm_extract_epi64(C, 0), _mm_extract_epi64(D, 0));
	printfn();

	printf("Total cycles:             %lu\n", elapsed);
	printf("Cycles / iteration:       %.2f\n", average);
	printf("Average cycles per block  %.2f\n", averagePerBlock);
}

void
aes_16wide() {
	__m128i RK1[11];
	__m128i RK2[11];
	__m128i RK3[11];
	__m128i RK4[11];
	__m128i RK5[11];
	__m128i RK6[11];
	__m128i RK7[11];
	__m128i RK8[11];
	__m128i RK9[11];
	__m128i RK10[11];
	__m128i RK11[11];
	__m128i RK12[11];
	__m128i RK13[11];
	__m128i RK14[11];
	__m128i RK15[11];
	__m128i RK16[11];
	for (uint32_t i = 0; i < 11; ++i)
	{
		RK1[i] = _mm_set1_epi8(i * 2);
		RK2[i] = _mm_set1_epi8(i * 5);
		RK3[i] = _mm_set1_epi8(i * 7);
		RK4[i] = _mm_set1_epi8(i * 11);
		RK5[i] = _mm_set1_epi8(i * 13);
		RK6[i] = _mm_set1_epi8(i * 17);
		RK7[i] = _mm_set1_epi8(i * 19);
		RK8[i] = _mm_set1_epi8(i * 23);
		RK9[i] = _mm_set1_epi8(i * 2 * 9);
		RK10[i] = _mm_set1_epi8(i * 5 * 9);
		RK11[i] = _mm_set1_epi8(i * 7 * 9);
		RK12[i] = _mm_set1_epi8(i * 11 * 9);
		RK13[i] = _mm_set1_epi8(i * 13 * 9);
		RK14[i] = _mm_set1_epi8(i * 17 * 9);
		RK15[i] = _mm_set1_epi8(i * 19 * 9);
		RK16[i] = _mm_set1_epi8(i * 23 * 9);
	}

	__m128i X = _mm_set1_epi8(42);
	__m128i Y = _mm_set1_epi8(14);
	__m128i Z = _mm_set1_epi8(13);
	__m128i W = _mm_set1_epi8(9);
	__m128i A = _mm_set1_epi8(42 * 3);
	__m128i B = _mm_set1_epi8(14 * 6);
	__m128i C = _mm_set1_epi8(13 * 5);
	__m128i D = _mm_set1_epi8(9 * 9);
	__m128i E = _mm_set1_epi8(42 * 3);
	__m128i F = _mm_set1_epi8(14 * 3);
	__m128i G = _mm_set1_epi8(13 * 3);
	__m128i H = _mm_set1_epi8(9 * 9);
	__m128i I = _mm_set1_epi8(42 * 3);
	__m128i J = _mm_set1_epi8(14 * 6);
	__m128i K = _mm_set1_epi8(13 * 5);
	__m128i L = _mm_set1_epi8(9 * 9	);

	printf("Benchmarking AES encrypting interleaved 16 block(s)\n");
	uint64_t start = __rdtsc();

	for (uint32_t a = 0; a < ITERATIONS; ++a)
	{
		X = _mm_xor_si128(X, RK1[0]);
		Y = _mm_xor_si128(Y, RK2[0]);
		Z = _mm_xor_si128(Z, RK3[0]);
		W = _mm_xor_si128(W, RK4[0]);
		A = _mm_xor_si128(A, RK5[0]);
		B = _mm_xor_si128(B, RK6[0]);
		C = _mm_xor_si128(C, RK7[0]);
		D = _mm_xor_si128(D, RK8[0]);
		E = _mm_xor_si128(E, RK9[0]);
		F = _mm_xor_si128(F, RK10[0]);
		G = _mm_xor_si128(G, RK11[0]);
		H = _mm_xor_si128(H, RK12[0]);
		I = _mm_xor_si128(I, RK13[0]);
		J = _mm_xor_si128(J, RK14[0]);
		K = _mm_xor_si128(K, RK15[0]);
		L = _mm_xor_si128(L, RK16[0]);

		for (uint32_t i = 1; i <= 9; ++i) {
			X = _mm_aesenc_si128(X, RK1[i]);
			Y = _mm_aesenc_si128(Y, RK2[i]);
			Z = _mm_aesenc_si128(Z, RK3[i]);
			W = _mm_aesenc_si128(W, RK4[i]);
			A = _mm_aesenc_si128(A, RK5[i]);
			B = _mm_aesenc_si128(B, RK6[i]);
			C = _mm_aesenc_si128(C, RK7[i]);
			D = _mm_aesenc_si128(D, RK8[i]);
			E = _mm_aesenc_si128(E, RK9[i]);
			F = _mm_aesenc_si128(F, RK10[i]);
			G = _mm_aesenc_si128(G, RK11[i]);
			H = _mm_aesenc_si128(H, RK12[i]);
			I = _mm_aesenc_si128(I, RK13[i]);
			J = _mm_aesenc_si128(J, RK14[i]);
			K = _mm_aesenc_si128(K, RK15[i]);
			L = _mm_aesenc_si128(L, RK16[i]);
		}

		X = _mm_aesenclast_si128(X, RK1[10]);
		Y = _mm_aesenclast_si128(Y, RK2[10]);
		Z = _mm_aesenclast_si128(Z, RK3[10]);
		W = _mm_aesenclast_si128(W, RK4[10]);
		A = _mm_aesenclast_si128(A, RK5[10]);
		B = _mm_aesenclast_si128(B, RK6[10]);
		C = _mm_aesenclast_si128(C, RK7[10]);
		D = _mm_aesenclast_si128(D, RK8[10]);
		E = _mm_aesenclast_si128(E, RK9[10]);
		G = _mm_aesenclast_si128(F, RK10[10]);
		H = _mm_aesenclast_si128(G, RK11[10]);
		W = _mm_aesenclast_si128(H, RK12[10]);
		I = _mm_aesenclast_si128(I, RK13[10]);
		J = _mm_aesenclast_si128(J, RK14[10]);
		K = _mm_aesenclast_si128(K, RK15[10]);
		L = _mm_aesenclast_si128(L, RK16[10]);
	}

	uint64_t elapsed = __rdtsc() - start;
	double average = (double) elapsed / (double) ITERATIONS;
	double averagePerBlock = average / 16;

	printf("X %016llx, Y %016llx\n", _mm_extract_epi64(X, 0), _mm_extract_epi64(Y, 0));
	printf("Z %016llx, W %016llx\n", _mm_extract_epi64(Z, 0), _mm_extract_epi64(W, 0));
	printf("A %016llx, B %016llx\n", _mm_extract_epi64(A, 0), _mm_extract_epi64(B, 0));
	printf("C %016llx, D %016llx\n", _mm_extract_epi64(C, 0), _mm_extract_epi64(D, 0));
	printf("E %016llx, F %016llx\n", _mm_extract_epi64(E, 0), _mm_extract_epi64(F, 0));
	printf("G %016llx, H %016llx\n", _mm_extract_epi64(G, 0), _mm_extract_epi64(H, 0));
	printf("I %016llx, J %016llx\n", _mm_extract_epi64(I, 0), _mm_extract_epi64(J, 0));
	printf("K %016llx, L %016llx\n", _mm_extract_epi64(K, 0), _mm_extract_epi64(L, 0));
	printfn();

	printf("Total cycles:             %lu\n", elapsed);
	printf("Cycles / iteration:       %.2f\n", average);
	printf("Average cycles per block  %.2f\n", averagePerBlock);
}

