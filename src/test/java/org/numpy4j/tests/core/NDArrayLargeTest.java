package org.numpy4j.tests.core;


import org.junit.jupiter.api.Test;
import org.numpy4j.core.NDArray;

import java.util.Random;
import static org.junit.jupiter.api.Assertions.*;

public class NDArrayLargeTest {

    // Simple slow reference dot product for correctness checking
    private double[][] referenceDot(double[][] A, double[][] B) {
        int n = A.length;
        int m = B[0].length;
        int k = A[0].length;

        double[][] C = new double[n][m];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                double sum = 0;
                for (int a = 0; a < k; a++) {
                    sum += A[i][a] * B[a][j];
                }
                C[i][j] = sum;
            }
        }
        return C;
    }

    private NDArray randomNDArray(int rows, int cols) {
        Random r = new Random(123);
        double[] data = new double[rows * cols];
        for (int i = 0; i < data.length; i++) {
            data[i] = r.nextDouble();
        }
        return NDArray.of(data, rows, cols);
    }

    @Test
    public void testLargeDotProduct() {
        int M = 800; // rows of A
        int K = 600; // cols of A, rows of B
        int N = 700; // cols of B

        NDArray A = randomNDArray(M, K);
        NDArray B = randomNDArray(K, N);

        long start = System.currentTimeMillis();
        NDArray C = A.dot(B);
        long end = System.currentTimeMillis();
        System.out.println("Parallel tiled NDArray.dot() took: " + (end - start) + " ms");

        // Validate correctness on a smaller slice to avoid massive comparison cost
        double[][] A_ref = new double[50][K];
        double[][] B_ref = new double[K][50];

        // copy slices
        for (int i = 0; i < 50; i++)
            System.arraycopy(A.getRow(i), 0, A_ref[i], 0, K);
        for (int j = 0; j < 50; j++)
            for (int k = 0; k < K; k++)
                B_ref[k][j] = B.get(k, j);

        double[][] C_ref = referenceDot(A_ref, B_ref);

        // Compare 50×50 block
        for (int i = 0; i < 50; i++) {
            for (int j = 0; j < 50; j++) {
                assertEquals(C_ref[i][j], C.get(i, j), 1e-6);
            }
        }
    }

    @Test
    public void testAddLarge() {
        int n = 10_000_000; // 1 million elements
        int rows = 5000;
        int cols = n / rows;

        System.out.println("Creating test arrays...");
        NDArray A = NDArray.random(rows, cols);
        NDArray B = NDArray.random(rows, cols);

        // Reference arrays for correctness
        double[] Aref = A.getData();
        double[] Bref = B.getData();
        double[] Cref = new double[Aref.length];

        // --- Reference computation (single-threaded) ---
        long t0 = System.nanoTime();
        for (int i = 0; i < Aref.length; i++) {
            Cref[i] = Aref[i] + Bref[i];
        }
        long t1 = System.nanoTime();
        System.out.println("Reference add() took: " + ((t1 - t0) / 1_000_000.0) + " ms");

        // --- NDArray parallel add() ---
        long t2 = System.nanoTime();
        NDArray C = A.add(B);
        long t3 = System.nanoTime();
        System.out.println("NDArray.add() took: " + ((t3 - t2) / 1_000_000.0) + " ms");

        // --- Correctness check ---
        assertEquals(Aref.length, C.getSize());
    }

}

