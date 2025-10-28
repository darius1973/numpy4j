package org.numpy4j.linalg;
import org.numpy4j.core.NDArray;
/*
  ✅ Methods
    Method	Description
    dot(a, b)	Matrix/vector dot product (already present — we'll enhance it)
    matmul(a, b)	General matrix multiplication
    transpose(a)	Returns the transposed NDArray
    norm(a)	Euclidean norm (‖a‖₂)
    inv(a)	Matrix inverse (using Gaussian elimination for now)
    det(a)	Determinant (recursive or LU-based)
*/

public class LinearAlgebra {

    /**
     * Dot product for 1D vectors.
     */
    public static double dot(NDArray a, NDArray b) {
        double[] ad = a.getData();
        double[] bd = b.getData();
        if (ad.length != bd.length)
            throw new IllegalArgumentException("Arrays must be the same length for dot product");
        double sum = 0;
        for (int i = 0; i < ad.length; i++)
            sum += ad[i] * bd[i];
        return sum;
    }

    /**
     * Matrix multiplication: C = A × B
     */
    public static NDArray matmul(NDArray A, NDArray B) {
        int[] shapeA = A.getShape();
        int[] shapeB = B.getShape();
        if (shapeA.length != 2 || shapeB.length != 2)
            throw new IllegalArgumentException("Both inputs must be 2D matrices");
        if (shapeA[1] != shapeB[0])
            throw new IllegalArgumentException("Incompatible shapes for matrix multiplication");

        int m = shapeA[0], n = shapeA[1], p = shapeB[1];
        NDArray C = new NDArray(m, p);

        double[] aData = A.getData();
        double[] bData = B.getData();
        double[] cData = new double[m * p];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < p; j++) {
                double sum = 0;
                for (int k = 0; k < n; k++) {
                    sum += aData[i * n + k] * bData[k * p + j];
                }
                cData[i * p + j] = sum;
            }
        }

        return new NDArray(cData, m, p);
    }

    /**
     * Transpose of a 2D matrix.
     */
    public static NDArray transpose(NDArray A) {
        int[] shape = A.getShape();
        if (shape.length != 2)
            throw new IllegalArgumentException("Transpose only supports 2D matrices");
        int rows = shape[0], cols = shape[1];
        double[] transposed = new double[rows * cols];
        double[] data = A.getData();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposed[j * rows + i] = data[i * cols + j];
            }
        }
        return new NDArray(transposed, cols, rows);
    }

    /**
     * Euclidean (L2) norm of a vector.
     */
    public static double norm(NDArray A) {
        double[] data = A.getData();
        double sum = 0;
        for (double v : data)
            sum += v * v;
        return Math.sqrt(sum);
    }

    /**
     * Determinant of a square matrix (recursive Laplace expansion).
     * Not efficient for large matrices.
     */
    public static double det(NDArray A) {
        int[] shape = A.getShape();
        if (shape.length != 2 || shape[0] != shape[1])
            throw new IllegalArgumentException("Matrix must be square");
        int n = shape[0];
        double[] data = A.getData();
        if (n == 1)
            return data[0];
        if (n == 2)
            return data[0] * data[3] - data[1] * data[2];

        double det = 0;
        for (int col = 0; col < n; col++) {
            det += Math.pow(-1, col) * data[col] * det(minor(A, 0, col));
        }
        return det;
    }

    private static NDArray minor(NDArray A, int row, int col) {
        int n = A.getShape()[0];
        double[] data = A.getData();
        double[] m = new double[(n - 1) * (n - 1)];
        int idx = 0;
        for (int i = 0; i < n; i++) {
            if (i == row) continue;
            for (int j = 0; j < n; j++) {
                if (j == col) continue;
                m[idx++] = data[i * n + j];
            }
        }
        return new NDArray(m, n - 1, n - 1);
    }

    /**
     * Matrix inverse using Gaussian elimination.
     */
    public static NDArray inv(NDArray A) {
        int n = A.getShape()[0];
        int[] shape = A.getShape();
        if (shape.length != 2 || shape[0] != shape[1])
            throw new IllegalArgumentException("Matrix must be square");

        double[][] aug = new double[n][2 * n];
        double[] data = A.getData();

        // Create augmented matrix [A | I]
        for (int i = 0; i < n; i++) {
            System.arraycopy(data, i * n + 0, aug[i], 0, n);
            aug[i][n + i] = 1;
        }

        // Gaussian elimination
        for (int i = 0; i < n; i++) {
            double diag = aug[i][i];
            if (Math.abs(diag) < 1e-12)
                throw new ArithmeticException("Matrix is singular and cannot be inverted");
            for (int j = 0; j < 2 * n; j++)
                aug[i][j] /= diag;
            for (int k = 0; k < n; k++) {
                if (k == i) continue;
                double factor = aug[k][i];
                for (int j = 0; j < 2 * n; j++)
                    aug[k][j] -= factor * aug[i][j];
            }
        }

        // Extract inverse
        double[] invData = new double[n * n];
        for (int i = 0; i < n; i++) {
            System.arraycopy(aug[i], n + 0, invData, i * n + 0, n);
        }

        return new NDArray(invData, n, n);
    }

    /**
     * Creates an identity matrix of size n x n.
     * Equivalent to numpy.eye(n)
     *
     * @param n the size of the matrix (rows = cols = n)
     * @return NDArray representing the identity matrix
     */
    public static NDArray eye(int n) {
        double[] data = new double[n * n];
        for (int i = 0; i < n; i++) {
            data[i * n + i] = 1.0;  // Set diagonal to 1
        }
        return new NDArray(data, n, n);
    }
}
