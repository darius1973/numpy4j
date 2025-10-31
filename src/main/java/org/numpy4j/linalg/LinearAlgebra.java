package org.numpy4j.linalg;
import org.numpy4j.core.NDArray;

/**
 * A utility class providing core linear algebra operations on {@link NDArray} objects,
 * including vector and matrix computations such as dot products, matrix multiplication,
 * transposition, norms, determinants, inverses, and identity matrix creation.
 * <p>
 * This class is intended as a lightweight alternative to libraries such as NumPy or BLAS,
 * supporting small to medium-sized datasets directly in pure Java.
 * </p>
 *
 * <h2>Example Usage:</h2>
 * <pre>{@code
 * NDArray a = new NDArray(new double[]{1, 2, 3});
 * NDArray b = new NDArray(new double[]{4, 5, 6});
 * double dot = LinearAlgebra.dot(a, b); // 32.0
 *
 * NDArray A = new NDArray(new double[]{1, 2, 3, 4}, 2, 2);
 * NDArray B = new NDArray(new double[]{5, 6, 7, 8}, 2, 2);
 * NDArray C = LinearAlgebra.matmul(A, B);
 * // C = [[19, 22],
 * //      [43, 50]]
 *
 * NDArray T = LinearAlgebra.transpose(A);
 * // T = [[1, 3],
 * //      [2, 4]]
 * }</pre>
 */
public class LinearAlgebra {

    /**
     * Computes the <b>dot product</b> (inner product) of two 1D vectors.
     * <p>
     * This operation sums the elementwise products:
     * <pre>{@code
     * dot(a, b) = a₁·b₁ + a₂·b₂ + ... + aₙ·bₙ
     * }</pre>
     *
     * @param a the first 1D vector
     * @param b the second 1D vector
     * @return the dot product as a scalar value
     * @throws IllegalArgumentException if the vectors have different lengths
     *
     * <h3>Example:</h3>
     * <pre>{@code
     * NDArray a = new NDArray(new double[]{1, 2, 3});
     * NDArray b = new NDArray(new double[]{4, 5, 6});
     * double result = LinearAlgebra.dot(a, b); // 32.0
     * }</pre>
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
     * Performs <b>matrix multiplication</b> between two 2D matrices:
     * <pre>{@code
     * C = A × B
     * }</pre>
     * <p>
     * The number of columns in A must match the number of rows in B.
     * The result is an m×p matrix, where:
     * <ul>
     *   <li>m = A.rows</li>
     *   <li>p = B.cols</li>
     * </ul>
     *
     * @param A the left-hand matrix
     * @param B the right-hand matrix
     * @return a new {@link NDArray} representing the product matrix C
     * @throws IllegalArgumentException if either input is not 2D or shapes are incompatible
     *
     * <h3>Example:</h3>
     * <pre>{@code
     * NDArray A = new NDArray(new double[]{1, 2, 3, 4}, 2, 2);
     * NDArray B = new NDArray(new double[]{5, 6, 7, 8}, 2, 2);
     * NDArray C = LinearAlgebra.matmul(A, B);
     * // C = [[19, 22],
     * //      [43, 50]]
     * }</pre>
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
     * Returns the <b>transpose</b> of a 2D matrix.
     * <p>
     * The rows and columns are swapped:
     * <pre>{@code
     * Aᵀ[i, j] = A[j, i]
     * }</pre>
     *
     * @param A the input 2D matrix
     * @return a new {@link NDArray} representing the transposed matrix
     * @throws IllegalArgumentException if the input is not a 2D matrix
     *
     * <h3>Example:</h3>
     * <pre>{@code
     * NDArray A = new NDArray(new double[]{1, 2, 3, 4}, 2, 2);
     * NDArray T = LinearAlgebra.transpose(A);
     * // T = [[1, 3],
     * //      [2, 4]]
     * }</pre>
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
     * Computes the <b>Euclidean (L2) norm</b> of a vector:
     * <pre>{@code
     * ||A||₂ = sqrt(a₁² + a₂² + ... + aₙ²)
     * }</pre>
     *
     * @param A a 1D vector
     * @return the L2 norm (magnitude) of the vector
     *
     * <h3>Example:</h3>
     * <pre>{@code
     * NDArray v = new NDArray(new double[]{3, 4});
     * double n = LinearAlgebra.norm(v); // 5.0
     * }</pre>
     */
    public static double norm(NDArray A) {
        double[] data = A.getData();
        double sum = 0;
        for (double v : data)
            sum += v * v;
        return Math.sqrt(sum);
    }

    /**
     * Computes the <b>determinant</b> of a square matrix using recursive Laplace expansion.
     * <p>
     * This implementation is simple but not efficient for large matrices (O(n!)).
     *
     * @param A a square matrix (n×n)
     * @return the determinant of the matrix
     * @throws IllegalArgumentException if the input is not square
     *
     * <h3>Example:</h3>
     * <pre>{@code
     * NDArray A = new NDArray(new double[]{1, 2, 3, 4}, 2, 2);
     * double det = LinearAlgebra.det(A); // -2.0
     * }</pre>
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
     * Computes the <b>inverse</b> of a square matrix using Gaussian elimination.
     * <p>
     * This method augments the matrix with the identity and applies row operations
     * to transform [A | I] → [I | A⁻¹].
     *
     * @param A the input square matrix
     * @return the inverse matrix A⁻¹ as an {@link NDArray}
     * @throws IllegalArgumentException if the matrix is not square
     * @throws ArithmeticException if the matrix is singular (non-invertible)
     *
     * <h3>Example:</h3>
     * <pre>{@code
     * NDArray A = new NDArray(new double[]{4, 7, 2, 6}, 2, 2);
     * NDArray inv = LinearAlgebra.inv(A);
     * // inv ≈ [[0.6, -0.7],
     * //         [-0.2, 0.4]]
     * }</pre>
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
     * Creates an <b>identity matrix</b> of size n×n.
     * <p>
     * The identity matrix has ones on the main diagonal and zeros elsewhere:
     * <pre>{@code
     * eye(3) =
     * [[1, 0, 0],
     *  [0, 1, 0],
     *  [0, 0, 1]]
     * }</pre>
     *
     * @param n the number of rows and columns
     * @return an n×n identity matrix as an {@link NDArray}
     *
     * <h3>Example:</h3>
     * <pre>{@code
     * NDArray I = LinearAlgebra.eye(3);
     * // [[1, 0, 0],
     * //  [0, 1, 0],
     * //  [0, 0, 1]]
     * }</pre>
     */
    public static NDArray eye(int n) {
        double[] data = new double[n * n];
        for (int i = 0; i < n; i++) {
            data[i * n + i] = 1.0;  // Set diagonal to 1
        }
        return new NDArray(data, n, n);
    }
}
