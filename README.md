# 🧮 NumPy4J — NumPy for Java

NumPy4J is a Java library that brings the functionality and expressiveness of Python’s NumPy to the JVM.
It allows Java developers to perform scientific computing, numerical analysis,
and matrix operations efficiently, using native Java arrays, multithreading,
and optional integration with high-performance native libraries like OpenBLAS and Intel MKL via JNI.
NumPy4J combines Pythonic convenience with Java-grade performance.

---

## Features

- **NDArray:** n-dimensional arrays with linear indexing, broadcasting, slicing, arithmetic operations, and reshaping.
- **Linear Algebra:** dot product, matrix multiplication, identity matrices, and more advanced operations.
- **Random:** seeded random number generation compatible with reproducible experiments.
- **Performance:** multithreaded arithmetic, direct memory access using DoubleBuffer or Unsafe, and optional native acceleration.
- **Testing:** JSON-based Python-compatible test generator for validating results.

---

## 🧠 Future Roadmap
    ✅ NDArray core operations (completed)
    🔄 Broadcasting and slicing
    🧮 FFT module
    🧰 Statistical utilities
    ⚙️ Integration with OpenBLAS / Intel MKL via JNI
    📊 JSON-based Python compatibility tests generator
    🧬 GPU-accelerated backend (OpenCL / CUDA)

## Maven Project Structure

numpy4j/
├── core/ NDArray and basic operations
├── linalg/ Linear algebra utilities
├── random/ Random number generation
├── fft/ Fourier transforms (future)
├── api/ Public API entry points
├── tests/ Unit tests
└── pom.xml Maven descriptor


---

## Quick Start

### NDArray Example

```java
import org.numpy4j.api.Numpy;
import org.numpy4j.core.NDArray;

public class Example {
    public static void main(String[] args) {
        NDArray a = Numpy.arange(0, 6).reshape(2, 3);
        NDArray b = Numpy.zeros(2, 3);
        NDArray c = a.add(b);
        System.out.println(Arrays.toString(c.getData())); // [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    }
}

```

## Utilities

```java
NDArray arr = Numpy.arange(0, 5, 2);
System.out.println(Arrays.toString(arr.getData())); // [0.0, 2.0, 4.0]

NDArray zeros = Numpy.zeros(3, 3);
NDArray empty = Numpy.empty(2, 2);
```

## Linear Algebra

```java
import org.numpy4j.linalg.LinearAlgebra;

NDArray x = Numpy.arange(0, 3);
NDArray I = LinearAlgebra.eye(3);
double result = LinearAlgebra.dot(x, x); // 5.0

NDArray product = LinearAlgebra.matmul(Numpy.arange(0, 6).reshape(2, 3), I);
```

## Random Numbers

```java
Numpy.seed(42);
NDArray rand = Numpy.random(2, 2);
System.out.println(rand);
```
## Testing

```java
@Test
public void testArange() {
    NDArray arr = Numpy.arange(0, 5, 2);
    double[] expected = {0.0, 2.0, 4.0};
    assertArrayEquals(expected, arr.getData(), 1e-10);
}

@Test
public void testReshape() {
    NDArray arr = Numpy.arange(0, 6);
    NDArray reshaped = arr.reshape(2, 3);
    double[] expected = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    assertArrayEquals(expected, reshaped.getData(), 1e-10);
}
```

## python test generator
```python
import numpy as np, json

tests = {
    "arange_0_5_2": np.arange(0, 5, 2).tolist(),
    "zeros_3x3": np.zeros((3, 3)).tolist(),
}

with open("numpy_tests.json", "w") as f:
    json.dump(tests, f, indent=2)
```

## Performance

Multithreaded arithmetic for large arrays.

Direct memory access using DoubleBuffer or Unsafe.

Optional integration with OpenBLAS or Intel MKL via JNI.

Benchmarks show that for large-scale operations, NumPy4J approaches NumPy’s performance
while remaining pure Java-compatible.

## Build & Run

Requirements:
Java 17+
Maven 3.8+

Commands:
mvn clean package
mvn test

##License
Apache License 2.0 — free for commercial or open-source use.

