#!/usr/bin/env python3
"""
generate_tests.py
-----------------
Generates JSON test cases from NumPy for validating NumPy4J operations in Java.
Includes arithmetic, dot product, reshape, broadcasting, slicing, transpose, and power tests.
"""

import json
import numpy as np
import os

# Folder to save testcases
os.makedirs("../resources/testcases", exist_ok=True)

def to_flat_json(array):
    """Convert numpy array to flat list and shape."""
    return {
        "data": array.flatten().tolist(),
        "shape": list(array.shape)
    }

# ----------------------------
# 1. Basic operations
# ----------------------------
def generate_basic_ops():
    tests = []
    for _ in range(5):
        A = np.random.randn(2, 2)
        B = np.random.randn(2, 2)
        result = A + B
        tests.append({
            "A": to_flat_json(A),
            "B": to_flat_json(B),
            "result": to_flat_json(result)
        })
    return tests

# ----------------------------
# 2. Dot product
# ----------------------------
def generate_dot_tests():
    tests = []
    for _ in range(5):
        A = np.random.randn(2, 3)
        B = np.random.randn(3, 2)
        result = A @ B
        tests.append({
            "A": to_flat_json(A),
            "B": to_flat_json(B),
            "result": to_flat_json(result)
        })
    return tests

# ----------------------------
# 3. Reshape
# ----------------------------
def generate_reshape_tests():
    tests = []
    for _ in range(5):
        A = np.random.randn(2, 3)
        new_shape = (3, 2)
        result = A.reshape(new_shape)
        tests.append({
            "A": to_flat_json(A),
            "new_shape": list(new_shape),
            "result": to_flat_json(result)
        })
    return tests

# ----------------------------
# 4. Broadcast
# ----------------------------
def generate_broadcast_tests():
    tests = []
    for _ in range(5):
        A = np.random.randn(2, 1)
        B = np.random.randn(1, 3)
        result = A + B  # broadcasting sum
        tests.append({
            "A": to_flat_json(A),
            "B": to_flat_json(B),
            "result": to_flat_json(result)
        })
    return tests

# ----------------------------
# 5. Aggregate (sum, mean)
# ----------------------------
def generate_aggregate_tests():
    tests = []
    for _ in range(5):
        A = np.random.randn(2, 2)
        tests.append({
            "A": to_flat_json(A),
            "sum": float(np.sum(A)),
            "mean": float(np.mean(A))
        })
    return tests

# ----------------------------
# 6. Slicing
# ----------------------------
def generate_slicing_tests():
    tests = []
    for _ in range(5):
        A = np.random.randn(3, 3)
        slice_ = A[:2, 1:]  # example slice
        tests.append({
            "A": to_flat_json(A),
            "slice_indices": [[0,2], [1,3]],  # start:end per axis
            "result": to_flat_json(slice_)
        })
    return tests

# ----------------------------
# 7. Transpose
# ----------------------------
def generate_transpose_tests():
    tests = []
    for _ in range(5):
        A = np.random.randn(2, 3)
        result = A.T
        tests.append({
            "A": to_flat_json(A),
            "result": to_flat_json(result)
        })
    return tests

# ----------------------------
# 8. Power
# ----------------------------
def generate_power_tests():
    tests = []
    for _ in range(5):
        A = np.random.rand(2, 2) * 5  # positive values
        exponent = 2
        result = np.power(A, exponent)
        tests.append({
            "A": to_flat_json(A),
            "exponent": exponent,
            "result": to_flat_json(result)
        })
    return tests

# ----------------------------
# 9. Ones
# ----------------------------
def generate_ones_tests():
    ones_array = np.ones((2, 3))
    return [{
        "A": {"method": "ones", "shape": list(ones_array.shape)},
        "result": to_flat_json(ones_array)
    }]

# ----------------------------
# 10. Full
# ----------------------------
def generate_full_tests():
    full_array = np.full((3, 2), 7.5)
    return [{
        "A": {"method": "full", "shape": list(full_array.shape), "fillValue": 7.5},
        "result": to_flat_json(full_array)
    }]

# ----------------------------
# 11. Line Space
# ----------------------------
def generate_linspace_tests():
    linspace_array = np.linspace(0, 5, 6)
    return [{
        "A": {"method": "linspace", "start": 0, "stop": 5, "num": 6},
        "result": to_flat_json(linspace_array)
    }]

# ----------------------------
# 12. Eye
# ----------------------------
def generate_eye_tests():
    eye_array = np.eye(3)
    return [{
        "A": {"method": "eye", "n": 3},
        "result": to_flat_json(eye_array)
    }]

# ----------------------------
# 13. Flatten
# ----------------------------
def generate_flatten_tests():
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    return [{
        "A": {"method": "flatten", "data": arr.flatten().tolist(), "shape": list(arr.shape)},
        "result": to_flat_json(arr.flatten())
    }]
# ----------------------------
# 14. Ravel
# ----------------------------
def generate_ravel_tests():
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    return [{
        "A": {"method": "ravel", "data": arr.flatten().tolist(), "shape": list(arr.shape)},
        "result": to_flat_json(arr.ravel())
    }]

# ----------------------------
# 15. Reshape
# ----------------------------
def generate_reshape_tests2():
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    new_shape = (3, 2)
    return [{
        "A": {"method": "reshape", "data": arr.flatten().tolist(), "shape": list(arr.shape), "newShape": list(new_shape)},
        "result": to_flat_json(arr.reshape(new_shape))
    }]

# ----------------------------
# 16. Copy
# ----------------------------
def generate_copy_tests():
    arr = np.array([[1, 2], [3, 4]])
    return [{
        "A": {"method": "copy", "data": arr.flatten().tolist(), "shape": list(arr.shape)},
        "result": to_flat_json(arr)
    }]

# ----------------------------
# 17. Element-wise subtract
# ----------------------------
def generate_subtract_tests():
    tests = []
    for _ in range(5):
        A = np.random.randn(2, 2)
        B = np.random.randn(2, 2)
        result = A - B
        tests.append({
            "A": to_flat_json(A),
            "B": to_flat_json(B),
            "result": to_flat_json(result)
        })
    return tests

# ----------------------------
# 18. Element-wise multiply
# ----------------------------
def generate_multiply_tests():
    tests = []
    for _ in range(5):
        A = np.random.randn(2, 3)
        B = np.random.randn(3)
        result = A * B  # broadcasting
        tests.append({
            "A": to_flat_json(A),
            "B": to_flat_json(B),
            "result": to_flat_json(result)
        })
    return tests

# ----------------------------
# 19. Element-wise divide
# ----------------------------
def generate_divide_tests():
    tests = []
    for _ in range(5):
        A = np.random.rand(2, 2) * 10 + 1e-3  # avoid division by zero
        B = np.random.rand(2, 2) * 5 + 1e-3
        result = A / B
        tests.append({
            "A": to_flat_json(A),
            "B": to_flat_json(B),
            "result": to_flat_json(result)
        })
    return tests

# ----------------------------
# 20. Element-wise add
# ----------------------------
def generate_add_tests():
    tests = []
    for _ in range(5):
        # Basic same-shape addition
        A = np.random.randn(2, 2)
        B = np.random.randn(2, 2)
        result = A + B
        tests.append({
            "A": to_flat_json(A),
            "B": to_flat_json(B),
            "result": to_flat_json(result)
        })

    for _ in range(5):
        # Broadcasting example
        A = np.random.randn(2, 3)
        B = np.random.randn(3)  # will broadcast along first axis
        result = A + B
        tests.append({
            "A": to_flat_json(A),
            "B": to_flat_json(B),
            "result": to_flat_json(result)
        })
    return tests

# ----------------------------
# 21. Solve linear systems
# ----------------------------
def generate_solve_tests():
    tests = []

    for _ in range(5):

        # Generate invertible matrix
        while True:
            A = np.random.randn(3, 3)

            # Ensure matrix is not singular
            if abs(np.linalg.det(A)) > 1e-6:
                break

        b = np.random.randn(3)

        x = np.linalg.solve(A, b)

        tests.append({
            "A": to_flat_json(A),
            "b": to_flat_json(b),
            "result": to_flat_json(x)
        })

    return tests

# ----------------------------
# Main function
# ----------------------------
if __name__ == "__main__":
    all_tests = {
        "basic_ops.json": generate_basic_ops(),
        "dot.json": generate_dot_tests(),
        "reshape.json": generate_reshape_tests(),
        "broadcast.json": generate_broadcast_tests(),
        "aggregate.json": generate_aggregate_tests(),
        "slicing.json": generate_slicing_tests(),
        "transpose.json": generate_transpose_tests(),
        "power.json": generate_power_tests(),
        "eye.json":  generate_eye_tests(),
        "ravel.json": generate_ravel_tests(),
        "one.json": generate_ones_tests(),
        "copy.json": generate_copy_tests(),
        "full.json": generate_full_tests(),
        "divide.json": generate_divide_tests(),
        "multiply.json": generate_multiply_tests(),
        "subtract.json": generate_subtract_tests(),
        "add.json": generate_add_tests(),
        "solve.json": generate_solve_tests()
    }

    for filename, content in all_tests.items():
        path = os.path.join("../resources/testcases", filename)
        with open(path, "w") as f:
            json.dump(content, f, indent=4)
        print(f"✅ Wrote {filename}")

