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
        "power.json": generate_power_tests()
    }

    for filename, content in all_tests.items():
        path = os.path.join("../resources/testcases", filename)
        with open(path, "w") as f:
            json.dump(content, f, indent=4)
        print(f"✅ Wrote {filename}")
