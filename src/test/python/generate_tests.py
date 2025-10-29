import numpy as np
import json
import os


def generate_add_test():
    a = np.random.rand(2, 3)
    b = np.random.rand(2, 3)
    result = a + b

    return {
        "name": "add_test_2x3",
        "operation": "add",
        "a": a.tolist(),
        "b": b.tolist(),
        "shape": [2, 3],
        "result": result.tolist()
    }


def generate_reshape_test():
    a = np.arange(6).reshape((2, 3))
    reshaped = a.reshape((3, 2))

    return {
        "name": "reshape_test_2x3_to_3x2",
        "operation": "reshape",
        "a": a.tolist(),
        "from_shape": [2, 3],
        "to_shape": [3, 2],
        "result": reshaped.tolist()
    }


def write_tests(output_dir="tests/json"):
    os.makedirs(output_dir, exist_ok=True)
    tests = [generate_add_test(), generate_reshape_test()]
    for test in tests:
        filename = f"{test['name']}.json"
        with open(os.path.join(output_dir, filename), 'w') as f:
            json.dump(test, f, indent=2)


if __name__ == '__main__':
    write_tests()
