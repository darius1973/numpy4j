package org.numpy4j.numpytests;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.numpy4j.core.NDArray;
import org.numpy4j.linalg.LinearAlgebra;

import java.io.InputStream;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class NDArrayJsonTest {

    private final ObjectMapper mapper = new ObjectMapper();

    private JsonNode loadTestCase(String filename) throws Exception {
        InputStream is = getClass().getClassLoader().getResourceAsStream("testcases/" + filename);
        if (is == null) throw new RuntimeException("Cannot find test file: " + filename);
        return mapper.readTree(is);
    }

    private NDArray createNDArray(JsonNode node) {
        double[] data = mapper.convertValue(node.get("data"), double[].class);
        int[] shape = mapper.convertValue(node.get("shape"), int[].class);
        return new NDArray(data, shape);
    }

    @Test
    public void runAllJsonTests() throws Exception {
        String[] files = {
                "basic_ops.json",
                "dot.json",
                "reshape.json",
                "broadcast.json",
                "aggregate.json",
                "slicing.json",
                "transpose.json",
                "power.json"
        };

        for (String file : files) {
            JsonNode root = loadTestCase(file);
            for (JsonNode testCase : root) {
                switch (file) {
                    case "basic_ops.json":
                        NDArray A = createNDArray(testCase.get("A"));
                        NDArray B = createNDArray(testCase.get("B"));
                        NDArray expectedAdd = createNDArray(testCase.get("result"));
                        NDArray resultAdd = A.add(B);
                        assertArrayEquals(expectedAdd.getData(), resultAdd.getData(), 1e-9);
                        break;
                    case "dot.json":
                        NDArray dotA = createNDArray(testCase.get("A"));
                        NDArray dotB = createNDArray(testCase.get("B"));
                        NDArray expectedDot = createNDArray(testCase.get("result"));
                        NDArray resultDot = dotA.dot(dotB); // implement dot in NDArray
                        assertArrayEquals(expectedDot.getData(), resultDot.getData(), 1e-9);
                        break;
                    case "reshape.json":
                        NDArray reshapeA = createNDArray(testCase.get("A"));
                        int[] newShape = mapper.convertValue(testCase.get("new_shape"), int[].class);
                        NDArray expectedReshape = createNDArray(testCase.get("result"));
                        NDArray resultReshape = reshapeA.reshape(newShape);
                        assertArrayEquals(expectedReshape.getData(), resultReshape.getData(), 1e-9);
                        break;

                    case "broadcast.json":
                        NDArray broadcastA = createNDArray(testCase.get("A"));
                        NDArray broadcastB = createNDArray(testCase.get("B"));
                        NDArray expectedBroadcast = createNDArray(testCase.get("result"));
                        NDArray resultBroadcast = broadcastA.add(broadcastB); // broadcasting sum
                        assertArrayEquals(expectedBroadcast.getData(), resultBroadcast.getData(), 1e-9);
                        break;

                    case "aggregate.json":
                        NDArray aggA = createNDArray(testCase.get("A"));
                        double expectedSum = testCase.get("sum").asDouble();
                        double expectedMean = testCase.get("mean").asDouble();
                        assertEquals(expectedSum, aggA.sum(), 1e-9);     // implement sum() in NDArray
                        assertEquals(expectedMean, aggA.mean(), 1e-9);   // implement mean() in NDArray
                        break;

                    case "slicing.json":
                        NDArray sliceA = createNDArray(testCase.get("A"));
                        int[][] sliceIndices = mapper.convertValue(testCase.get("slice_indices"), int[][].class);
                        NDArray expectedSlice = createNDArray(testCase.get("result"));
                        NDArray resultSlice = sliceA.slice(sliceIndices); // implement slice() in NDArray
                        assertArrayEquals(expectedSlice.getData(), resultSlice.getData(), 1e-9);
                        break;

                    case "transpose.json":
                        NDArray transA = createNDArray(testCase.get("A"));
                        NDArray expectedTrans = createNDArray(testCase.get("result"));
                        NDArray resultTrans = LinearAlgebra.transpose(transA);
                        assertArrayEquals(expectedTrans.getData(), resultTrans.getData(), 1e-9);
                        break;

                    case "power.json":
                        NDArray powA = createNDArray(testCase.get("A"));
                        int exponent = testCase.get("exponent").asInt();
                        NDArray expectedPow = createNDArray(testCase.get("result"));
                        NDArray resultPow = powA.power(exponent);
                        assertArrayEquals(expectedPow.getData(), resultPow.getData(), 1e-9);
                        break;
                }
            }
        }
    }
}



