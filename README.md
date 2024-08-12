# o1js Example

## Introduction

This project demonstrates the implementation of both Linear Regression and a Multi-Layer Perceptron (MLP) using the `o1js` library. 
The Linear Regression model serves as a fundamental building block, while the MLP is designed to take five input values and predict an output through two hidden layers. 

## How to Run

### Environment

- Node.js (`v20.11.1`)

### Steps to Execute

1. **Build the project**: This step compiles TypeScript code (`index.ts`) into JavaScript (`index.js`).
   
   ```bash
   npm run build
   ```

2. **Run the project**: Execute the compiled JavaScript file.
   
   ```bash
   npm run start
   ```

## MLP Implementation

### Overview

The MLP (Multi-Layer Perceptron) implemented in this project consists of the following components:

- **Input Layer**: Takes 5 input values.
- **Hidden Layer 1**: Applies linear transformation followed by a ReLU activation function.
- **Hidden Layer 2**: Takes the output of the first hidden layer, applies another linear transformation, and passes it through another ReLU activation.
- **Output Layer**: Produces the final output value using a linear transformation.

### Diagram of MLP Structure

To better understand the MLP structure, refer to the following diagram, which represents the data flow and operations within the network:

```
Input Layer (5 inputs)
  |
  v
+------------------------+
|  Hidden Layer 1        |  (Linear Regression + ReLU)
|  Weighted Sum (z1)     | -> ReLU(z1) -> a1
+------------------------+
  |
  v
+------------------------+
|  Hidden Layer 2        |  (Linear Regression + ReLU)
|  Weighted Sum (z2)     | -> ReLU(z2) -> a2
+------------------------+
  |
  v
+------------------------+
|  Output Layer          |  (Linear Regression)
|  Weighted Sum (z3)     | -> Output (z3)
+------------------------+
  |
  v
Output
```

## Benchmarking

To benchmark the performance of the Linear Regression and MLP implementations, you can use `gtime` to measure execution time. Follow these steps:

1. **Install `gtime`**: If `gtime` is not already installed on your system, you can install it using the provided shell script.

   Run the script to install `gtime`:

   ```bash
   sh install_gtime.sh
   ```

2. **Run the benchmarks**: After installing `gtime`, you can benchmark the two models by running the following commands:

   ```bash
   gtime node dist/mlp.js
   gtime node dist/linear_regression.js
   ```

   These commands will execute the respective scripts and display the execution time, allowing you to compare the performance of the MLP and Linear Regression models.

## Result

```bash
# o1js
> o1js_example@1.0.0 start
> node dist/index.js

start
making proof
proof created
value:  525
Proof is valid: true
```
