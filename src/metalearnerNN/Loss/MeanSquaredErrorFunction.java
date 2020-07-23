package metalearnerNN.Loss;

import metalearnerNN.Loss.Loss;
import metalearnerNN.Matrix;

import java.io.Serializable;
import java.util.Arrays;

public class MeanSquaredErrorFunction implements Loss, Serializable {

    public MeanSquaredErrorFunction () {
    }

    // Returns meanSquaredError of input (mean of (actual - expected)^2)
    public double function(double[] actual, double[] expected) {

        //TODO: TEST
//        System.out.println("Input: " + Arrays.toString(actual));
//        System.out.println("Expected: " + Arrays.toString(expected));
        double[] holder = Matrix.matrixSubtract(actual, expected);
//        System.out.println("Matrix Subtract: " + Arrays.toString(holder));
        holder = Matrix.constantPower(holder, 2);
//        System.out.println("ConstantPower: " + Arrays.toString(holder));
        double returnVal = Matrix.mean(holder);
//        System.out.println("Loss: " + returnVal);
        return returnVal;


        //TODO: only line that should be here
//        return Matrix.mean(Matrix.constantPower(Matrix.matrixSubtract(actual, expected), 2));
    }

    // Returns derivative of mean squared error (2 * (expected - actual) / number of inputs)
    public double[] derivative(double[] actual, double[] expected) {
        return Matrix.constantMultiply(Matrix.constantDivide(Matrix.matrixSubtract(expected, actual), actual.length), 2);
    }
}
