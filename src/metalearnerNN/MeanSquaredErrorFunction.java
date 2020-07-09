package metalearnerNN;

public class MeanSquaredErrorFunction implements Loss {

    public MeanSquaredErrorFunction () {
    }

    // Returns meanSquaredError of input (mean of (actual - expected)^2)
    public double function(double[] actual, double[] expected) {
        return Matrix.mean(Matrix.constantPower(Matrix.matrixSubtract(actual, expected), 2));
    }


    // Returns derivative of mean squared error (2 * (expected - actual) / number of inputs)
    public double[] derivative(double[] actual, double[] expected) {
        return Matrix.constantMultiply(Matrix.constantDivide(Matrix.matrixSubtract(expected, actual), actual.length), 2);
    }
}
