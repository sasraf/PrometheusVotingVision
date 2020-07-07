package metalearnerNN;

public class FullyConnectedLayer {

    // Initializes a fully connected layer with a inputSize x outputSize matrix of weights and a 1 x outputSize matrix
    // of biases

    double[][] weights;
    double[] biases;
    double[] input;
    double[] output;

    public FullyConnectedLayer (int inputSize, int outputSize) {

        // Creates weights as a randomized matrix of inpusize x outputsize
        weights = new double[inputSize][outputSize];
        //TODO intiialize weights

        // Creates biases as a randomized 1 x outputsize matrix
        biases = new double[outputSize];
        //TODO intialize biases
    }

    // Feeds an array of inputs from a layer through this layer and returns the output
    // of this layer
    public double[] feedForward(double[] passedInput) {
        input = passedInput;

        //Returns a 1 x outputsize matrix of output values corresponding to each neuron in the next layer
        output = Matrix.matrixMultiply(input, weights);
        output = Matrix.matrixAdd(output, biases);

        return output;
    }

    //TODO: make everything double[][] instead of double[] ??????
    //TODO that's how numpy operates and might make things a lot easier

    // Adjusts weights/biases, returns derivative of error with respect to input
    public double[] backprop(double[] outputError, double learningRate) {

        double[] inputError = Matrix.matrixMultiply(outputError, Matrix.transpose(weights));

        //TODO: quickfix: turn outputerror into a 2d array for matrix multiplication; find a more elegant solution
        double[][] twoDOutputError = new double[1][outputError.length];
        for (int i = 0; i < outputError.length; i++) {
            twoDOutputError[0][i] = outputError[i];
        }

        double[][] weightsError = Matrix.matrixMultiply(twoDOutputError, Matrix.transpose(input));

        weights = Matrix.matrixSubtract(weights, Matrix.constantMultiply(weightsError, learningRate));
        biases = Matrix.matrixSubtract(biases, Matrix.constantMultiply(outputError, learningRate));

        return inputError;


    }

}
