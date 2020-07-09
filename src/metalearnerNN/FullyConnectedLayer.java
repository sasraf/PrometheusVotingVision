package metalearnerNN;

public class FullyConnectedLayer implements Layer {

    // Initializes a fully connected layer with a inputSize x outputSize matrix of weights and a 1 x outputSize matrix
    // of biases

    private double[][] weights;
    private double[] biases;
    private double[] input;
    private double[] output;

    public FullyConnectedLayer (int inputSize, int outputSize) {

        // Creates weights as a randomized matrix of inpusize x outputsize
        weights = new double[inputSize][outputSize];
        for (int i = 0; i < inputSize; i++) {
            for (int n = 0; n < outputSize; n++) {
                weights[i][n] = Math.random(); //TODO in the tutorial there's a "- .5"... make sure not necessary
            }
        }

        // Creates biases as a randomized 1 x outputsize matrix
        biases = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            biases[i] = Math.random(); //TODO in the tutorial there's a "- .5"... make sure not necessary
        }
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
    public double[] backProp(double[] outputError, double learningRate) {

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
