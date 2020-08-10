import metalearnerNN.ActivationLayer;
import metalearnerNN.Activations.softmaxActivationFunction;
import metalearnerNN.Activations.tanhActivationFunction;
import metalearnerNN.FullyConnectedLayer;
import metalearnerNN.Loss.MeanSquaredErrorFunction;
import metalearnerNN.NeuralNetwork;

import java.io.IOException;

public class MetaLearner {

    private NeuralNetwork network;

    // Default seetings:
    private final int inputs = 16;
    private final int outputs = 4;
    private double learningRate = .1;

    // Default settings
    public MetaLearner() {
        network = new NeuralNetwork();
        network.setLoss(new MeanSquaredErrorFunction());
        network.addLayer(new FullyConnectedLayer(inputs, 12));
        network.addLayer(new ActivationLayer(new tanhActivationFunction()));
        network.addLayer(new FullyConnectedLayer(12, outputs));
        network.addLayer(new ActivationLayer(new softmaxActivationFunction()));

    }

    // Allows easy initialization of MetaLearner with different settings
    public MetaLearner(int passedInputs, int passedOutputs, double passedLearningRate) {
        network = new NeuralNetwork();
        network.setLoss(new MeanSquaredErrorFunction());
        network.addLayer(new FullyConnectedLayer(passedInputs, 12));
        network.addLayer(new ActivationLayer(new tanhActivationFunction()));
        network.addLayer(new FullyConnectedLayer(12, passedOutputs));
        network.addLayer(new ActivationLayer(new softmaxActivationFunction()));

        learningRate = passedLearningRate;

    }

    // Saves weights and biases to a file
    public void save(String path) throws IOException {
        network.save(path);
    }

    public void load(String path) throws IOException, ClassNotFoundException {
        network = network.load(path);
    }

    //Runs backprop algorithm
    public void train(double[][] inputs, double[][] expected, int epochs) {
        network.train(inputs, expected, epochs, learningRate);
    }

    // Feedforwards inputs, returns output of neural network
    public double[][] feedForward(double[][] inputs) {
        return network.predict(inputs);
    }
}
