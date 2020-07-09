package metalearnerNN;

public class XORTest {
    public static void main(String[] args) {

        // Sample training data
        double[][] inputData = new double[][] {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        double[][] expectedOutput = new double[][] {{0}, {1}, {1}, {0}};

        // Setting up network with 3 layers & tanh activation functions
        NeuralNetwork network = new NeuralNetwork();
        network.addLayer(new FullyConnectedLayer(2, 3));
        network.addLayer(new ActivationLayer(new tanhActivationFunction()));
        network.addLayer(new FullyConnectedLayer(3, 1));
        network.addLayer(new ActivationLayer(new tanhActivationFunction()));

        // Training
        network.setLoss(new MeanSquaredErrorFunction());
        network.train(inputData, expectedOutput, 1000, .1);

        // Test network
        double[][] output = network.predict(inputData);
        for (int i = 0; i < inputData.length; i++) {
            System.out.println("For set " + inputData[i].toString() + " my prediction is " + output[i].toString() + " while the correct value is " + expectedOutput[i]);
        }
    }
}
