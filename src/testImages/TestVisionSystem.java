package testImages;

import metalearnerNN.ActivationLayer;
import metalearnerNN.Activations.softmaxActivationFunction;
import metalearnerNN.Activations.tanhActivationFunction;
import metalearnerNN.FullyConnectedLayer;
import metalearnerNN.Loss.MeanSquaredErrorFunction;
import metalearnerNN.NeuralNetwork;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.ArrayList;

// Test vision system used for the purpose of using my Neural networks to test if ensemble learning is more accurate than the algos alone
public class TestVisionSystem {

    // path where serliazed MNIST data is stored
    private static final String trainPath = "src/testImages/mnist_train";
    private static final String testPath = "src/testImages/mnist_test";
    private static double[][] mnist;
    private static double[][] inputData;
    private static double[][] expectedOutputs;



    public static void main(String[] args) throws IOException, ClassNotFoundException {

        ArrayList<double[][]> outputs = new ArrayList<double[][]>();
        AlgoForTesting[] algos = new AlgoForTesting[4];

        deserializeMNIST(trainPath);

        for (int i = 0; i < 4; i++) {
            algos[i] = new AlgoForTesting();
            algos[i].load(i);
            outputs.add(algos[i].processImage(inputData));
        }

        double[][] holder = consolidateInputs(outputs);

        // Create metalearner
        NeuralNetwork metaLearner = new NeuralNetwork();
        metaLearner.addLayer(new FullyConnectedLayer(40, 15));
        metaLearner.addLayer(new ActivationLayer(new tanhActivationFunction()));
        metaLearner.addLayer(new FullyConnectedLayer(15, 10));
        metaLearner.addLayer(new ActivationLayer(new softmaxActivationFunction()));
        metaLearner.setLoss(new MeanSquaredErrorFunction());

        // Training and keeping track of time it takes to train
        long startTime = System.nanoTime();
        metaLearner.train(holder, expectedOutputs, 10, .1);
        long endTime = System.nanoTime();
        long trainingTime = endTime - startTime;
        System.out.println("Trained in " + trainingTime / 1000000 + "ms\n");

        // Test with test data
        deserializeMNIST(testPath);
        outputs.clear();
        for (int i = 0; i < algos.length; i++) {
            outputs.add(algos[i].processImage(inputData));
        }
        double[][] consolidatedInputs = consolidateInputs(outputs);

        double[][] metaOutput = metaLearner.predict(consolidatedInputs);

        MeanSquaredErrorFunction mse = new MeanSquaredErrorFunction();

        for (int i = 0; i < outputs.size(); i++) {
            double[][] setOfOutputs = outputs.get(i);
            double error = 0;
            for (int n = 0; n < setOfOutputs.length; n++) {
                error += mse.function(expectedOutputs[i], setOfOutputs[n]);
            }
            error = error / setOfOutputs.length;

            System.out.println("TestNetwork" + (i + 1) + " ran with an avg error of: " + error);
        }

        double error = 0;
        for (int i = 0; i < metaOutput.length; i++) {
            error += mse.function(expectedOutputs[i], metaOutput[i]);
        }
        error = error / metaOutput.length;
        System.out.println("MetaLearner ran with an avg error of: " + error);
    }

    // Onehotencodes MNIST labels
    private static double[] oneHotEncodeMnist(double num) {
        double[] oneHotEncoded = new double[10];
        for (int i = 0; i < oneHotEncoded.length; i++) {
            oneHotEncoded[i] = 0;
        }
        oneHotEncoded[(int) num] = 1;

        return oneHotEncoded;
    }

    // deserializes MNIST from given path, turns into input and outputdata
    private static void deserializeMNIST(String MNISTPath) throws IOException, ClassNotFoundException {
        FileInputStream readFile = new FileInputStream(MNISTPath);
        ObjectInputStream in = new ObjectInputStream(readFile);
        mnist = (double[][]) in.readObject();
        in.close();
        readFile.close();

        inputData = new double[mnist.length][mnist[0].length - 1];
        expectedOutputs = new double[mnist.length][10];

        for (int i = 0; i < mnist.length; i++) {
            expectedOutputs[i] = oneHotEncodeMnist(mnist[i][0]);
            for (int n = 1; n < mnist[0].length; n++) {
                inputData[i][n - 1] = mnist[i][n];
            }
        }

    }

    // Consolidates a arraylist of nn outputs into a single array for inputting into metalearner
    private static double[][] consolidateInputs(ArrayList<double[][]> stackOutputs) {
        double[][] holder = new double[stackOutputs.get(0).length][stackOutputs.size() * stackOutputs.get(0)[0].length];

        for (int j = 0; j < stackOutputs.get(0).length; j++) {
            int counter = 0;
            for (int k = 0; k < stackOutputs.size(); k++) {
                for (int i = 0; i < stackOutputs.get(0)[0].length; i++) {
                    holder[j][counter] = stackOutputs.get(k)[j][i];
                    counter++;
                }
            }
        }
        return holder;
    }
}
