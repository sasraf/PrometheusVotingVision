package Testing;

import metalearnerNN.ActivationLayer;
import metalearnerNN.Activations.softmaxActivationFunction;
import metalearnerNN.Activations.tanhActivationFunction;
import metalearnerNN.FullyConnectedLayer;
import metalearnerNN.Loss.Loss;
import metalearnerNN.Loss.MeanSquaredErrorFunction;
import metalearnerNN.NeuralNetwork;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.ArrayList;

// Test vision system used for the purpose of using my Neural networks to test if ensemble learning is more accurate than the algos alone
public class TestVisionSystem {

    // path where serliazed MNIST data is stored
    private static final String trainPath = "src/Testing/TestData/mnist_train";
    private static final String testPath = "src/Testing/TestData/mnist_test";
    private static final Loss errorFunction = new MeanSquaredErrorFunction();


    public static void main(String[] args) throws IOException, ClassNotFoundException {

        ArrayList<double[][]> outputs = new ArrayList<double[][]>();
        AlgoForTesting[] algos = new AlgoForTesting[4];

        // Deserializes saved MNIST arrays
        double[][][] deserialized = deserializeMNIST(trainPath);
        double[][] inputData = deserialized[0];
        double[][] expectedOutputs = deserialized[1];

        // Loads and runs saved algos over deserialized mnist data
        for (int i = 0; i < 4; i++) {
            algos[i] = new AlgoForTesting();
            algos[i].load(i);
            outputs.add(algos[i].processImage(inputData));
            displayErrorAndAccuracy(outputs.get(i), expectedOutputs, ("LoadedTestNetwork" + (i + 1)));
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
        System.out.println("\n\nTraining MetaLearner:");
        long startTime = System.nanoTime();
        metaLearner.train(holder, expectedOutputs, 20, .01);
        long endTime = System.nanoTime();
        long trainingTime = endTime - startTime;
        System.out.println("Trained in " + trainingTime / 1000000 + "ms\n");

        // Test with test data
        System.out.println("Testing Individual Networks and MetaLearner with Individual Networks:");
        double[][][] testMNIST = deserializeMNIST(testPath);
        double[][] testInputData = testMNIST[0];
        double[][] testExpectedOutputs = testMNIST[1];

        outputs.clear();
        for (int i = 0; i < algos.length; i++) {
            outputs.add(algos[i].processImage(testInputData));
        }

        for (int i = 0; i < outputs.size(); i++) {
            double[][] setOfOutputs = outputs.get(i);
            displayErrorAndAccuracy(setOfOutputs, testExpectedOutputs, ("TestNetwork" + (i + 1)));
        }

        double[][] consolidatedInputs = consolidateInputs(outputs);
        double[][] metaOutput = metaLearner.predict(consolidatedInputs);
        displayErrorAndAccuracy(metaOutput, testExpectedOutputs, "MetaLearner");
    }

    // Prints out mse Error and Accuracy
    private static void displayErrorAndAccuracy(double[][] prediction, double[][] actual, String networkName) {
        double error = 0;
        double numOfTimesCorrect = 0;
        for (int i = 0; i < prediction.length; i++) {
            error += errorFunction.function(actual[i], prediction[i]);

            double max = -100;
            int maxIndex = 0;
            for (int j = 0; j < prediction[i].length; j++) {
                if (prediction[i][j] > max) {
                    max = prediction[i][j];
                    maxIndex = j;
                }
            }
            if (actual[i][maxIndex] == 1) {
                numOfTimesCorrect++;
            }
        }

        double accuracy = (numOfTimesCorrect / prediction.length) * 100;
        error = error / prediction.length;

        System.out.println(networkName + " ran over " + prediction.length + " test cases with an avg error of: " + error + " with an accuracy of: " + accuracy + "%");
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
    private static double[][][] deserializeMNIST(String MNISTPath) throws IOException, ClassNotFoundException {
        System.out.println("Deserializing MNIST Array Stored in " + MNISTPath);
        FileInputStream readFile = new FileInputStream(MNISTPath);
        ObjectInputStream in = new ObjectInputStream(readFile);
        double mnist[][] = (double[][]) in.readObject();
        in.close();
        readFile.close();

        System.out.println("Converting Array into inputData and expectedOutputs arrays");
        double[][] inputData = new double[mnist.length][mnist[0].length - 1];
        double[][] expectedOutputs = new double[mnist.length][10];

        for (int i = 0; i < mnist.length; i++) {
            expectedOutputs[i] = oneHotEncodeMnist(mnist[i][0]);
            for (int n = 1; n < mnist[0].length; n++) {
                inputData[i][n - 1] = mnist[i][n];
            }
        }
        double[][][] outputs = new double[2][][];
        outputs[0] = inputData;
        outputs[1] = expectedOutputs;
        return outputs;
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
