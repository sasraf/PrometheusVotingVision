import metalearnerNN.ActivationLayer;
import metalearnerNN.Activations.softmaxActivationFunction;
import metalearnerNN.Activations.tanhActivationFunction;
import metalearnerNN.FullyConnectedLayer;
import metalearnerNN.Loss.MeanSquaredErrorFunction;
import metalearnerNN.NeuralNetwork;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.Arrays;

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







        //TODO: uncomment from here down


//        System.out.println("holder.length: " + holder.length + "\nholder[0].length: " + holder[0].length + "\nexpectedout.length: " + expectedOutputs.length + "\nexpectedOutput[0].length " + expectedOutputs[0].length);
//
//
//        // Loading in other networks for comparison
//        NeuralNetwork testNetwork1 = new NeuralNetwork();
//        testNetwork1 = testNetwork1.load("src/TestNetwork1.txt");
//        NeuralNetwork testNetwork2 = new NeuralNetwork();
//        testNetwork2 = testNetwork2.load("src/TestNetwork2.txt");
//        NeuralNetwork testNetwork3 = new NeuralNetwork();
//        testNetwork3 = testNetwork3.load("src/TestNetwork3.txt");
//        NeuralNetwork testNetwork4 = new NeuralNetwork();
//        testNetwork4 = testNetwork4.load("src/TestNetwork4.txt");


//        //TODO: take in data from mnist test and test each network




//
//        readFile = new FileInputStream(path);
//        in = new ObjectInputStream(readFile);
//
//        mnist = (double[][]) in.readObject();
//
//        in.close();
//        readFile.close();
//
//        double[][] newInputData = new double[mnist.length][mnist[0].length - 1];
//        double[][] newexpectedOutput = new double[mnist.length][10];
//
//        for (int i = 0; i < mnist.length; i++) {
//            expectedOutput[i] = oneHotEncodeMnist(mnist[i][0]);
//            for (int n = 1; n < mnist[0].length; n++) {
//                inputData[i][n - 1] = mnist[i][n];
//            }
//        }
//
//        double[][] testInputData = new double[10][newInputData[0].length];
//        double[][] testExpectedOutput = new double[10][newexpectedOutput[0].length];
//        for (int i = 0; i < 10; i++) {
//            testInputData[i] = newInputData[i];
//            testExpectedOutput[i] = expectedOutput[i];
//        }
//
//        double[][] predictions1 = testNetwork1.predict(testInputData);
//        double[][] predictions2 = testNetwork2.predict(testInputData);
//        double[][] predictions3 = testNetwork3.predict(testInputData);
//        double[][] predictions4 = testNetwork4.predict(testInputData);
//
//        outputs.clear();
//        outputs.add(predictions1);
//        outputs.add(predictions2);
//        outputs.add(predictions3);
//        outputs.add(predictions4);
//
//
//        double[][] consolidatedPredictions = new double[outputs.get(0).length][outputs.size() * outputs.get(0)[0].length];
//        for (int j = 0; j < outputs.get(0).length; j++) {
//            int counter = 0;
//            for (int k = 0; k < outputs.size(); k++) {
//                for (int i = 0; i < outputs.get(0)[0].length; i++) {
//                    consolidatedPredictions[j][counter] = outputs.get(k)[j][i];
//                    counter++;
//                }
//            }
//        }
//
//
//
//
//        System.out.println("Network 1");
//        for (int i = 0; i < predictions1.length; i++) {
//            System.out.println("Prediction: " + Arrays.toString(predictions1[i]) + " Actual: " + Arrays.toString(testExpectedOutput[i]));
//        }
//        System.out.println("Network 2");
//        for (int i = 0; i < predictions1.length; i++) {
//            System.out.println("Prediction: " + Arrays.toString(predictions2[i]) + " Actual: " + Arrays.toString(testExpectedOutput[i]));
//        }
//        System.out.println("Network 3");
//        for (int i = 0; i < predictions1.length; i++) {
//            System.out.println("Prediction: " + Arrays.toString(predictions3[i]) + " Actual: " + Arrays.toString(testExpectedOutput[i]));
//        }
//        System.out.println("Network 4");
//        for (int i = 0; i < predictions1.length; i++) {
//            System.out.println("Prediction: " + Arrays.toString(predictions4[i]) + " Actual: " + Arrays.toString(testExpectedOutput[i]));
//        }
//        System.out.println("MetaLearner");
//        double[][] metaPredict = network.predict(consolidatedPredictions);
//        for (int i = 0; i < predictions1.length; i++) {
//            System.out.println("Prediction: " + Arrays.toString(metaPredict[i]) + " Actual: " + Arrays.toString(testExpectedOutput[i]));
//        }


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
