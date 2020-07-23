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
import java.util.Arrays;

// Test vision system used for the purpose of using my Neural networks to test if ensemble learning is more accurate than the algos alone
public class TestVisionSystem {

    // path where serliazed MNIST data is stored
    private static final String path = "src/testImages/mnist_train";

    private static double[] oneHotEncodeMnist(double num) {
        double[] oneHotEncoded = new double[10];
        for (int i = 0; i < oneHotEncoded.length; i++) {
            oneHotEncoded[i] = 0;
        }
        oneHotEncoded[(int) num] = 1;

        return oneHotEncoded;
    }

    public static void main(String[] args) throws IOException, ClassNotFoundException {

        ArrayList<double[][]> outputs = new ArrayList<double[][]>();
        AlgoForTesting[] algos = new AlgoForTesting[4];


        double[][] mnist;

        FileInputStream readFile = new FileInputStream(path);
        ObjectInputStream in = new ObjectInputStream(readFile);
        mnist = (double[][]) in.readObject();
        in.close();
        readFile.close();

        double[][] inputData = new double[mnist.length][mnist[0].length - 1];
        double[][] expectedOutput = new double[mnist.length][10];

        for (int i = 0; i < mnist.length; i++) {
            expectedOutput[i] = oneHotEncodeMnist(mnist[i][0]);
            for (int n = 1; n < mnist[0].length; n++) {
                inputData[i][n - 1] = mnist[i][n];
            }
        }


        for (int i = 0; i < 4; i++) {
            algos[i] = new AlgoForTesting();
            algos[i].load(i);
            outputs.add(algos[i].processImage(inputData));
        }

        //outputs is an arraylist containing double[][] of outputs from the image processing algos

        //TODO: need tripple for loop

        double[][] holder = new double[outputs.get(0).length][outputs.size() * outputs.get(0)[0].length];

        // [ [ i, i, i ] j, j, j ]
        // [ [ i, i, i ] j, j, j ]
        // k
        // k

        // [ [ 1, 2 ], [ 3, 4 ] ]
        // [ [ 5, 6 ], [ 7, 8 ] ]
        // -> [ [ 1, 2, 5, 6 ], [ 3, 4, 7, 8 ] ]

//        int counter = 0;


        for (int j = 0; j < outputs.get(0).length; j++) {
            int counter = 0;
            for (int k = 0; k < outputs.size(); k++) {
                for (int i = 0; i < outputs.get(0)[0].length; i++) {
                    holder[j][counter] = outputs.get(k)[j][i];
                    counter++;
                }
            }
        }


        System.out.println(Arrays.toString(holder[0]));
        System.out.println("output length == input length: " + (holder.length == outputs.get(0).length));
        System.out.println("holder[0].length: " + holder[0].length);

        NeuralNetwork network = new NeuralNetwork();
        network.addLayer(new FullyConnectedLayer(40, 15));
        network.addLayer(new ActivationLayer(new tanhActivationFunction()));
        network.addLayer(new FullyConnectedLayer(15, 10));
        network.addLayer(new ActivationLayer(new softmaxActivationFunction()));
        network.setLoss(new MeanSquaredErrorFunction());

        // Training and keeping track of time it takes to train
        long startTime = System.nanoTime();
        network.train(holder, expectedOutput, 10, .1);
        long endTime = System.nanoTime();
        long trainingTime = endTime - startTime;
        System.out.println("Trained in " + trainingTime / 1000000 + "ms");

        //TODO: uncomment from here down

//
//        System.out.println("holder.length: " + holder.length + "\nholder[0].length: " + holder[0].length + "\nexpectedout.length: " + expectedOutput.length + "\nexpectedOutput[0].length " + expectedOutput[0].length);
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
//
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
//            testExpectedOutput[i] = newexpectedOutput[i];
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
//        double[][] metaPredict = network.predict(testInputData);
//        for (int i = 0; i < predictions1.length; i++) {
//            System.out.println("Prediction: " + Arrays.toString(metaPredict[i]) + " Actual: " + Arrays.toString(testExpectedOutput[i]));
//        }


    }
}
