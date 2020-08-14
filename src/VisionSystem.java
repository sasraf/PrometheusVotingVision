import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

public class VisionSystem {

    private static MetaLearner metaLearner;
    private static ModelStack modelStack;

    public static void main(String[] args) throws IOException, ClassNotFoundException {
        // Test
        metaLearner = new MetaLearner();
        modelStack = new ModelStack();

        //TODO: use this when adding algorithms:
//        metaLearner = new MetaLearner(int passedInputs, int passedOutputs, double passedLearningRate);
//        modelStack = new  ModelStack(ArrayList<Algorithm> inputAlgorithms);

        Scanner scanner = new Scanner(System.in);
        String input = "";

        while (!input.equals("q")) {
            System.out.println("Please select what you would like to do or press 'q' to quit:");
            System.out.println("A.) Test Models\nB.) Test ModelStack\nC.) Save MetaLearner\nD.) Load MetaLearner\nE.) Train MetaLearner");

            input = scanner.nextLine();

            if (input.equals("a") || input.equals("A")) {
                String dirPath = getFilePath("Testing/Images", "Enter the name of your folder containing images or press enter to use the default \"Testing/images\" folder: ", scanner);

                // Processes every image in the dirPath using the modelStack, consolidates the modelStack's outputs
                double[][] metaLearnerInputs = processAndConsolidateOutputs(dirPath);

                // Pass inputs to metaLearner
                double[][] metaOutputs = metaLearner.feedForward(metaLearnerInputs);

                System.out.println(Arrays.deepToString(metaOutputs));
            }
            // Print processImage output
            else if (input.equals("b") || input.equals("B")) {
                String dirPath = getFilePath("Testing/Images/0.png", "Enter the filepath of an image to test the modelStack on or press enter to use the default \"Testing/Images/0.png\" file: ", scanner);

                double[] stackOutput = modelStack.processImage("test");
                System.out.println();

                for (int i = 0; i < stackOutput.length; i++) {
                    System.out.print(stackOutput[i] + ", ");
                }
                System.out.println();
            }
            // Save metalearner
            else if (input.equals("c") || input.equals("C")) {
                String dirPath = getFilePath("Testing/metaLearnerSave.txt", "Enter the filepath or press enter to use the default \"Testing/metaLearnerSave.txt\" file: ", scanner);
                metaLearner.save(dirPath);

            }
            // load metalearner
            else if (input.equals("d") || input.equals("D")) {
                String dirPath = getFilePath("Testing/metaLearnerSave.txt", "Enter the filepath or press enter to use the default \"Testing/metaLearnerSave.txt\" file: ", scanner);
                metaLearner.load(dirPath);
            }
            // Train metalearner
            // Currently takes in a serialized array as its expectedOutput
            else if (input.equals("e") || input.equals("E")) {
                String dirPath = getFilePath("Testing/Images/imageKey.txt", "Enter the filepath or press enter to use the default \"Testing/Images\". File must contain serialized double[][] of expectedOutputs: ", scanner);

                // Deserialize array of expected outputs
                FileInputStream readFile = new FileInputStream(dirPath);
                ObjectInputStream in = new ObjectInputStream(readFile);
                double[][] expectedOutputs = (double[][]) in.readObject();
                in.close();
                readFile.close();

                dirPath = getFilePath("Testing/Images", "Enter the name of your folder containing images or press enter to use the default \"Testing/Images\" folder: ", scanner);

                // Processes every image in the dirPath using the modelStack, consolidates the modelStack's outputs
                double[][] metaLearnerInputs = processAndConsolidateOutputs(dirPath);

                System.out.println("Enter number of epochs: ");
                int epochs = scanner.nextInt();
                metaLearner.train(metaLearnerInputs, expectedOutputs, epochs);
            } else {
                System.out.println("Pick an action from one of the below choices:");
            }
        }
    }

    // Processes every image in the dirPath using the modelStack, consolidates the modelStack's outputs, returns those outputs
    private static double[][] processAndConsolidateOutputs(String dirPath) {
        // Creates an arraylist of the paths of each image in the given directory
        ArrayList<String> imagePaths = listImagesInDir(dirPath);

        // For each image, outputs of all algorithms into processedImage, collect those inputs into a metaLearnerInput double array to be passed to the metaLearner
        double[] processedImage = modelStack.processImage(imagePaths.get(0));
        double[][] metaLearnerInputs = new double[imagePaths.size()][processedImage.length];
        metaLearnerInputs[0] = processedImage;
        for (int i = 1; i < imagePaths.size(); i++) {
            processedImage = modelStack.processImage(imagePaths.get(i));
            metaLearnerInputs[i] = processedImage;
        }

        return metaLearnerInputs;
    }

    // Displays a prompt and returns the filePath the user would like to use, otherwise returns default filepath
    private static String getFilePath(String defaultString, String prompt, Scanner scanner) {
        // Set a dir containing images to pass through the visionsystem
        System.out.println(prompt);
        String input = scanner.nextLine();
        if (!input.equals("")) {
            defaultString = input;
        }

        return defaultString;
    }


    // Given a folder with images, creates an arraylist of the images' filePaths
    private static ArrayList<String> listImagesInDir(String filePath) {
        ArrayList<String> imagePaths = new ArrayList<String>();

        // Check that given path is a folder
        File thisdir = new File(filePath);
        if (!thisdir.isDirectory()) {
            throw new IllegalArgumentException("Image filePath is not a directory");
        }

        File[] files = new File(filePath).listFiles();

        // Checks that there are more than 0 images in folder
        if (files.length == 0) {
            throw new IllegalArgumentException("There are no files in the image filePath");
        }

        // Adds image paths to arraylist
        for (File file : files) {
            imagePaths.add(filePath + file.getName());
        }

        return imagePaths;
    }
}
