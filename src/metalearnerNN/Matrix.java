package metalearnerNN;

import java.lang.Math;

public class Matrix {

    // Returns mean of
    public static double mean(double[] input) {
        double sum = 0;
        for (int i = 0; i < input.length; i++)
        {
            sum += input[i];
        }

        return sum / input.length;
    }

    // Takes each element of the array to the power of the inputted int
    public static double[] constantPower(double[] input, int power) {
        double[] newMatrix = new double[input.length];

        for (int i = 0; i < input.length; i++) {
            newMatrix[i] = Math.pow(input[i], power);
        }

        return newMatrix;
    }

    // Subtracts one matrix from another
    public static double[] matrixSubtract(double[] minuend, double[] subtrahend) {
        if (minuend.length != subtrahend.length) {
            throw new IllegalArgumentException("matrixSubtract failed due to the matrices being different sizes");
        }

        double[] c = new double[minuend.length];

        for (int i = 0; i < minuend.length; i++) {
            c[i] = minuend[i] - subtrahend[i];
        }

        return  c;
    }

    // Matrix subtraction for 2d matrices
    public static  double[][] matrixSubtract(double[][] minuend, double[][] subtrahend) {
        if (minuend.length != subtrahend.length || minuend[0].length != subtrahend[0].length) {
            throw new IllegalArgumentException("matrixSubtract failed due to the matrices being different sizes");
        }

        double[][] output = new double[minuend.length][minuend[0].length];

        for (int i = 0; i < minuend.length; i++) {
            for (int n = 0; n < minuend[0].length; n++) {
                output[i][n] = minuend[i][n] - subtrahend[i][n];
            }
        }

        return output;
    }


    public static double[] matrixAdd(double[] a, double[] b) {

        if (a.length != b.length)
        {
            throw new IllegalArgumentException("matrixAdd failed due to the matrices being different sizes");
        }

        double[] c = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            c[i] = a[i] + b[i];
        }

        return c;
    }

    // Multiplies two matrices
    // Code from https://introcs.cs.princeton.edu/java/22library/Matrix.java.html
    public static double[][] matrixMultiply(double[][] matrix1, double[][] matrix2) {
        // Stores lengths in appropriately labeled variables for ease of writing/reading
        int m1 = matrix1.length;
        int n1 = matrix1[0].length;
        int m2 = matrix2.length;
        int n2 = matrix2[0].length;

        // ensures that matrices are multiplicable
        if (n1 != m2) {
            throw new IllegalArgumentException("Matrices Inputted into Matrix.matrixMultiply cannot be multiplied");
        }

        double[][] output = new double[m1][n2];

        for (int i = 0; i < m1; i++) {
            for (int j = 0; j < n2; j++) {
                for (int k = 0; k < n1; k++) {
                    output[i][j] += matrix1[i][k] * matrix2[k][j];
                }
            }
        }

        return output;
    }

    // For multiplying a 2d matrix with a 1d matrix
    // Code from https://introcs.cs.princeton.edu/java/22library/Matrix.java.html
    public static double[] matrixMultiply(double[] matrix1, double[][] matrix2)
    {
        int m = matrix2.length;
        int n = matrix2[0].length;

        // ensures that matrices are multiplicable
        if (matrix1.length != n) {
            System.out.println("Matrices: " + matrix1.toString() + " and " +matrix2.toString());
            throw new IllegalArgumentException("Matrices Inputted into Matrix.matrixMultiply cannot be multiplied");
        }

        double[] output = new double[m];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                output[i] += matrix2[i][j] * matrix1[j];
            }
        }

        return output;
    }

    public static double[] matrixMultiply(double[] matrix1, double[] matrix2) {
        if (matrix1.length != matrix2.length) {
            throw new IllegalArgumentException("Matrices Inputted into Matrix.matrixMultiply cannot be multiplied");
        }

        double[] output = new double[matrix1.length];

        for (int i = 0; i < matrix1.length; i++) {
            output[i] = matrix1[i] * matrix2[i];
        }

        return output;
    }


    // Multiplies a 1d matrix by a constant
    public static double[] constantMultiply(double[] matrix, double multiplier) {
        double[] c = new double[matrix.length];

        for (int i = 0; i < matrix.length; i++) {
            c[i] = matrix[i] * multiplier;
        }

        return c;
    }

    // Multiplies a 2d matrix by a constant
    public static double[][] constantMultiply(double[][] matrix, double multiplier) {
        double[][] c = new double[matrix.length][matrix[0].length];

        for (int i = 0; i < matrix.length; i++) {
            for (int n = 0; n < matrix[0].length; n++)
            {
                c[i][n] = matrix[i][n] * multiplier;
            }
        }

        return c;
    }

    // Divides a matrix by a constant
    public static double[] constantDivide(double[] matrix, double divisor) {
        double[] c = new double[matrix.length];

        for (int i = 0; i < matrix.length; i++) {
            c[i] = matrix[i] / divisor;
        }

        return c;
    }

    // Transposes matrix
    public static double[][] transpose(double[][] input) {
        int m = input.length;
        int n = input[0].length;

        double[][] output = new double[n][m];

        for (int i = 0; i < m; i++) {
            for (int k = 0; k < n; k++) {
                output[k][i] = input[i][k];
            }
        }
        return output;
    }

    public static double[][] transpose(double[] input) {
        int m = 1;
        int n = input.length;

        double[][] output = new double[n][m];

        for (int i = 0; i < n; n++) {
            output[m][i] = input[i];
        }

        return output;

    }

    //TODO: TESTING
    public static void main(String[] args) {
//        double[] a = new double[] {}
        double[][] b = new double[][] {{1.0, 2.0}, {3.0, 4.0}};
    }
}
