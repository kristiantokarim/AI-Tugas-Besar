package aitubesweka;


public class NeuralNetwork {
	private Node[] firstLayer;
	private Node[] outputLayer;
	private double[] input;
	private int numberOfInput;
	private int numberOfFirstLayer;
	private int numberOfOutputLayer;
	
	public NeuralNetwork(int numOfInput, int numOfFirstLayer, int numOfOutputLayer) {
		
		numberOfInput = numOfInput;
		numberOfFirstLayer = numOfFirstLayer;
		numberOfOutputLayer = numOfOutputLayer;
		
		input = new double[numberOfInput];
		firstLayer = new Node[numberOfFirstLayer];
		outputLayer =  new Node[numberOfOutputLayer];
		
		for (int i =0; i < numberOfFirstLayer; i++) {
			firstLayer[i] = new Node(numberOfInput);
		}
		
		for (int i = 0; i < numberOfOutputLayer; i++) {
			outputLayer[i] = new Node(numberOfFirstLayer);
		}
		
	}
	
	public void setInput(double[] newInput) {
		if (input.length == numberOfInput) {
			for (int i = 0; i< numberOfInput; i++) {
				input[i] = newInput[i];
			}
		}
		else {
			System.out.println("Number Of Attribut Instance Not Match");
		}
	}
	
	public double[] countFirstLayerOutput() {
		double[] result = new double[numberOfFirstLayer];
		for (int i =0; i < numberOfFirstLayer; i++) {
			result[i] = firstLayer[i].countOutput(input);
 		}
		return result;
	}
	
	public double[] countOuputLayerOutput() {
		double[] result = new double[numberOfOutputLayer];
		for (int i =0; i < numberOfOutputLayer; i++) {
			result[i] = outputLayer[i].countOutput(input);
 		}
		return result;
	}
	
	public double[] countOutput() {
		double[] result;
		result = countFirstLayerOutput();
		result = countOuputLayerOutput();
		return result;
	}
	
	public double[] countOutputLayerError(double[] target) {
		double[] error = new double[numberOfOutputLayer];
		if (numberOfOutputLayer == target.length) {
			double[] result;
			result = countOutput();
			for (int i = 0; i < target.length; i++) {
				error[i] = target[i] - result[i];
			}
		}
		else {
			System.out.println("Target Array Size Not Match");
		}
		return error;
		
	}
}
