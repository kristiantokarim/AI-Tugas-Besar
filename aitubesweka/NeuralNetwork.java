/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ffn;

import java.io.Serializable;

public class NeuralNetwork implements Serializable {
	private Node[] firstLayer;
	private Node[] outputLayer;
	private double[] input;
	private double[] target;
	private int numberOfInput;
	private int numberOfFirstLayer;
	private int numberOfOutputLayer;
	
	/*
	 * 
	 * Constructor untuk NeuralNetwork
	 * Input: Jumlah input, jumlah node pada hidden layer dan jumlah node pada Output Layer
	 * Proses : Mengkonstruksi neural network dengan kelas node dan memberikan initial weight 1
	 * 
	 */
	
	
	public NeuralNetwork(int numOfInput, int numOfFirstLayer, int numOfOutputLayer) {
		
		numberOfInput = numOfInput;
		numberOfFirstLayer = numOfFirstLayer;
		numberOfOutputLayer = numOfOutputLayer;
		
		input = new double[numberOfInput];
		firstLayer = new Node[numberOfFirstLayer];
		outputLayer =  new Node[numberOfOutputLayer];
		target = new double[numberOfOutputLayer];
		
		for (int i =0; i < numberOfFirstLayer; i++) {
			firstLayer[i] = new Node(numberOfInput);
		}
		
		for (int i = 0; i < numberOfOutputLayer; i++) {
			outputLayer[i] = new Node(numberOfFirstLayer);
		}
		
	}
	
	/*
	 * 
	 * Memasukkan Input ke Neural Network, Input adalah nilai-nilai sebuah instance yang 
	 * disimpan pada array of double
	 * 
	 */
	
	public void setInput(double[] newInput) {
		if (newInput.length == numberOfInput) {
			for (int i = 0; i< numberOfInput; i++) {
				input[i] = newInput[i];
				
			}
		}
		else {
			System.out.println("On NeuralNetwork class, setInput method, Number Of Attribut Instance Not Match");
			System.out.println("NumberOfInput = " + numberOfInput + " and Array of Input length is = " + newInput.length);
		}
	}
	
	/*
	 * 
	 * Setter untuk target output, disimpan dalam array of double, hanya target output untuk sebuah instance
	 * 
	 */
	
	public void setTarget(double[] newTarget) {
		if (newTarget.length == numberOfOutputLayer) {
			for (int i = 0; i < numberOfOutputLayer; i++) {
				target[i] = newTarget[i];
				outputLayer[i].setTarget(newTarget[i]);
			}
		}else {
			System.out.println("Target Array Size Not Match");
		}
	}
	
	/*
	 * 
	 * Meghitung output untuk hidden layer pertama
	 * Requirement : input telah dimasukkan melalui void setInput()
	 * 
	 */
	
	public double[] countFirstLayerOutput() {
		double[] result = new double[numberOfFirstLayer];
		for (int i =0; i < numberOfFirstLayer; i++) {
			result[i] = firstLayer[i].countOutput(input);
 		}
		return result;
	}
	
	/*
	 * 
	 * Menghitung output dari neural network
	 * Requirement : input telah dimasukkan melalui void setInput() dan output untuk hidden layer sudah dihitung
	 * Input : input layer adalah input khusus untuk layer itu, bukan input awal
	 * 
	 */
	
	public double[] countOuputLayerOutput(double[] inputLayer) {
		double[] result = new double[numberOfOutputLayer];
		for (int i =0; i < numberOfOutputLayer; i++) {
			result[i] = outputLayer[i].countOutput(inputLayer);
 		}
		return result;
	}
	
	/*
	 * 
	 * Menghitung output akhir berdasarkan input awal
	 * Requirement : input telah dimasukkan melalui void setInput()
	 * 
	 */
	
	public double[] countOutput() {
		double[] result;
		result = countFirstLayerOutput();
		result = countOuputLayerOutput(result);
		return result;
	}
	
	/*
	 * 
	 * Menghitung error pada output layer
	 * Requirement : output neural network keseluruhan telah dihitung
	 * 
	 */
	
	public double[] countOutputLayerError() {
		double[] error = new double[numberOfOutputLayer];
		for(int i = 0; i < numberOfOutputLayer; i++) {
			error[i] = outputLayer[i].countErrorOutputLayer();
		}
		return error;
	}
	
	public double[] countFirstLayerError() {
		double[] error = new double[numberOfFirstLayer];
		double weightxerror;
		for(int j = 0; j < numberOfFirstLayer; j++) {
			weightxerror = 0;
			for (int i = 0; i < numberOfOutputLayer; i++) {
				weightxerror = weightxerror + (outputLayer[i].getError()*outputLayer[i].getWeight(j));
			}
			
			error[j] = firstLayer[j].countErrorFirstLayer(weightxerror);
		}
		return error;
	}
	
	/*
	 * 
	 * Setter untuk learning rate
	 * 
	 */
	
	public void setLearningRate(double value) {
		for(int i = 0; i < numberOfFirstLayer; i++) {
			firstLayer[i].setLearningRate(value);
		}
		for (int i = 0; i < numberOfOutputLayer; i++) {
			outputLayer[i].setLearningRate(value);
		}
	}
	
	/*
	 * 
	 * Mengupdate weight output layer
	 * 
	 */
	
	public void updateOutputLayerWeight() {
		double[] firstLayerOutput = new double[numberOfFirstLayer];
		for (int i =0; i < numberOfFirstLayer; i++) {
			firstLayerOutput[i] = firstLayer[i].getOutput();
		}
		for(int i =0; i < outputLayer.length; i++) {
			outputLayer[i].updateWeight(firstLayerOutput);
		}
	}
	
	/*
	 * 
	 * Mengupdate weight first layer
	 * 
	 */
	
	public void updateFirstLayerWeight() {
		for (int i =0; i < numberOfFirstLayer; i++) {
			firstLayer[i].updateWeight(input);
		}
	}
	
	/*
	 * 
	 * Mencetak weight seluruh node
	 * 
	 */
	
	public void printAllWeight() {
		System.out.println("First Layer Node");
		for(int i =0; i < numberOfFirstLayer; i++) {
			firstLayer[i].printWeight();
		}
		System.out.println("Output Layer Node");
		for (int i = 0; i < numberOfOutputLayer; i++) {
			outputLayer[i].printWeight();
		}
	}
}
