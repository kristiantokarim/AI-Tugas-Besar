package aitubesweka;

public class Node {
	
	private static final int MaxNumberOfInput = 100;
	private static final double InitialWeight = 1;
	private double[] weight;
	
	/*
	 * Input : Jumlah input menuju node tersebut
	 * Proses : Inisialisasi weight setiap edge menuju node tersebut dengan InitialWeight 
	 * 
	 */
	public Node(int numberOfInput) {
		weight = new double[numberOfInput];
		for (int i = 0; i < numberOfInput; i++) {
			weight[i]= InitialWeight;
		}
	}
	
	/*
	 * Input : Array of Double yang berisi input menuju Node
	 * Proses : Menghitung output yang diberikan node terseut menggunakan array input dan array weight
	 *
	 */
	
	public double countOutput(double[] input) {
		double sum = 0;
		for (int i =0; i < input.length; i++) {
			sum = sum + input[i] * weight[i];
		}
		return sum;
	}
	
	/*
	 * 
	 * Input : Index array weight, dimulai dari nol
	 * 
	 */
	
	public double getWeight(int inputIndex) {
		return weight[inputIndex];
	}
	
	/*
	 * 
	 * Input : Index yang ingin diubah weightnya dan inputValue adalah weight baru
	 * Proses : Mengganti weight lama dengan weight baru
	 * 
	 */
	
	public void setWeight(int inputIndex, double inputValue) {
		weight[inputIndex] = inputValue;
	}
}
