package aitubesweka;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NumericToNominal;

public class Test {
	
	
	public static Instances filterNumericToNominal(Instances dataset, String[] opts) throws Exception {
		NumericToNominal numToNom = new NumericToNominal();
		numToNom.setOptions(opts);
		numToNom.setInputFormat(dataset);
		Instances nomData = Filter.useFilter(dataset, numToNom);
		return nomData;
	}
	
	public static Instances filterDiscretize(Instances dataset, String[] opts) throws Exception {
		Discretize discretize = new Discretize();
		discretize.setOptions(opts);
		discretize.setInputFormat(dataset);
		Instances numData = Filter.useFilter(dataset, discretize);
		return numData;
	}

	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		//Read Data Set and assign to dataset
		DataSource source = new DataSource("/home/cmrudi/weka-3-8-0/data/iris.arff");
		Instances dataset = source.getDataSet();
				
		//System.out.println(dataset);
		
		dataset.setClassIndex(dataset.numAttributes()-1);
		FeedForwardNeuralNetwork FFNN = new FeedForwardNeuralNetwork();
		FFNN.buildClassifier(dataset);
		
	}

}