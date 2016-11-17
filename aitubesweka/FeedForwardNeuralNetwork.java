package aitubesweka;

import weka.classifiers.AbstractClassifier;
import weka.core.Instances;

public class FeedForwardNeuralNetwork extends AbstractClassifier {
	private Instances proInstances;
    private int classCount;
    private int attrCount;
	
	public void buildClassifier(Instances ins) throws Exception {
		// TODO Auto-generated method stub
		if (ins.checkForStringAttributes()) {
            throw new Exception("Can't handle string attributes!");
        }
        if (ins.classAttribute().isNumeric()) {
            throw new Exception("Naive Bayes: Class is numeric!");
        }
      
        proInstances = new Instances(ins,0);
        classCount = ins.numClasses();
        attrCount = ins.numAttributes() - 1;
        
        System.out.println(ins);
        System.out.println(proInstances);
        System.out.println(classCount);
        System.out.println(attrCount);
		
	}
	

	
}
