package aitubesweka;


import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.Instance;
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
        //System.out.println(proInstances);
        //System.out.println(classCount);
        //System.out.println(attrCount);
        Attribute proAttr = ins.attribute(1);
        
        System.out.println("=====");
        Instance proIns;
        double[] instance;
        double[] check;
        double[] input = new double[attrCount];
        double[] target = new double[1];
        NeuralNetwork nn = new NeuralNetwork(attrCount,2,1);
        for (int i =0; i < ins.numInstances(); i++) {
        	proIns = ins.get(i);
        	instance = proIns.toDoubleArray();
        	for (int j = 0; j < instance.length - 1; j++) {
        		input[j] = instance[j];
        	}
        	target[0] = instance[attrCount];
        	
        	//Initialize Input
        	nn.setInput(input);
            nn.setTarget(target);
            //Feed Forward
            nn.countOutput();
            //Back Propagation
            nn.countOutputLayerError();
            nn.setLearningRate(1);
            nn.updateOutputLayerWeight();
            nn.countFirstLayerError();
            nn.updateFirstLayerWeight();
            nn.printAllWeight();
        	
        }
	}
	
	
	
}
