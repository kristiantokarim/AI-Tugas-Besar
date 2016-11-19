package wekatubes;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author programming
 */
public class naiveBayes extends AbstractClassifier {
    public Instances proInstances;
    public int classCount;
    public int attrCount;
    public int [][][] freqTable; 
    public int [] classFreqTable;
    public double [][][] probTable;
    public double [] classProbTable;
    
    public void generateFreqTable(Instances ins) {
        probTable = new double[classCount][attrCount][0];
        freqTable = new int[classCount][attrCount][0];
        for (int i = 0 ; i < attrCount ; i++) {
            Attribute proAttr = ins.attribute(i);
            if (proAttr.isNominal()) {
                for (int j = 0 ; j < classCount; j++) {
                    probTable[j][i] = new double[proAttr.numValues()];
                    freqTable[j][i] = new int[proAttr.numValues()];
                }
            }
            else {
                for (int j = 0 ; j < classCount; j++) {
                    probTable[j][i] = new double[1];
                    freqTable[j][i] = new int[1];
                }
            }
        }
        
        classFreqTable = new int[classCount];
        classProbTable = new double[classCount];
        for (int i = 0 ; i < ins.numInstances() ; i++) {
            Instance proIns = ins.get(i);
            if (!proIns.classIsMissing()) {
                for (int j = 0 ; j < attrCount ; j++) {
                    Attribute proAttr = ins.attribute(j);
                    freqTable[(int)proIns.classValue()][j][(int)proIns.value(proAttr)]++;
                }
            }
            classFreqTable[(int)proIns.classValue()]++;
        }
    }
    
    public void generateProbTable (Instances ins) {
        for (int i = 0 ; i < attrCount ; i++) {
            Attribute proAttr = ins.attribute(i);
            if (proAttr.isNominal()) {
                for (int j = 0 ; j < classCount ; j++) {
                    int sum = Utils.sum(freqTable[j][i]);
                    for (int k = 0 ; k < proAttr.numValues() ; k++) {
                        probTable[j][i][k] = ((double)freqTable[j][i][k])/((double)sum);
                    }
                }
            }
        } 
        
        int sum = Utils.sum(classFreqTable);
        for (int i = 0 ; i < classCount ; i++) {
            classProbTable[i] = ((double)classFreqTable[i])/((double)sum);
        }
    }
    @Override
    public void buildClassifier(Instances ins) throws Exception {
        
        getCapabilities().testWithFail(ins);

        proInstances = new Instances(ins,0);
        classCount = ins.numClasses();
        attrCount = ins.numAttributes() - 1;
        generateFreqTable(ins);
        generateProbTable(ins);
    }
    @Override
    public double[] distributionForInstance(Instance ins) {
        double[] res = new double[classCount];
        double max = -1;
        int imax = -1;
        for (int i =0;i<classCount;i++){
            double result = 1;
            for (int j=0; j<attrCount;j++){
                String val = ins.stringValue(j);
                int valIdx = ins.attribute(j).indexOfValue(val);
                result *= probTable[i][j][valIdx]; 
            }
            res[i] = classProbTable[i]*result;
            if (res[i] > max) {
                max = res[i];
                imax = i;
            }
            System.out.format("%01.20f\t",res[i]);
        }
        System.out.println(ins.classAttribute().value(imax));
        return res;
    }
    
    public String resultString(Evaluation ev) throws Exception {
        StringBuilder out = new StringBuilder();
        out.append(ev.toSummaryString());
        out.append(ev.toClassDetailsString());
        out.append(ev.toMatrixString());
        return out.toString();
    }
    
}
