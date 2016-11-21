package wekatubes;

import java.util.Enumeration;
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
    public int classCount;
    public int attrCount;
    public int [][][] freqTable; 
    public int [] classFreqTable;
    public double [][][] probTable;
    public double [] classProbTable;
    
    public void generateFreqTable(Instances ins) {
        probTable = new double[classCount][attrCount][0];
        freqTable = new int[classCount][attrCount][0];
        
        Enumeration enums = ins.enumerateAttributes();
        int i =0;
        while (enums.hasMoreElements()) {
          Attribute proAttr = (Attribute) enums.nextElement();
          if (proAttr.isNominal()) {
            for (int j = 0; j < ins.numClasses(); j++) {
                probTable[j][i] = new double[proAttr.numValues()];
                freqTable[j][i] = new int[proAttr.numValues()];
            }
          } else {
            for (int j = 0 ; j < classCount; j++) {
                probTable[j][i] = new double[1];
                freqTable[j][i] = new int[1];
            }
            }
          i++;
        }
        
        classFreqTable = new int[classCount];
        classProbTable = new double[classCount];
        for (int k = 0 ; k < ins.numInstances() ; k++) {
            Instance proIns = ins.get(k);
            if (!proIns.classIsMissing()) {
                Enumeration enumAttr = ins.enumerateAttributes();
                i =0;
                while (enumAttr.hasMoreElements()) {
                    Attribute proAttr = (Attribute) enumAttr.nextElement();
                    freqTable[(int)proIns.classValue()][i][(int)proIns.value(proAttr)]++;
                    i++;
                }
            }
            classFreqTable[(int)proIns.classValue()]++;
        }
    }
    
    public void generateProbTable (Instances ins) {
        Enumeration enumAttr = ins.enumerateAttributes();
        int i =0;
        while(enumAttr.hasMoreElements()) {
            Attribute proAttr = (Attribute) enumAttr.nextElement();
            if (proAttr.isNominal()) {
                for (int j = 0 ; j < classCount ; j++) {
                    int sum = Utils.sum(freqTable[j][i]);
                    for (int k = 0 ; k < proAttr.numValues() ; k++) {
                        probTable[j][i][k] = ((double)freqTable[j][i][k]+1)/((double)sum + proAttr.numValues());
                    }
                }
            }
            i++;
        } 
        
        int sum = Utils.sum(classFreqTable);
        for (i = 0 ; i < classCount ; i++) {
            classProbTable[i] = ((double)classFreqTable[i] + 1)/((double)sum + (double)classCount);
        }
    }
    @Override
    public void buildClassifier(Instances ins) throws Exception {
        
        getCapabilities().testWithFail(ins);

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
            Enumeration enumAttr = ins.enumerateAttributes();
            int j =0;
            while (enumAttr.hasMoreElements()){
                result *= probTable[i][j][(int)ins.value((Attribute) enumAttr.nextElement())]; 
                j++;
            }
            res[i] = classProbTable[i]*result;
            if (res[i] > max) {
                max = res[i];
                imax = i;
            }
            System.out.format("%01.20f\t",res[i]);
        }
        Utils.normalize(res);
        System.out.println(ins.classAttribute().value(imax));
        return res;
    }
    
    
    
}
