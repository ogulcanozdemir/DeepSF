
import sys
import numpy as np
import os
from keras.models import model_from_json
from keras.engine.topology import Layer
import theano.tensor as T
from keras import backend as K
from keras.constraints import maxnorm

from keras.models import Model
from keras.layers import Activation, Dense, Dropout, Flatten, Input, Merge, Convolution1D
from keras.layers.normalization import BatchNormalization

two_stream = False

def import_DLS2FSVM(filename, delimiter='\t', delimiter2=' ',comment='>',skiprows=0, start=0, end = 0,target_col = 1, dtype=np.float32):
    # Open a file
    file = open(filename, "r")
    #print "Name of the file: ", file.name
    if skiprows !=0:
       dataset = file.read().splitlines()[skiprows:]
    if skiprows ==0 and start ==0 and end !=0:
       dataset = file.read().splitlines()[0:end]
    if skiprows ==0 and start !=0:
       dataset = file.read().splitlines()[start:]
    if skiprows ==0 and start !=0 and end !=0:
       dataset = file.read().splitlines()[start:end]
    else:
       dataset = file.read().splitlines()
    #print dataset
    newdata = []
    for i in range(0,len(dataset)):
        line = dataset[i]
        if line[0] != comment:
           temp = line.split(delimiter,target_col)
           feature = temp[target_col]
           label = temp[0]
           if label == 'N':
               label = 0
           fea = feature.split(delimiter2)
           newline = []
           newline.append(int(label))
           for j in range(0,len(fea)):
               if fea[j].find(':') >0 :
                   (num,val) = fea[j].split(':')
                   newline.append(float(val))
            
           newdata.append(newline)
    data = np.array(newdata, dtype=dtype)
    file.close()
    return data

class K_max_pooling1d(Layer):
    def __init__(self,  ktop, **kwargs):
        self.ktop = ktop
        super(K_max_pooling1d, self).__init__(**kwargs)
    
    def get_output_shape_for(self, input_shape):
        return (input_shape[0],self.ktop,input_shape[2])
    
    def call(self,x,mask=None):
        output = x[T.arange(x.shape[0]).dimshuffle(0, "x", "x"),
              T.sort(T.argsort(x, axis=1)[:, -self.ktop:, :], axis=1),
              T.arange(x.shape[2]).dimshuffle("x", "x", 0)]
        return output
    
    def get_config(self):
        config = {'ktop': self.ktop}
        base_config = super(K_max_pooling1d, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

if __name__ == '__main__':

    #print len(sys.argv)
    if len(sys.argv) != 8:
            print 'please input the right parameters: list, model, weight, kmax'
            sys.exit(1)
    
    test_list=sys.argv[1] 
    model_file=sys.argv[2] 
    model_weight=sys.argv[3]
    feature_dir=sys.argv[4]
    pssm_dir=sys.argv[5]
    Resultsdir=sys.argv[6] 
    kmaxnode=int(sys.argv[7]) 
    if not os.path.exists(model_file):
         raise Exception("model file %s not exists!" % model_file)
    if not os.path.exists(model_weight):
         raise Exception("model file %s not exists!" % model_weight)
    print "Loading Model file ",model_file
    print "Loading Model weight ",model_weight
    json_file_model = open(model_file, 'r')
    loaded_model_json = json_file_model.read()
    json_file_model.close()    
    DLS2F_CNN = model_from_json(loaded_model_json, custom_objects={'K_max_pooling1d': K_max_pooling1d})        
    
    print "######## Loading existing weights ",model_weight;
    DLS2F_CNN.load_weights(model_weight)
    DLS2F_CNN.compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer="nadam")
    #if(two_stream):
    #    get_flatten_layer_output = K.function([DLS2F_CNN.layers[0].input, K.learning_phase()],[DLS2F_CNN.layers[-3].output]) # input to flatten layer
    #else:
    #    get_flatten_layer_output = K.function([DLS2F_CNN.layers[0].input, K.learning_phase()],[DLS2F_CNN.layers[-3].output]) # input to flatten layer
    print "Start loading data"
    Testlist_data_keys = dict()
    Testlist_targets_keys = dict()
    sequence_file=open(test_list,'r').readlines() 
    for i in xrange(len(sequence_file)):
        if sequence_file[i].find('Length') >0 :
            print "Skip line ",sequence_file[i]
            continue
        pdb_name = sequence_file[i].rstrip().split('\t')[0]
        #print "Processing ",pdb_name
        featurefile = feature_dir + '/' + pdb_name + '.fea_aa_ss_sa'
        pssmfile = pssm_dir + '/' + pdb_name + '.pssm_fea'
        if not os.path.isfile(featurefile):
                    print "feature file not exists: ",featurefile, " pass!"
                    #continue         
        
        if not os.path.isfile(pssmfile):
                    print "pssm feature file not exists: ",pssmfile, " pass!"
                    #continue         
        
        predict_out = Resultsdir+'/'+pdb_name+'.prediction'
        hidden_feature_out = Resultsdir+'/'+pdb_name+'.hidden_feature'
 
        if os.path.isfile(predict_out) and os.path.isfile(hidden_feature_out):
                continue
        
        featuredata = import_DLS2FSVM(featurefile)
        pssmdata = import_DLS2FSVM(pssmfile)
        pssm_fea = pssmdata[:,1:]
        
        fea_len = (featuredata.shape[1]-1)/(20+3+2)
        train_labels = featuredata[:,0]
        train_feature = featuredata[:,1:]
        train_feature_seq = train_feature.reshape(fea_len,25)
        train_feature_aa = train_feature_seq[:,0:20]
        train_feature_ss = train_feature_seq[:,20:23]
        train_feature_sa = train_feature_seq[:,23:25]
        train_feature_pssm = pssm_fea.reshape(fea_len,20)
        min_pssm=-8
        max_pssm=16
        
        train_feature_pssm_normalize = np.empty_like(train_feature_pssm)
        train_feature_pssm_normalize[:] = train_feature_pssm
        train_feature_pssm_normalize=(train_feature_pssm_normalize-min_pssm)/(max_pssm-min_pssm)
        featuredata_all_tmp = np.concatenate((train_feature_aa,train_feature_ss,train_feature_sa,train_feature_pssm_normalize), axis=1)
                    
        if fea_len <kmaxnode: # suppose k-max = 30
            fea_len = kmaxnode
            train_featuredata_all = np.zeros((kmaxnode,featuredata_all_tmp.shape[1]))
            train_featuredata_all[:featuredata_all_tmp.shape[0],:featuredata_all_tmp.shape[1]] = featuredata_all_tmp
        else:
            train_featuredata_all = featuredata_all_tmp
        
        #print "test_featuredata_all: ",train_featuredata_all.shape
        train_targets = np.zeros((train_labels.shape[0], 1195 ), dtype=int)
        for i in range(0, train_labels.shape[0]):
            train_targets[i][int(train_labels[i])] = 1
        
        train_featuredata_all=train_featuredata_all.reshape(1,train_featuredata_all.shape[0],train_featuredata_all.shape[1])
        if pdb_name in Testlist_data_keys:
            raise Exception("Duplicate pdb name %s in Test list " % pdb_name)
        else:
            Testlist_data_keys[pdb_name]=train_featuredata_all
        
        if pdb_name in Testlist_targets_keys:
            raise Exception("Duplicate pdb name %s in Test list " % pdb_name)
        else:
            Testlist_targets_keys[pdb_name]=train_targets
        
    sequence_file=open(test_list,'r').readlines() 
    for i in xrange(len(sequence_file)):
        if sequence_file[i].find('Length') >0 :
            #print "Skip line ",sequence_file[i]
            continue
        pdb_name = sequence_file[i].rstrip().split('\t')[0]
       
        predict_out = Resultsdir+'/'+pdb_name+'.prediction'
        hidden_feature_out = Resultsdir+'/'+pdb_name+'.hidden_feature'

        if os.path.isfile(predict_out) and os.path.isfile(hidden_feature_out):
                continue
        
        val_featuredata_all=Testlist_data_keys[pdb_name]
        val_targets=Testlist_targets_keys[pdb_name] 
        
        if(two_stream):
            predict_val= DLS2F_CNN.predict([val_featuredata_all[:,:,:20],val_featuredata_all[:,:,20:]])
            #hidden_feature= get_flatten_layer_output([[val_featuredata_all[:,:,:20],val_featuredata_all[:,:,20:]],1])[0] ## output in train mode = 1 https://keras.io/getting-started/faq/
            predict_out = Resultsdir+'/'+pdb_name+'.prediction'
            hidden_feature_out = Resultsdir+'/'+pdb_name+'.hidden_feature'
            np.savetxt(predict_out,predict_val,delimiter='\t')
            #np.savetxt(hidden_feature_out,hidden_feature,delimiter='\t')
        else:
            predict_val= DLS2F_CNN.predict([val_featuredata_all])
            #hidden_feature= get_flatten_layer_output([val_featuredata_all,1])[0] ## output in train mode = 1 https://keras.io/getting-started/faq/
            predict_out = Resultsdir+'/'+pdb_name+'.prediction'
            hidden_feature_out = Resultsdir+'/'+pdb_name+'.hidden_feature'
            np.savetxt(predict_out,predict_val,delimiter='\t')
            #np.savetxt(hidden_feature_out,hidden_feature,delimiter='\t')
        
