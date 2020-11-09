import data_reader as dr
import numpy as np
import math
images, labels, test_images, test_labels =  dr.images, dr.labels, dr.test_images, dr.test_labels

np.random.seed(1)

def tanh(x):
    return np.tan(x)
def tanh_deriv(x):
    return 1 - (x**2)
def softmax(x):
    
    tmp = np.exp(x) 
    #print(tmp)
    return tmp / np.sum(tmp, axis=1, keepdims=True)

def get_image_section(layer, row_from, row_to, col_from, col_to):
    sub_section = layer[:,row_from:row_to, col_from:col_to]
    return sub_section.reshape(-1, 1, row_to-row_from, col_to-col_from)


input_rows, input_cols = 28, 28

kernel_rows, kernel_cols = 3, 3
num_kernels = 16
kernels = 0.02 * np.random.random((kernel_rows*kernel_cols, num_kernels)) - 0.01


hidden_size = ((input_rows-kernel_rows) * (input_cols-kernel_cols)) * num_kernels
datapoints_count = len(images)
pixs_per_img, batch_size, iterations, labels_size = test_images.shape[1], 128, 200, 10
alpha = 2

w_1_2 = 0.2 * np.random.random((hidden_size, labels_size)) - 0.1


import stat_rec 
stat = stat_rec


for j in range(iterations):
    
    corrects, error = (0, 0.0)
    for i in range(int(len(images)/batch_size)):
        
        batch_start, batch_end =   ((batch_size*i), (batch_size*(i+1)))  

        layer_0 = images[batch_start:batch_end]

        layer_0 = layer_0.reshape(layer_0.shape[0], 28,28)
        layer_0.shape
        sects = list()

        for row_start in range(layer_0.shape[1]-kernel_rows):
            for col_start in range(layer_0.shape[2]-kernel_cols):
                sect = get_image_section(layer_0, row_start, row_start + kernel_rows, col_start, col_start+kernel_cols)
                sects.append(sect)

        expanded_input = np.concatenate(sects, axis=1)
        es = expanded_input.shape
        flattened_input = expanded_input.reshape(es[0]*es[1], -1)
    
        kernel_output = flattened_input.dot(kernels)



        layer_1 = tanh(kernel_output.reshape(es[0], -1))
        
        
        dropout_mask = np.random.randint(2, size=layer_1.shape)
        layer_1 *= dropout_mask * 2
        
        layer_2 = softmax(np.dot(layer_1, w_1_2))
        
       # error +=  np.sum( (labels[batch_start:batch_end] - layer_2 ) ** 2 ) 

        for k in range(batch_size):

            corrects += int( np.argmax(labels[batch_start + k : batch_start+k+1]) == np.argmax(layer_2[k:k+1]) )
            stat.add_results(np.argmax(layer_2[k:k+1]), np.argmax(labels[batch_start + k : batch_start+k+1]))
            
            
        delta_2 = ( (labels[batch_start:batch_end] - layer_2  )/ (batch_size * layer_2.shape[0] ) )
        delta_1 = np.dot(delta_2, w_1_2.T) * tanh_deriv(layer_1)
            
        delta_1 *= dropout_mask
            
        w_1_2 += alpha * np.dot(layer_1.T, delta_2)

        l1d_reshape = delta_1.reshape(kernel_output.shape)
        k_update = flattened_input.T.dot(l1d_reshape)

        kernels -= alpha * k_update
            
        
        
    if j % 5 == 0:
        
        t_corrects =  0.0 
        for k in range(len(test_images)):
            
            layer_0 = test_images[k:k+1]

            layer_0 = layer_0.reshape(layer_0.shape[0], 28,28)
            layer_0.shape

            sects = list()
            for row_start in range(layer_0.shape[1]-kernel_rows):
                for col_start in range(layer_0.shape[2]-kernel_cols):
                    sect = get_image_section(layer_0, row_start, row_start + kernel_rows, col_start, col_start+kernel_cols)
                    sects.append(sect)

            expanded_input = np.concatenate(sects, axis=1)
            es = expanded_input.shape
            flattened_input = expanded_input.reshape(es[0]*es[1], -1)

            kernel_output = flattened_input.dot(kernels)



            layer_1 = tanh(kernel_output.reshape(es[0], -1))
            
            layer_2 = np.dot(layer_1, w_1_2)
            
            t_corrects += int(   np.argmax(test_labels[k:k+1])  == np.argmax(layer_2))
            
        
        stat.add_acc(t_corrects/len(test_images))
        print("i-", j , " Train_acc: ", corrects/datapoints_count, " Test_acc: ", t_corrects/len(test_images) )
    
stat.stat_show()