#Application Class 
#Holds Main project Neural Network

#NeuralNetworkClassiferProject

#Import from other Python Classes
#Configuration Class, import fixed variables
from Configuration import NUM_TRAINING_ITERATIONS, CONVERGENCE_THRESHOLD
#Formulas Class, import formulas
from Formulas import err
#ExtModels Class, import models 
from ExtModels import LayerFunction, cfile

#Variable
f = None
curr_point = 0
target = []
attrs = []
total_runs = 0
data = None
num_incorrect = 0
prev_sample_err = 0
curr_sample_err = 0

#Parse Data

def parse_data(fname):
    #Declare global variables
    global curr_point
    global total_runs
    global target
    global attrs
    global num_incorrect
    global prev_sample_err
    global curr_sample_err
    global data
    global f
    global PosionCounter
    global EdableCounter

    #Variable holders
    curr_point = 0
    total_runs = 0
    target = []
    attrs = []
    num_incorrect = 0
    prev_sample_err = 0
    curr_sample_err = 0
    #Counter for number of Posion Mushrooms
    PosionCounter = 0
    #Counter for number of Edable Mushrooms
    EdableCounter = 0

    #Select the error text file
    data_file = 'err.txt'

    #Select Training text file 
    if fname == 'mushroom_train.txt':
        #Output testing error from Neural Network
        data_file = 'Training_error.txt'
    #Select Testing text file
    elif fname == 'mushroom_test.txt':
        #Select Testing error from Neural Network
        data_file = 'Testing_error.txt'

    #Clear the data file
    open(data_file, 'w+').close()

     # open the data file for logging
    data = cfile(data_file, 'w')

    #Declare f as Open File
    f = open(fname, 'r').readlines()

    #For each row in file, do;
    for row in f:
     row = [x.strip() for x in row.split(',')]
     row = [int(num) for num in row]
     target.append(int(row[0]))
     attrs.append(row[1:])

#Main Applicaiton
#Houses the Neural Network
if __name__ == '__main__':
    #Console Update
    print ("Parsing the training dataset...")
    # parse the training dataset and store its information into globals
    parse_data('mushroom_train.txt')

    # set up the layers to be used
    x = LayerFunction(6, attrs[curr_point], 1)
    y = LayerFunction(1, x.layer_out, 2)

    #Console Update
    print ("Begining training the neural network:")

    #Neural Network Start
    #Primary Neural Network Program
    #While total_runs is less than the total number of operations
    while total_runs < NUM_TRAINING_ITERATIONS:
        #Set New Input into neuron
        x.input_vals = attrs[curr_point]

        # set up the first layer and evaluate it
        x.input_vals = attrs[curr_point]
        x.eval()

        # set up the second layer and evaluate it
        y.input_vals = x.layer_out
        y.eval()

        # backpropogate learning
        y.backprop(target[curr_point])
        x.backprop(y)

        # get the current error
        curr_err = err(y.layer_out[0], target[curr_point])

        #Perform Rounding, Sort Neuron Output as binary One or Zero
        if y.layer_out[0] >= 0.5:
            #Set Neuron as On
            temp = 1
        else:
            #Set Neuron as Off
            temp = 0


        # increment the number incorrect if its wrong
        if(temp != target[curr_point]):
            num_incorrect += 1

        # Increment Counter for checking number of Posion Mushrooms
        if (target[curr_point] == 1):
            PosionCounter += 1

        #Increment Counter for Checking number of Edable Mushrooms
        if (target[curr_point] == 0):
            EdableCounter += 1

                # check to see if we have converged
        if total_runs % 100 == 0:
            prev_sample_err = curr_sample_err
            curr_sample_err = curr_err
            if abs(prev_sample_err - curr_sample_err) < CONVERGENCE_THRESHOLD:
                print ("Data has converged at the " + str(total_runs) + "th run.")
                break;

        #Print information about the current iteration
        print ("Current iteration: " + str(total_runs))
        print ("Current error: ") + str(curr_err) + "\n"
        print ("Neuron Output:") + str(y.layer_out[0])
        #For display purpose
        if (target[curr_point] == 1):
            print ("Mushroom Status: Posionious")
        else:
            print("Mushroom Status: Edable")

        #print ("Neuron Status [1]Posion [0]Edable:") + str(target[curr_point])
        data.w(curr_err)

        # iterate
        total_runs += 1
        curr_point += 1

        if curr_point >= len(f):
            curr_point = 0

    # close the file
    data.close()

    #Print Details of Training
    print("Training Completed, Statistics;") 
    print("Number of Posionious Mushrooms: ") + str(PosionCounter)
    print("Number of Edable Mushrooms: ") + str(EdableCounter)
    print("Number of Errors: ") + str(num_incorrect)

    #User Updated Message
    print ("Neural network is done training! Hit enter to test it.")
    print ("Error percentage on testing set: " + str(float(num_incorrect)/NUM_TRAINING_ITERATIONS))

    #Input...
    raw_input()

    print ("Begin testing the neural network:")
    # parse the testing data and store its information into globals
    parse_data('mushroom_test.txt')

    # iterate through to test the neural network
    while curr_point < len(f):
        # set the new input values
        x.input_vals = attrs[curr_point]

        # set up the first layer and evaluate it
        x.input_vals = attrs[curr_point]
        x.eval()

        # set up the second layer and evaluate it
        y.input_vals = x.layer_out
        y.eval()

        # backpropogate
        y.backprop(target[curr_point])
        x.backprop(y)

        # get the current error
        curr_err = err(y.layer_out[0], target[curr_point])

        # round up and down to check err
        if y.layer_out[0] >= 0.5:
            temp = 1
        else:
            temp = 0

        # Increment Counter for checking number of Posion Mushrooms
        if (target[curr_point] == 1):
            PosionCounter += 1

        #Increment Counter for Checking number of Edable Mushrooms
        if (target[curr_point] == 0):
            EdableCounter += 1

        # increment the number incorrect if its wrong
        if(temp != target[curr_point]):
            num_incorrect += 1

        # check to see if we have converged
        if total_runs % 100 == 0:
            prev_sample_err = curr_sample_err
            curr_sample_err = curr_err
            if abs(prev_sample_err - curr_sample_err) < CONVERGENCE_THRESHOLD:
                print ("Data has converged at the " + str(total_runs) + "th run.")
                break;

        #Print information about the current iteration
        print ("Current iteration: " + str(total_runs))
        print ("Current error: ") + str(curr_err) + "\n"
        print ("Neuron Output:") + str(y.layer_out[0])
        #For display purpose
        if (target[curr_point] == 1):
            print ("Mushroom Status: Posionious")
        else:
            print("Mushroom Status: Edable")

        data.w(curr_err)

        # iterate
        total_runs += 1
        curr_point += 1

    data.close()

    #Finial Output
    print ("Testing done! Check out the generated output files ('testing_err.txt' and 'training_err.txt')")
    print("Number of Posionious Mushrooms: ") + str(PosionCounter)
    print("Number of Edable Mushrooms: ") + str(EdableCounter)
    print("Number of Errors: ") + str(num_incorrect)
    print ("Error percentage on training set: " + str(float(num_incorrect)/len(f)))