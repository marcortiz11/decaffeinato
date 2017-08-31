import numpy as np
import re
import click
import glob, os
import matplotlib.pyplot as mplot
mplot.switch_backend("TkAgg")
from matplotlib import pylab as plt
import matplotlib
import operator
import ntpath


@click.command()
@click.argument('files', nargs=-1, type=click.Path(exists=True))
def main(files):
    if not files:
        print 'no args found'
        print '\n\rloading all files with .log extension from current directory'
        os.chdir(".")
        files = glob.glob("*.log")
    
    colors = ['r','b','g','k']
    linestyle = ['.-','.-','.-','']
    
    plt.figure(1)
    plt.style.use('ggplot')
    
    training = plt.subplot(211)
    training.set_ylabel('Training error')
    
    test = plt.subplot(212)
    test.set_xlabel('Epoch')
    test.set_ylabel('Test error (%)')
    test.set_ylim([0,5])
    
    for i, log_file in enumerate(files):
        
        loss_iterations, losses, accuracy_iterations, accuracies, accuracies_iteration_checkpoints_ind, fileName = parse_log(log_file)
        
        #Imprimim training loss
        plt.subplot(211)
        training.plot(loss_iterations,losses, colors[i%len(colors)] + linestyle[i%len(linestyle)], label = fileName)
        plt.legend(loc='upper right') 
        plt.yscale('log')
        
        plt.subplot(212)
        test.plot(accuracy_iterations,accuracies,colors[i%len(colors)] + linestyle[i%len(linestyle)],label = fileName)
        plt.yscale('linear')
        plt.legend(loc='upper right') 
    
    plt.show()
    

def parse_log(log_file):
    with open(log_file, 'r') as log_file2:
        log = log_file2.read()

    loss_pattern = r"Iteration (?P<iter_num>\d+) .*, loss = (?P<loss_val>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
    losses = []
    loss_iterations = []
    
    epoch = 0
    
    fileName= os.path.basename(log_file)
    for r in re.findall(loss_pattern, log):
        loss_iterations.append(float(epoch))
        losses.append(float(r[1]))
        epoch = epoch+1

    losses = np.array(losses)
    loss_iterations = np.array(loss_iterations)
    
    accuracy_pattern = r"Test net output #0: accuracy = (?P<accuracy>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
    accuracies = []
    accuracy_iterations = []
    accuracies_iteration_checkpoints_ind = []

    epoch = 0
    
    for r in re.findall(accuracy_pattern, log):
        iteration = int(epoch)
        accuracy = (1-float(r[1])) * 100

        if iteration % 10000 == 0 and iteration > 0:
            accuracies_iteration_checkpoints_ind.append(len(accuracy_iterations))

        accuracy_iterations.append(iteration)
        accuracies.append(accuracy)
        epoch = epoch + 1

    accuracy_iterations = np.array(accuracy_iterations)
    accuracies = np.array(accuracies)
	
    return loss_iterations, losses, accuracy_iterations, accuracies, accuracies_iteration_checkpoints_ind, fileName


if __name__ == '__main__':
    main()
