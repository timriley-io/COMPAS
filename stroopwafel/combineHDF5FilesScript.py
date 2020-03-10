import numpy as np
import h5py as h5
import argparse

#-- Get command line arguments
parser = argparse.ArgumentParser(description='''Combine two COMPASOutput.h5 files''',
								epilog='''
								''')

parser.add_argument("--input-file1", type=str, default='COMPASOutput1.h5', help="Input file e.g. COMPASOutput1.h5")
parser.add_argument("--input-file2", type=str, default='COMPASOutput2.h5', help="Input file e.g. COMPASOutput2.h5")
parser.add_argument("--output-file", type=str, default='COMPASOutputCombined.h5', help="Output file e.g. COMPASOutputCombined.h5")
args = parser.parse_args()

#-- Input files to be combined
h5File1 = h5.File(args.input_file1, 'r')
h5File2 = h5.File(args.input_file2, 'r')

groups = list(h5File1.keys())

#-- Output file
h5File3 = h5.File(args.output_file, 'w')

for group in groups:
    
    #-- Create a group in the output
    #print(group)
    
    h5File3.create_group(group)
    
    datasets = list(h5File1[group].keys())
    
    #-- Loop through datasets, combine and write
    for dataset in datasets:
        
        #print(dataset)
        
        #-- Concatenate datasets
        combined_dataset = np.concatenate([h5File1[group][dataset],
                                           h5File2[group][dataset]])
        
        #-- Save datasets
        h5File3[group][dataset] = combined_dataset

#-- Close all the files
h5File1.close()
h5File2.close()
h5File3.close()