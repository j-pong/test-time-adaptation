import os
import glob
import argparse

import numpy as np

parser = argparse.ArgumentParser(description='Basic')

parser.add_argument('--root_path', type=str, default="", required=True)
parser.add_argument('--seeds', nargs='+', default=[1, 2, 3, 4])
parser.add_argument('--setting', type=str, default=f"cs")
parser.add_argument('--dataset', type=str, default="imagenet_c")
parser.add_argument('--method', type=str, default="cmwa")
parser.add_argument('--models', nargs='+', default=["resnet50", "vit_b_16", "swin_b"])
parser.add_argument('--tag', type=str, default="")
parser.add_argument('--deltas', nargs='+', default=["0.0", "0.01", "0.1", "1.0", "5.0"])
args = parser.parse_args()

path_root = os.path.join(args.root_path, "output")
seeds = args.seeds
setting = args.setting
dataset = args.dataset
method = args.method
models = args.models
tag = args.tag
deltas = args.deltas

if setting == "mixed_domains":
    for m in models:
        multiple_error = []
        for s in seeds:
            errs = []
            if tag != "":
                res_tag = f"{setting}/{m}/{deltas[0]}_{method}_{tag}_seed{s}"
            else:
                res_tag = f"{setting}/{m}/{deltas[0]}_{method}_seed{s}"
            root = os.path.join(path_root, res_tag)

            os.chdir(root)
            for j, file in enumerate(glob.glob(f"{root}/*/*.txt", recursive=True)):
                # print(file)
                with open(file, "r") as f:
                    for l in f.readlines():
                        l = l.replace("\n", "")
                        if "error:" in l and not ("mean error" in l):
                            errs.append(float(l.split("error:")[-1].strip().strip('%')))
                        elif "Average error across all domains:" in l:
                            errs.append(float(l.split(":")[-1].strip().strip('%')))
            # last_err = l.split()[-1].strip('%')
            # errs.append(float(last_err))
            multiple_error.append(errs)
        multiple_error = np.array(multiple_error)
        
        if len(multiple_error) > 1:
            std = multiple_error.std(0)
            mean = multiple_error.mean(0)
    
        print(f"{m} " + " ".join(format(x, ".2f") for x in mean) + u"\u00B1" + f"{std[-1]:.2f}") 

elif setting == "continual":
    for m in models:
        multiple_error = []
        multiple_times = []
        multiple_steps = [] # for debugging
        for s in seeds:
            errs = []
            times = []
            steps = [] # for debugging
            if tag != "":
                res_tag = f"{setting}/{m}/{deltas[0]}_{method}_{tag}_seed{s}"
            else:
                res_tag = f"{setting}/{m}/{deltas[0]}_{method}_seed{s}"
            root = os.path.join(path_root, res_tag)

            os.chdir(root)
            # error rate for each tasks
            for j, file in enumerate(glob.glob(f"{root}/*/*.txt", recursive=True)):
                with open(file, "r") as f:
                    for l in f.readlines():
                        l = l.replace("\n", "")
                        if f"{dataset} error" in l:
                            errs.append(float(l.split(":")[-1].strip().strip('%')))
                        elif "elapsed time:" in l:
                            times.append(float(l.split(":")[-1].strip()))
                            multiple_times.append(times)
                        elif "outputs_inference:" in l: # for debugging
                            times.append(float(l.split(":")[-1].strip()))
                            multiple_steps.append(times) 
            # mean error rate across tasks
            last_err = l.split()[-1].strip('%')
            errs.append(float(last_err))
            multiple_error.append(errs)
        
        multiple_error = np.array(multiple_error)
        multiple_times = np.array(multiple_times)
        if len(multiple_steps) > 0:
            multiple_steps = np.array(multiple_steps) # for debugging
        
        if len(multiple_error) > 1:
            std = multiple_error.std(0)
            mean = multiple_error.mean(0)
            # std_t = multiple_times.std(0)
            mean_t = multiple_times.mean(0)
            if len(multiple_steps) > 0:
                mean_s = multiple_steps.mean(0) # for debugging
    
        print(f"{m} " + " ".join(format(x, ".2f") for x in mean) + u"\u00B1" + f"{std[-1]:.2f}" + f" {mean_t[-1]}")
        if len(multiple_steps) > 0:
            print(f"{m} " + " ".join(str(x) for x in mean_s)) # for debugging
            
elif setting == "correlated":
    for m in models:
        # print(f"Dichilet distribution delta : {d}")
        for d in deltas:
            multiple_error = []
            multiple_times = []
            multiple_steps = [] # for debugging
            for s in seeds:
                errs = []
                times = []
                steps = [] # for debugging
                if tag != "":
                    res_tag = f"{setting}/{m}/{d}_{method}_{tag}_seed{s}"
                else:
                    res_tag = f"{setting}/{m}/{d}_{method}_seed{s}"
                root = os.path.join(path_root, res_tag)

                os.chdir(root)
                for j, file in enumerate(glob.glob(f"{root}/*/*.txt", recursive=True)):
                    # print(file)
                    with open(file, "r") as f:
                        for l in f.readlines():
                            l = l.replace("\n", "")
                            if f"{dataset} error" in l:
                                errs.append(float(l.split(":")[-1].strip().strip('%')))
                            elif "elapsed time:" in l:
                                times.append(float(l.split(":")[-1].strip()))
                                multiple_times.append(times)
                            elif "outputs_inference:" in l: # for debugging
                                times.append(float(l.split(":")[-1].strip()))
                                multiple_steps.append(times) 
                last_err = l.split()[-1].strip('%')
                errs.append(float(last_err))
                multiple_error.append(errs)
            multiple_error = np.array(multiple_error)
            multiple_times = np.array(multiple_times)
            if len(multiple_steps) > 0:
                multiple_steps = np.array(multiple_steps) # for debugging

            if len(multiple_error) > 1:
                std = multiple_error.std(0)
                mean = multiple_error.mean(0)
                mean_t = multiple_times.mean(0)
                if len(multiple_steps) > 0:
                    mean_s = multiple_steps.mean(0) # for debugging
                
        
            print(f"{m} " + " ".join(format(x, ".2f") for x in mean) + u"\u00B1" + f"{std[-1]:.2f}" + f" {mean_t[-1]}") 
            if len(multiple_steps) > 0:
                print(f"{m} " + " ".join(str(x) for x in mean_s)) # for debugging
            # print(" ".join(std) + f" {i}") 
            
elif setting == "mixed_domains_correlated":
    for m in models:
        for d in deltas:
            multiple_error = []
            for s in seeds:
                errs = []
                if tag != "":
                    res_tag = f"{setting}/{m}/{d}_{method}_{tag}_seed{s}"
                else:
                    res_tag = f"{setting}/{m}/{d}_{method}_seed{s}"
                root = os.path.join(path_root, res_tag)

                os.chdir(root)
                for j, file in enumerate(glob.glob(f"{root}/*/*.txt", recursive=True)):
                    # print(file)
                    with open(file, "r") as f:
                        for l in f.readlines():
                            l = l.replace("\n", "")
                            if "error:" in l and not ("mean error" in l):
                                errs.append(float(l.split("error:")[-1].strip().strip('%')))
                            elif "Average error across all domains:" in l:
                                errs.append(float(l.split(":")[-1].strip().strip('%')))
                # last_err = l.split()[-1].strip('%')
                # errs.append(float(last_err))
                multiple_error.append(errs)
            multiple_error = np.array(multiple_error)
            
            if len(multiple_error) > 1:
                std = multiple_error.std(0)
                mean = multiple_error.mean(0)
        
            print(f"{m} " + " ".join(format(x, ".2f") for x in mean) + u"\u00B1" + f"{std[-1]:.2f}") 



