#!/usr/bin/python3

import sys, subprocess, os, argparse

EXE_PATH = "mnistFuse"
OUT_FILE = "result/result"

def parseResult(txt, res_dict):
  txt = txt.split("\n")
  for l in txt:
    name = l.split(":")[0].strip()
    if name not in res_dict:
      res_dict[name] = {}
    result = l.split(":")[1].strip().split(",")
    for item in result:
      if "Accuracy" in item:
        title = "Accuracy"
        value = float(item.split()[1].strip().strip("%"))/100
      else:
        title = " ".join(item.split()[:-2])
        value = float(item.split()[-2])
      if title not in res_dict[name]:
        res_dict[name][title] = []
      res_dict[name][title].append(value)

def outputResult(outfile, test, res_dict, iterations):
  keys = res_dict.keys()
  with open(outfile+"_"+test.replace(" ","_")+".csv", "w") as f:
    f.write(", ".join(keys)+"\n")
    for i in range(iterations):
      line = []
      for key in keys:
        line.append(str(res_dict[key][i]))
      f.write(", ".join(line)+"\n")

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--iter', type=int, default=100, help="Number of Iterations")
parser.add_argument('--train', type=str, default="60000", help="Number of Train Samples")
parser.add_argument('--test', type=str, default="10000", help="Number of Test Samples")
parser.add_argument('outfile', nargs="?", type=str, default=OUT_FILE, help="Output File for Result")
args = parser.parse_args()

if not os.path.isfile(EXE_PATH):
  subprocess.run(['make'])
  if not os.path.isfile(EXE_PATH):
    print(f"ERROR: {EXE_PATH} not found")
    exit(1)

res_dict = {}

for i in range(args.iter):
  print(f"{i+1}/{args.iter}")
  txt = subprocess.run(["./"+EXE_PATH, args.train, args.test], stdout=subprocess.PIPE).stdout.decode("utf-8").split("TESTING")[1].strip()
  parseResult(txt, res_dict)

for key, val in res_dict.items():
  outputResult(args.outfile, key, val, args.iter)

print("Done")
