# DurOrNoDur
Project for Statistical Machine Learning to classify images or deer

## Instructions for setting up VS Code with Python/Pip
Use this link from Microsoft to set up you VS Code IDe and Python so that you can easily install packages via pip. I used this guide over the summer and it shouldn't take more than 30-60 min to set up 

https://code.visualstudio.com/docs/python/python-tutorial

## Setup
After cloning the repo, run the following to point to a virtual environment and install all pip depedencies:
1. useVirtualEnv.bat
2. installDepedencies.bat

If you add additional pip depdencies, please update the requirements.txt. This can be done most easily using the following command:
`updateRequirements.bat`

## Useful Links
Google Drive Folder - https://drive.google.com/drive/folders/14HpxbZN-GuVDOwKQypAx36eTKq4zwKGX

OverLeaf - https://www.overleaf.com/project/5f84fef392650b0001aeb062

## PR/Branch Management
Please use separate branches and utilize a pull request to master if you are working on a non-trivial change to the source code (e.g. implementing classifiers or working on image pre-processing).

## Overall Areas of Code:
1. imageParser.py - Process Image Files to get the Feature Features 
2. image.py - House properties of a file
3. classifier.py - Classification Implementation(s)
4. analysis.py - Post-processing of classification --> graphs lookin' pretty and shit
5. runner.py - Runner File to manage doing stuff.
