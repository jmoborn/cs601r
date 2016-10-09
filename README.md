Image Recognition Pipeline
==========================

Compile and run the C++ feature extraction:
```
g++ -O3 -std=c++0x extract_features.cpp -o extract_features `pkg-config --cflags --libs opencv`
./extract_features
```
Train and test the result:
```
python train_svm.py train
python train_svm.py predict
```
