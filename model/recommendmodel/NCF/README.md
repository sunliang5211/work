conda create -n python2 python=2.7
activate python2
conda install scipy==0.19.1
pip install numpy==1.13.1
pip install theano==0.8.0
pip install keras==1.0.7
pip install h5py==2.9.0

# Neural Collaborative Filtering

- python:  2.7
- scipy : 0.19.1
- numpy : 1.13.1
- theano: 0.8.0
- keras:  1.0.7
- h5py :  2.9.0

Linux 下：修改 ~/.keras/keras.json 中  "backend": "theano"
Windows 下： 修改 C:\Users\"本地用户名"\.keras\keras.json  中  "backend": "theano"

Run MLP:
```
python MLP.py --dataset ncf --epochs 20 --batch_size 256 --layers [64,32,16,8] --reg_layers [0,0,0,0] --num_neg 4 --lr 0.001 --learner adagrad --verbose 1 --out 1
