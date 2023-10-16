# CancerGATE
CancerGATE: Prediction of cancer driver genes using graph attention autoencoder
 
## Dependencies
This project is written in Python 3.8. We were tested on [the tensorflow docker environment](https://www.tensorflow.org/install/docker).
The latest tensorflow docker environment was set in following code:
```angular2html
docker pull tensorflow/tensorflow:latest-gpu
docker run --gpus all -it -p {server port}:{container port} -v {server directory path}:{container directory path} --name cancerGATE tensorflow/tensorflow:latest-gpu bash
```

CancerGATE has the [dgl](https://www.dgl.ai/) in [tensorflow backend](https://docs.dgl.ai/en/1.1.x/install/index.html#working-with-different-backends) as a dependency.
To download the dgl package with the appropriate CUDA of the tensorflow docker, use the following code.
```angular2html
pip install dgl-cu111 dglgo -f https://data.dgl.ai/wheels/repo.html
```

## Run CancerGATE
Run the following code in `src` folder.
```angular2html
python main_specific_cancers
```
It will return the learned GATE models in `result/checkpoints/`, the AUROC and AUPRC value in `result/performance`, and CancerGATE score in `result/prediction_score`.
