## gpt_mnist
Here are the instructions to run the codes to obtain results in [_Convolutional Neural Network Interpretability with General Pattern Theory_](https://arxiv.org/abs/2102.04247). We show here the quick start version from scratch. 

**Assume we run everything from the root folder, gpt.**

The following shows a very short training process. checkpoint/myproject0001 will appear with model.net and a few more items.
```
python main_gpt_mnist.py --PROJECT_ID myproject0001 --N_EPOCH 1 --N_PER_EPOCH 120 --realtime_print 1
```


Let us finetune it. Copy the whole _myproject0001_ folder from checkpoint to *_treasure_trove* folder. We keep it in this folder as a permanent checkpoint. Ensure that nothing in the script can affect or change it. Rename it to *NSCC_NICE1*. This folder name is fixed. It is used by the customized script *src\gpt_mnist\finetuner\finetune_NSCC_NICE1.py*. 
```
python main_gpt_mnist.py --PROJECT_ID NSCC_NICE1 --mode finetune --FINETUNE_ID myproject0001.FINETUNE --realtime_print 1 --load_from_trove 1 --N_EPOCH 1 --N_PER_EPOCH 120
```
checkpoint/myproject0001.FINETUNE folder will appear. If you re-run the above, the content of this folder will be overridden. If we set --load_from_trove 1 instead, finetuning will continue from where we left off.

### Results
Fig. 1 is produced using the following. You will need to train the model longer to obtain similar results as the paper.
```
python main_gpt_mnist.py --PROJECT_ID myproject0001.FINETUNE --mode generate_samples --random_batch 0
```

We can produce the images from fig. 4 from the finetuned model. It will be saved in checkpoint/myproject0001.FINETUNE/XAI
```
python main_gpt_mnist.py --PROJECT_ID myproject0001.FINETUNE --mode heatmaps
python main_gpt_mnist.py --PROJECT_ID myproject0001.FINETUNE --mode heatmapsGC
```

The following will produce basic.csv and basic_summary.csv folders that contain the accuracy, precision and recall metrics.
```
python main_gpt_mnist_eval.py --PROJECT_ID myproject0001.FINETUNE
```

Fig. 5 can be obtained by the following. It will take a while if larger samples are used. These will create a lot of files in the project's eval folder. The files store the results.
```
python main_gpt_mnist_eval.py --PROJECT_ID myproject0001.FINETUNE --mode sanity_weight_randomization --N_EPOCH 1 --N_EVAL_SAMPLE 8

python main_gpt_mnist_eval.py --PROJECT_ID myproject0001.FINETUNE --mode compute_AOPC --N_EPOCH 1 --N_EVAL_SAMPLE 8
```

To view the files, run the following respectively.
```
python main_gpt_mnist_eval.py --PROJECT_ID myproject0001.FINETUNE --mode view_weight_randomization

python main_gpt_mnist_eval.py --PROJECT_ID myproject0001.FINETUNE --mode view_AOPC
```

### Final fine-tuning

In the paper, actually another very short fine-tuning process is also done. Copy the whole _myproject0001.FINETUNE_ to *_treasure_trove* folder too. This is to save a copy of the trained model we have so far (especially useful if we have spent a lot of time training it). Rename it to *NSCC_NICE2*. Now let us do the second fine-tuning.
```
python main_gpt_mnist.py --PROJECT_ID NSCC_NICE2 --mode finetune --FINETUNE_ID myproject0001.FINETUNE2 --realtime_print 1 --load_from_trove 1 --N_EPOCH 1 --N_PER_EPOCH 64
```
checkpoint/myproject0001.FINETUNE2 folder will appear. However, model.net.best will only appear if there is improvemnt in average losses. Likewise, if you set --load_from_trove 1 instead, finetuning will continue from where we left off.

We can run the commands in **Results** section after this fine-tuning. But remember to change to --PROJECT_ID myproject0001.FINETUNE2.

<br>
<br>
<br>
=== end ===
<br>
<br>
<br>
