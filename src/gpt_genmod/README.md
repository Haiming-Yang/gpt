## gpt_genmod
Here are the instructions to run the codes to obtain results shown in [_A Modified Convolutional Network for Auto-encoding based on Pattern Theory Growth Function_ ](https://arxiv.org/abs/2104.02651). We show here the quick start version from scratch. 

**Assume we run everything from the root folder, gpt.**

For CIFAR-10 result:
```
python main_gpt_genmod.py --mode training --dataset cifar10 --PROJECT_ID cifarN0 --N_EPOCH 48 --n 4096 --realtime_print 1 --SAVE_IMG_EVERY_N_ITER 3000 
python main_gpt_genmod.py --mode plot_results --PROJECT_ID cifarN0
python main_gpt_genmod.py --mode sample --PROJECT_ID cifarN0
python main_gpt_genmod.py --mode sample --PROJECT_ID cifarN0 --sampling_mode transit
```

For CelebA 64x64 result:
```
python main_gpt_genmod.py --mode training --PROJECT_ID celebaN0 --dataset celeba64 --N_EPOCH 2 --realtime_print 1 --OBSERVE 0 --EVALUATE 1 --SAVE_IMG_EVERY_N_ITER 640 --n 1024
python main_gpt_genmod.py --mode sample --PROJECT_ID celebaN0 --dataset celeba64
python main_gpt_genmod.py --mode sample --PROJECT_ID celebaN0 --sampling_mode transit --dataset celeba64
```

<br>
<br>
<br>
=== end ===
<br>
<br>
<br>
