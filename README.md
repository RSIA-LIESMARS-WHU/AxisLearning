This project hosts the code for implementing the Axis Learning for Orientated Objects Detection in Aerial Images algorithm for object detection, as presented in our paper:

Xiao, Z.; Qian, L.; Shao, W.; Tan, X.; Wang, K. Axis Learning for Orientated Objects Detection in Aerial Images. Remote Sens. 2020, 12, 908.

The full paper is available at: https://www.mdpi.com/2072-4292/12/6/908.

Implementation based on FCOS: https://github.com/tianzhi0549/FCOS.git

we propose a new one-stage anchor-free method to detect orientated objects in per-pixel prediction fashion with less computational complexity. Arbitrary orientated objects are detected by predicting the axis of the object, which is the line connecting the head and tail of the object, and the width of the object is vertical to the axis.

## Required hardware
We use 1 Nvidia titan xp GPU. 

## use tips
1. config conda environment as FCOS
2. cd ~project and run 


    python setup.py build develop


3. run following cmd and the copy the .so result file to rotation directory

    python rotation_setup.py build



## Training
sh ./tools/train_fcos_range.sh


## License

For academic use, this project is licensed under the 2-clause BSD License - see the LICENSE file for details. For commercial use, please contact the authors. 
