#this repo is based on FCOS:
https://github.com/tianzhi0549/FCOS.git

#paper link:
https://www.mdpi.com/2072-4292/12/6/908

#Required hardware
We use 1 Nvidia titan xp GPU. 

#Training
The following command line will train FCOS_imprv_R_50_FPN_1x on 8 GPUs with Synchronous Stochastic Gradient Descent (SGD):

sh ./tools/train_fcos_range.sh
  
 #Citations
Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follows.

@inproceedings{tian2019fcos,
  title   =  {{FCOS}: Fully Convolutional One-Stage Object Detection},
  author  =  {Tian, Zhi and Shen, Chunhua and Chen, Hao and He, Tong},
  booktitle =  {Proc. Int. Conf. Computer Vision (ICCV)},
  year    =  {2019}
}


## License

For academic use, this project is licensed under the 2-clause BSD License - see the LICENSE file for details. For commercial use, please contact the authors. 
