Code for the paper "Complex Unitary recurrent Neural Network using Scaled Cayley Transform " the paper "Orthogonal Recurrent Neural Networks with Scaled Cayley Transform", https://arxiv.org/abs/1811.04142.

Uses Tensorflow. To run, download the desired experiment code as well as the "scuRNN.py" script.

Each script uses command-line arguments to specify the desired architecture. For example, to run the MNIST experiment with a hidden size of 116 For the unpermuted MNIST experiment, an RMSProp optimizer to update the skew-Hermitian matrix and an Adagrad optimizer to update the scaling matrix with all other parameters updated using the Adam optimizer the learning rateswere10−4,10−3, and 10−3 respectively use the following in the command line:


python scuRNN_MNIST.py 116 adagrad 1e-4 rmsprop 1e-3 adam 1e-3

For permuted experiments use permuteflag = True in line 34 of the scuRNN_MNIST.py script and use the appropriate command.

Citation

If you find scuRNN useful in your research, please consider to cite the following related papers:
@INPROCEEDINGS{madu20, title={Complex Unitary Recurrent Neural Networks Using Scaled Cayley Transform}, url={https://www.aaai.org/ojs/index.php/AAAI/article/view/4371}, journal={Proceedings of the AAAI Conference on Artificial Intelligence}, author={Maduranga, Kehelwala D. G. and Helfrich, Kyle E. and Ye, Qiang},year ={2019}}


@article{madu18, author = {Maduranga, Kehelwala D. G. and Helfrich, Kyle E. and Ye, Qiang}, title = {Complex Unitary Recurrent Neural Networks using Scaled Cayley Transform}, year = {2018}, url = {https://arxiv.org/abs/1811.04142},
 }
