# Cross-Age Effect Classifier
A Keras based, DeepNet Age Classifier. The behavioural test is moved to: http://cagataygursoy.xyz.  

The classifier is directly linked to my Master of Science studies at METU CogSci. Since I am a dedicated supporter of [Open Science](https://www.fosteropenscience.eu/content/what-open-science-introduction), I have released my thesis in pdf format in this [repository](https://github.com/caggursoy/Cross-Age-Effect/blob/master/necati_cagatay_gursoy_msc_thesis.pdf) and in my personal [webpage](http://cagataygursoy.xyz/).  

It is subject to change as the classifier is directly related with my MSc thesis work and more updates may come in the future.
Right now will the classifier only work with Park Aging Mind Lab's (PAML) Aging Dataset's Neutral Face dataset. For other datasets, some filename extracting alterations and other changes might be required.  

I have used Yann LeCun's LeNet for the classifier's basis. More info on LeCun and LeNet here: http://yann.lecun.com/exdb/lenet/

More info about PAML can be find here: http://agingmind.utdallas.edu/  

During Deep Learning implementation, I have used various sources and tutorials. PyImageSearch's following two tutorials were the key tutorials that I have implemented my classifier upon:  
1) Tutorial about LeNet: https://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/
2) Tutorial about image classification: https://www.pyimagesearch.com/2017/12/11/image-classification-with-keras-and-deep-learning/
  
For GradCam codes, I have heavily influenced from eclique's really useful repository called Keras-Gradcam. More info on repository here: https://github.com/eclique/keras-gradcam/blob/master/grad_cam.py. More info on GradCam can be found here: https://arxiv.org/abs/1610.02391
  
Update @ 23.07.2019
My thesis is completed right now. I will be publishing the codes and intermediate layer activation output images here shortly. Moreover, the thesis will be published in my website.

Update @ 10.07.2019
As my thesis comes to a conclusion, the whole code will be released here with the how-to-run readmes soon.

Update @ 23.02.2019
Old code for Behavioural test is moved to legacy folder.  
A new folder for Keras based DeepNet Age Classifier is created and necessary files were uploaded. Comments for each file will be inserted in order to increase readability of the code.  

Update @ 07.10.2018  
The Python code is updated as a new facial image dataset from Park Aging Mind Lab at UTDallas has arrived.  
You can find further info about them from the following webpage: http://agingmind.utdallas.edu  

Update @ 18.05.2018  
You can find the executable Python files below, for Windows and MacOS separately.  
Windows version: https://drive.google.com/open?id=1FmdAbw2mwW0-4X-QHtndPpR4rchVgm8w  
MacOS version: https://drive.google.com/open?id=1fZiERz0EuyGXVAgxd4mgMCSUDSW2ftNp  

Update @ 15.05.2018:  
I decided to have Python version as well, since obtaining an executable version is easier in Python. Moreover, I have made slight changes in Matlab version.  

Previously used face image dataset is provided from FG-NET group's facial database.  
About FG-NET: http://www-prima.inrialpes.fr/FGnet/html/about.html  
Since FG-NET is not distributing their database right now, I have reached their dataset thanks to Yangwei Fu.  
About Yangwei Fu: http://yanweifu.github.io/FG_NET_data/  

[License](https://github.com/caggursoy/crossageeffect/blob/master/LICENSE)
