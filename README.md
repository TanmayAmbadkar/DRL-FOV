# Deep reinforcement learning approach to predict head movement in 360° videos - *Tanmay Ambadkar, Pramit Mazumdar*

## Abstract

*The popularity of 360&deg; videos has grown immensely in the last few years. One probable reason is the availability of low-cost devices and ease in capturing them. Additionally, users have shown interest in this particular type of media due to its inherent feature of being immersive, which is completely absent in traditional 2D videos. Nowadays such powerful 360&deg; videos have many applications such as generating various content-specific videos (gaming, knowledge, travel, sports, educational, etc.), during surgeries by medical professionals, in autonomous vehicles, etc. A typical 360&deg; video when seen through a Head Mounted Display (HMD) gives an immersive feeling, where the viewer perceives standing within the real environment in a virtual platform. Similar to real life, at any point in time, the viewer can view only a particular region and not the entire 360&deg; content. Viewers adopts physical movement for exploring the total 360&deg; content. However, due to the large volume of 360&deg; media, it faces challenges during transmission. Adaptive compression techniques have been incorporated in this regard, which is in accordance with the viewing behaviour of a viewer. Therefore, with the growing popularity and usage of 360&deg; media, the adaptive compression methodologies are in development. One important factor in adaptive compression is the estimation of the natural field-of-view (FOV) of a viewer watching 360&deg; content using a HMD. The FOV estimation task becomes more challenging due to the spatial displacement of the viewer with respect to the dynamically changing video content. In this work, we propose a model to estimate the FOV of a user viewing a 360&deg; video using an HMD. This task is popularly known as the Virtual Cinematography. The proposed FOVSelectionNet is primarily based on a reinforcement learning framework. In addition to this, saliency estimation is proved to be a very powerful indicator for attention modelling. Therefore, in this proposed network we utilise a saliency indicator for driving the reward function of the reinforcement learning framework. Experiments are performed on the benchmark Pano2Vid 360&deg; dataset, and the results are observed to be similar to human exploration.*

[[pdf]](https://library.imaging.org/ei/articles/34/10/IPAS-367) [[video]](https://higherlogicstream.s3.amazonaws.com/IST/42bfe16d-f5e8-0bad-05c5-8a5a24b6bfe7_file.mp4)

## Model Architecture

The model architecture can be broken down into two parts. The first part processes an individual 360 saliency frame to get an embedding. The downstream model takes all embeddings to output an action. 

### Frame Embedding Model

![frame embedding model](assets/Single_ConvBlock.png)

### Downstream Model

![down stream modek](assets/Final_Model.png)

## Framework

A Deep RL framework consists of an environment and an agent. We have created a custom environment for this task, which provides the starting frame, the next frame for performing an action, and a reward function. The agent is a Deep QN agent with nine actions. It interacts with the environment to learn how to find the highest probable FOV. The following image describes the interactions between the environment and the agent. 

![framework](assets/block_diagram.png)

## Results

Some selected FOVs can be seen here

![result](assets/results.png)

A short gif showing the selected FOV is

![fov](assets/project1.gif)

# Setup

Install the requirements in a virual environment using 
```
pip install -r requirements.txt
```
To replicate results in the paper, please follow [Saliency-prediction-for-360-video](https://github.com/vhchuong1997/Saliency-prediction-for-360-degree-video). This is the saliency model used in the paper. The dataset can be downloaded here - [dataset](https://drive.google.com/file/d/1ZPPUrgYxRqpryhGgaM31ov8v82tdMVF8/view?usp=sharing).  

Unzip the dataset in the project folder, accordingly modify the parameters in *params.py*.

Run main.py to train model and get a test run of the model. Trained weights to be added soon.
```
python main.py
```
----
If you find this work helpful, please consider citing
 ```
 @article{Ambadkar2022,
  doi = {10.2352/ei.2022.34.10.ipas-367},
  url = {https://doi.org/10.2352/ei.2022.34.10.ipas-367},
  year = {2022},
  month = jan,
  publisher = {Society for Imaging Science {\&} Technology},
  volume = {34},
  number = {10},
  pages = {367--1--367--5},
  author = {Tanmay Ambadkar and Pramit Mazumdar},
  title = {Deep reinforcement learning approach to predict head movement in 360{\&}amp$\mathsemicolon${\#}{xB}0$\mathsemicolon$ videos},
  journal = {Electronic Imaging}
}
```

