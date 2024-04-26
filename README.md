# Uncovering-Narratives
Contains the OnlineAgglomerative class, the MacroNarrative class, and a tutorial for their use.

**How it works**
The detection and evolution tracking works in two steps:

1. First, we track story clusters and how they evolve throughout timesteps. We track these story clusters using the OnlineAgglomerative class.

The OnlineAgglomerative class dynamically clusteris incoming data and re-evaluates existing clusters as new information becomes available. The key function, incremental_fit(), is designed to adjust clusters on the fly, deciding whether new data should merge with existing clusters or form new ones, based on a similarity threshold that has been set in advance.

The class then employs measures like silhouette scores and the pseudo-F index to evaluate the potential merging of clusters for each new batch of data, balancing immediate data integration and long-term structural integrity and coherence of clusters over time. We illustrate this methodology below:

![online_agglo_full_v3](https://github.com/patrikgerard/Uncovering-Narratives/assets/43653986/c2e07c7e-4980-45ed-bfaa-7e87e3aa3d4d)


We believe this real-time clustering -- which enables the discovery of patterns and structures in the data as it evolves, providing valuable insights into the underlying dynamics at play -- can be particularly useful in applications where data is continuously generated and needs to be categorized and understood as it is received, such as monitoring social media for information diffusion, narrative-building, or even detecting astroturfing activities.

Moreover, this allows us to uncover persistent stories, trending stories, and large themes, which can then help us to understand the discursive atmosphere of the group being studied.


2. Then, we apply an additional (optional) layer, where we use the clusters tracked in the previous step as ``story clusters'' and employ the MacroNarrative class to infuse expert knowledge and examine how these story clusters form larger narrative clusters. The formation and tracking of these narrative clusters is illustrated below:

![bfs](https://github.com/patrikgerard/Uncovering-Narratives/assets/43653986/8f0daeac-3b1a-41e7-a7bf-8d01b3cee39d)


**Directory:**
- The OnlineAgglomerative class can be found in the OnlineAgglomerative.py file
- The MacroCluster class can be found in the MacroCluster.py file
- A tutorial notebook can be found in the Tutorial_Jupyter_notebook.py
  - This contains an illustrative example and a data-intensive example of both the online, cross-temporal _story_ clustering and the additional (optional) narrative (MacroCluster) clustering.
 

**To Use the Notebook**
- We created the tutorial notebook in google colab to allows for users to quickly port it. However, there are data-intense elements, and so we had to reduce some of the data analyzed to allow for it to work within the hardware confines of google colab. However, we provide the full, downloadable data for the more data-intense task [here](https://drive.google.com/drive/folders/1NH7HSk3m5eR2wLcmPTTIIqJfC6w6Mkl9?usp=sharing).
  - We recommend downloading the notebook and the data to perform the more data-intensive task, but we have provided a clear example in the outputs of the current notebook.
 
