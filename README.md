# COCOA
Data enrichment, also known as horizontal data augmentation, is an important element in many preprocessing pipelines, especially if the dataset at hand does not contain the related information for the user's machine learning (ML) application. Current data augmentation solutions enrich the input dataset with joinable tables, i.e.,~features that are related to the domain of the given dataset.
However, they do not take into account the downstream ML application leading to lower downstream accuracy and requiring extensive feature selection techniques on top.
Using large database repositories for feature extraction typically results in a large number of feature candidates and requires a more restrictive enrichment process that not only verifies the relatedness of features but also their informativeness to unburden the feature selection process. 
In this paper, we propose \system a data augmentation solution that supports the downstream ML task by taking into account the correlation of the extracted features to the target of the ML application.
To generalize our solution to large scale repositories with millions of tables, we also introduce a new index structure that enables the system to detect non-linear correlations between external features with the user-defined target column in linear time complexity.
Our index structure, thus, leads to a faster and more scalable enrichment process.
Our experimental results show that our index structure allows us to enrich the data at hand hundreds of times faster than current correlation-driven augmentation solutions.
