# COCOA
Data enrichment, also known as horizontal data augmentation, is an important element in many preprocessing pipelines, especially if the dataset at hand does not contain the related information for the user's machine learning (ML) application. Current data augmentation solutions enrich the input dataset with joinable tables, i.e.,~features that are related to the domain of the given dataset.
However, they do not take into account the downstream ML application leading to lower downstream accuracy and requiring extensive feature selection techniques on top.
Using large database repositories for feature extraction typically results in a large number of feature candidates and requires a more restrictive enrichment process that not only verifies the relatedness of features but also their informativeness to unburden the feature selection process. 
In this paper, we propose \system a data augmentation solution that supports the downstream ML task by taking into account the correlation of the extracted features to the target of the ML application.
To generalize our solution to large scale repositories with millions of tables, we also introduce a new index structure that enables the system to detect non-linear correlations between external features with the user-defined target column in linear time complexity.
Our index structure, thus, leads to a faster and more scalable enrichment process.
Our experimental results show that our index structure allows us to enrich the data at hand hundreds of times faster than current correlation-driven augmentation solutions.

##Using the system
The project contains three main python files: index_generation.py, COCOA.py, and SBE.py.
The index_generation.py file is responsible to generate the inverted and also the order index out of the tables stored in the DB. These indices structures are used to efficiently find the joinable external tables and also the non-linear correlation between external columns and the ML target column.
SBE.py contains methods to find the joinable tables using the inverted index generated by ``generate_inverted_index()'' function in index_generation.py. ``enrich_SBE()'' function, based on the provided parameters, enriches the given dataset. ``enrich_COCOA()'' function in COCOA.py enriches the input dataset in the same way but by leveraging the order index generated by ``generate_order_index'' function in index_generation.py.

