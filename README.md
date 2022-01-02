# political-parties-analysis

The solution is provided as a Python Jupyter notebook. It can be loaded in a working jupyter notebook server, but to make it simpler to test and deploy, a Docker solution is also provided. This solution requires only working Docker installation.

There are two shell scripts provided for MacOS for starting and stoping the container with the notebook server.
To start the conatiner you can use ./run.sh. 
This scrpt will pull the latest jupyter/datascience-notebook docker image and run it localy. It will map the subfolder "analysis" from the host to the folder "/home/jovian/work" in the container. The notebook server is runing in the conainer on port 8888 which is mapped to the host port 8888 (which can be changed by changng the PORT variable in the run.sh file).
The script is then waiting for the container to be ready after which it opens the url of the notebook using the users default browser.

To stop the running container, you can use the ./stop.sh command.
 
This work is presented in jupyter notebook, because it gives me more space for explanation and I find it more sutable for this kind of exploratory analysis. Otherwise it has a package structure, the functions are modulated in separate python file, so it could be easily implemented in a pycharm project.

Justify your methodological choices in each step (1–5). What other choices could you have made? What would have been their pros and cons?
I choose for the simple approach of feature selection (detal description in the notebook) because of the interpretablity of the model. It's intuitive and it could be easily explained. Also the accuracy of the model is good.

PCA (Principal Component Analysis) is also an option when comes down to dimensionality reduction. PCA is designed to select the features based on the variance that they cause in the output. Original features of the dataset are converted to the Principal Components which are the linear combinations of the existing features. The feature that causes highest variance is the first Principal Component. The feature that is responsible for second highest variance is the second Principal Component, and so on. So, PCA tries to extract the max variance combining all features, what could lead to more accurate model than selecting only one feature. However principal components are not as readable and interpretable as original features.
PCA also asks for data standardization, since it's affected by the magnitude of the scale. In our case all variables are on scale [1-7] or [1-10], so the difference is not big, but in principle it's always recommendable standardization to be included. 
Standard PCA can detect only linear relationships between variables/features. If not all the variables are linearly dependent of each other, than model doesn't perform well. There are standard python packages packages for non-linear cases and seems that all these shortcomings could be compensate, but interpretability will still be there as a big issue.
In short the main difference between PCA and feature selectionis is that PCA will try to reduce the dimensionality by exploring how one feature of the data is expressed in tearms of the other features (linear dependecy). Feature selection instead, takes the target into consideration. 
At the end of the document (in APPENDIX) a small experiment is done applying PCA to the small set of selected most important variables. The idea is to see how much more information could be obtained with by adding a few statistically significant features. Or how much information is lost by keeping the simple model with one explanatory variable. With only a few known features we can interpret the results, but applying the PCA on a huge dataset, whitout knowing in advance anything about the relationship between the variables, will make the interpretation of the results very hard. 

Explain in a non-technical way what the low-dimensional representation of the data means (your visualization in the step 2). What could you teach a politician about the European political landscape based on the dimensionality reduction and/or possible additional analyses you may produce.

It's explained in the notebook as well. Variables/features represent the dimensions. Number of the features determines the dimensionality of the data set. If we have 55 variables it's 55 dimensional set. It's very hard to visuelize 55 dimensions. We'll need 1485 individual graphs, to represent all mutual relationships between the variables. However, not all variables bear the same importance. Variables that are not important don't contribute to the model. It's always better to extract only a few statistically most significant variables, who can explain most of the variance of the model. We can than easily visualize and explain the results. 
It turns out that control of the budget is the main factor to determine how much one party supports EU leadership.
Basically, parties that support EU leadrship are ready to leave control of the budget to the EU authorities. They also trust EU to the other important and highly correlated issues to budget like EU foreign and security policy or markets. They are usually parties that promote liberal values. However about some sensitive issues like migration policy the liberal parties are not united. Oppositelly conservative parties are more inclined to their national states and nationalism as a value than cosmopolitism or EU leadership. They are also unanimous about the migration policy. 
Also East European parties show some differences comparing to the western parties. The divison on liberal-conservatives on supporting EU leadership is not that clear as at the west side. Some parties that claim to be liberal are very sceptical towards EU leadership. It seems that what they are declared for and what they really are, are two different things. 

How would you deploy the model to a cloud environment so that it would be able to withstand 1 million users per hour?
I woud use CodeDeploy a deployment service that automates application deployments to Amazon EC2 instances, on-premises instances, serverless Lambda functions, or Amazon ECS services.
