# AI-Agriculture-Capstone-Project
Repository for AI Capstone Project. 

Agro-AI, An Educational Source for AI Usage in Precision Agriculture. This is a web application that goes through the machine learning process in creating and training a model that detects leaf blight in corn images. The user will be able to interact with the machine learning process through the labeling step and see the quality of their model through the confidence level and the decision tree generated based on the model. The user can then save their model to use it later or keep training their model. By seeing how the generated tree changes and the confidence level as the user keeps training or making new models we hope that they get a better understanding of how machine learning works.

# Deployment/ How To Run in your machine

# Create a directory
Create a directory to hold the application, in this case we will use agro-ai

cd agro-ai

# Clone the github repository
git clone -b Deployment_with_samples https://github.com/SgtThunderWaffle/AI-Agriculture-Capstone-Project.git

cd AI-Agriculture-Capstone-Project

# set up virtual environment - just need to do this once
python3 -m venv env # this creates a folder called "env" that will hold all the necessary libraries

# install graphviz
apt install graphviz | use other package manager | download and run installer # this is required for creating images of model decision trees

# Activate the virtual environment
source ./env/bin/activate # tells Python that you're using this virtual environment on your console, you should see the prompt change to include "(env)" at the start of the prompt. for future sessions, you just need to run the "source ./env/bin/activate" command to have the correct virtual environment

# install all the necessary libraries - just need to do this once
pip3 install -r requirements.txt

# run the application
python3 -m flask run -h 0.0.0.0


# Release Notes
Milestone 1 :
In milestone 1, our group explained what we had learned and managed to reverse engineer from the previous projects github, which most of which we were able to recycle into our new model and web infrastructure. 

Milestone 2: 
In milestone 2, our group had successfully ran the old files, and updated them somewhat to begin work on training our AI model to take in new data, switching it’s data to the drone images we are currently using. 

Milestone 3:
In milestone 3 our group had the AI model trained in a state to be presentable and gauge its initial confidence level when being changed to a new type of images. We made necessary changes after this to better improve the ability of the model to make predictions based on drone data. 

Milestone 4:
In milestone 4 our group pivoted to focusing on the other half of the project, the website and active learning objective. As such we created skeleton pages to fit our initial wireframes and populated them with fitting information, and while the pages were not completely designed, they were populated with information and had a flow that would be maintained as the project reached it’s final milestone. 

Milestone 5: 
Finally for milestone 5, we updated our pages and implemented our last few features, mainly implementing saving and loading models for users, and contrasting that to the default model. As well as finally pushing all of our progress to the main branch at the end. 

