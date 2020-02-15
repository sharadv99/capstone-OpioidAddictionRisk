# Opioid Misuse Risk Tool

--Data Science project at UC Berkeley--  
--In collaboration with Cameron Kennedy, Aditi Khullar, Rachel Kramer--  


Description
-----------

The Opioid Misuse Risk Tool is a custom web-app that uses machine learning to impact the opioid crisis, specifically by enabling physicians to make more informed decisions about prescribing opioids. Patients submit responses to ~25 demographic and health related questions, which triggers the data pipeline to output a report detailing their personalized probability of opioid misuse (risk score), their percentile of misuse among all patients, and the impact that each of their responses has on their risk score. A calibrated XG Boost model produces the risk score, with Shapley values generating the contributions of individualized risk factors.

Our minimum viable product:

- uncovers hidden insights on each patient through sophisticated, predictive modeling
- showcases transparent, actionable results for physician's to utilize when interacting with each patient
- presents each patient's misuse risk likelihood and ranks the distinct attributes driving their results
- processes a user-friendly, concise assessment and outputs results seconds after submission

For a comprehensive discussion on the problem motivating the solution (ie. Opioid Crisis), the tool's technical design/workflow, model performance, instructions for using the tool, etc. please visit our [website](https://opioidmisuserisk.github.io/)

You can access the tool directly from our website, or by clicking [here](https://opioidrisk.herokuapp.com/polls)

Tech Stack/Methods
-----------

- Python
- XG Boost
- Calibrated Classifiers
- Calibration Curves
- Brier Loss Score
- Shapley Values
- Django (w. HTML/CSS/JS)
- Bootstrap
- Heroku
- FusionCharts

Key Files
-----------

- **`NSDUH-2017-DS0001-info-codebook.pdf`**: This PDF file is the National Survey on Drug Use and Health codebook that contains all the variable names and details of the survey.
- **`Opioid_01LoadData.ipynb`**: This brief file loads data.pickle.zip and adds the outcome variable for Opioid Misuse, saving its output to ./data/misuse.pickle.zip
- **`Opioid_02EDA.ipynb`**: Loads ./data/misuse.pickle.zip and performs some exploratory data analysis. It is not necessary for developing the model, but contains the Weighted Standard Deviation calculations that were instrumental in selecting the variables in the - Opioid_03FeatureSel.ipynb file.
- **`Opioid_03FeatureSel.ipynb`**: This file performs two tasks. 1) It reduces the feature space from 2,631 potential input variables (columns) down to the 28 variables that are used to build the model and correspondingly asked in the web app input form. 2) It executes the data preprocessing module OpioidDataPrep.py used to one hot encode and bucket the variables for input into the model.
- **`Opioid_04Modeling.ipynb`**: This file first splits the data into training, validation, and test sets. It then trains 3 uncalibrated and 3 calibrated models as candidates for the application. It numerically and visually evaluates these models, and then saves the “winning” model along with validation and test data.
- **`Opioid_05FeatureImp.ipynb`**: This file generates the features importance “explainer” in the form of Shapley values for the selected model.
- **`Opioid_06FullPipe.ipynb`**: This file calls OpioidExecution.py to test code from end to end. It also runs a more manual form of this test in the notebook itself.
- **`Opioid_07Testing.ipynb`**: This file further tests the by generating each possible value for the web app input, and random values for all other inputs. In addition to the test, it also records the outputs of the predictions and feature importance values from each test, and then creates box plots of each input and feature importance. These plots were critical in examining the distribution of feature importance for each input variable.
- **`OpioidDataPrep.py`**: This file is called by Opioid_03FeatureSel.ipynb, OpioidExecution.py, and the web app, and does all the data preprocessing. This preprocessing includes scaling and centering continuous variables, one hot encoding all other variables, and logically bucketing responses for variables requiring it.
- **`OpioidExecution.py`**: This file takes inputs from the web app, Opioid_06FullPipe.ipynb, or Opioid_07Testing.ipynb runs the model and feature importance explainer, and returns the risk probability, the percentile of that probability, and the feature importance of all inputs.
- **`Initial_Data_Prep_and_Exploration.ipynb`**: This file contains some initial exploratory data analysis. It lacks a number because it was created before the team organized the files into more modular with numbers, and it is not necessary for developing the model. It also creates > 1 GB files because of an issue the team was having running it on CoLab early in the project, so beware when running it. The CoLab solution was abandoned because it wasn’t sufficiently stable.

