<H1><strong>Creation of a Machine Learning Model with FastAPI</strong></H1>

The data used for this example is diabetes from sklearn. The variables that have highest linear correlation with the target variable are "bmi", "bp", "s4", "s5", so this ones are used in this example. 
<br>
<br><img src="https://raw.githubusercontent.com/Jorgearredondoe/Machine-Learning-fastapi/master/assets/img1.png" alt="variables"/>
<br><br>The Regression method is Random Forest from sklearn, with default hyperparameters.
<br>The results of the training are not meassures, because that is not the objective of this project.
<br>
We use FastAPI with Uvicorn server implementation. To access to the main interface of the API you can do it with the following URL http://127.0.0.1:8000/docs.
<br>
<br><img src="https://raw.githubusercontent.com/Jorgearredondoe/Machine-Learning-fastapi/master/assets/img2.png" alt="variables"/>
<br>
Inside the <strong>Predict</strong> Module you can try out a prediction using 4 floats numbers, and you will get a response with the prediction.

<br><img src="https://raw.githubusercontent.com/Jorgearredondoe/Machine-Learning-fastapi/master/assets/img3.png" alt="variables"/>
<br>
<br><img src="https://raw.githubusercontent.com/Jorgearredondoe/Machine-Learning-fastapi/master/assets/img4.png" alt="variables"/>
<br>