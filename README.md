<h1 align="center">A Neural Network Trained (and Deployed) on the Client Side</h1>


<p align="center">
    A neural network is first trained, manually selecting examples, and then deployed for inference, all happening in the client's browser. It learns to predict which ingredient among the ones displayed is selected after a first one is clicked.
</p>


## A Video-Example:

At the beginning, pairs of ingredients visited consecutively are selected, emulating past choices:
<br><br>
![](https://github.com/MattiaSarti/next-ingredient-prediction/raw/main/readme_videos/sample_collection.gif)

When clicking on **Train**, a neural network is trained on the selected ingredient pairs to predict the second ingredient after the first one is selected:
<br><br>
![](https://github.com/MattiaSarti/next-ingredient-prediction/raw/main/readme_videos/training.gif)

Now, after selecting an ingredient (highlighted in light blue), it highlights the ingredient predicted as selected next in green:
<br><br>
![](https://github.com/MattiaSarti/next-ingredient-prediction/raw/main/readme_videos/inference.gif)
