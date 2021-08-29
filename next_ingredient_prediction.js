/*
Model for next ingredient prediction training and deployment for inference.
*/


const cannotTrainMessage = "Cannot train: the last sample has not a label " +
    "associated yet. Select a further label before proceeding.";
const greenBorderShadowStyle = "0 0 60px rgb(0, 255, 0)";
const highlighTime = 600;  // [ms]
const ingredientDivIDs = [
    "first-ingredient",
    "second-ingredient",
    "third-ingredient",
    "fourth-ingredient",
    "fifth-ingredient",
    "sixth-ingredient",
    "seventh-ingredient",
    "eighth-ingredient",
    "ninth-ingredient"
];
const learningRate = 0.1;
const lightBlueBorderShadowStyle = "0 0 60px rgb(0, 162, 255)";
const miniBatchSize = 1;
const nEpochs = 15;
const numClasses = 9;

let ingredientClasses2IDsMapping;
let ingredientIDs2BindedEventsMapping;
let ingredientIDs2ClassesMapping
let ingredientIDs2DivsMapping;
let currentIsLabel = false;
let labels;
let model;
let previousPicturePath;
let samples;
let status;
let trainButton;


function assembleHTMLOfCollectedSample(samplePicturePath, labelPicturePath) {
    return `
      <div class="collected-data">
        <img class="sample" src="${samplePicturePath}"/>
        <img class="label" src="${labelPicturePath}"/>
        <p class="arrow">→</p>
      </div>
    `;
}


function buildModelArchitecture() {
    // defining the architectural hyperparameters:
    let dense_layers_hyperparameters = [
        // {
        //     inputShape: [9],
        //     units: 32,
        //     activation:'relu'
        // },
        {
            inputShape: [9],
            units: 9,
            activation:'softmax',
            dtype: 'float32'
        }
    ];

    // defining the model:
    let model = tf.sequential();
    dense_layers_hyperparameters.forEach(
        function(hyperparameters) {
            model.add(
                tf.layers.dense(hyperparameters)
            );
        }
    );
    return model;
}


function collectSample(category) {
    if (currentIsLabel) {
        labels.push(category);
    } else {
        samples.push(category);
    }

    // considering the next clicked ingredient as a sample if the current one
    // was a label or a label if the current one was a sample:
    currentIsLabel = !currentIsLabel;
}


function displayCollectedSample(samplePicturePath, labelPicturePath) {

    let div = document.createElement('div');
    div.innerHTML = assembleHTMLOfCollectedSample(
        samplePicturePath,
        labelPicturePath
    );
    document.body.appendChild(div);

    div = document.createElement('div');
    div.innerHTML = '<div class="space"></div>';
    document.body.appendChild(div);
}


function dryRun() {
    /*
    Let the model make a useless prediction, so as to be internally ready,
    serving following inferences with no first-time internal prediction setup
    delays.
    */
    model.predict(turnClassesIntoOneHotTensors([4]));
}


function highlightAndCollectCurrent(id, picturePath) {
    // highlighting the currently selected ingredient immediately:
    highlightIngredient(id, 'light blue');

    // collecting the currently selected sample (or label):
    collectSample(ingredientIDs2ClassesMapping[id]);

    if (!currentIsLabel) {
        // displaying the collected sample-label pair:
        displayCollectedSample(previousPicturePath, picturePath);
    } else {
        // remembering the picture of the collected sample, to be displayed
        // with the next one - its label:
        previousPicturePath = picturePath;
    }
}


function highlightCurrentAndPredictedNext(id) {
    // highlighting the currently selected ingredient immediately:
    highlightIngredient(id, 'light blue');

    // getting the id of the ingredient predicted to be selected next from the
    // id of the currently selected ingredient:
    let lastIngredientClass = ingredientIDs2ClassesMapping[id];
    let NextIngredientClass = predictNextIngredientClass(lastIngredientClass);
    let NextIngredientID = ingredientClasses2IDsMapping[NextIngredientClass];

    // highlighting the ingredient predicted to be selected next after the set
    // temporal delay:
    highlightIngredient(NextIngredientID, 'green', true);
}


function highlightIngredient(layoutElementID, colorName,
                             toBeDelayed = false) {
    function setShadowStyle(toBeReset = false) {
        let shadowStyle;
        if (toBeReset) {
            shadowStyle = "none";
        } else {
            // choosing the border shadow style:
            if (colorName === "green") {
                shadowStyle = greenBorderShadowStyle;
            } else if (colorName === "light blue") {
                shadowStyle = lightBlueBorderShadowStyle;
            } else {
                throw("Unknown box shadow style name.");
            }
        }

        // setting the chosen shadow style:
        ingredientIDs2DivsMapping[layoutElementID]
            .style["boxShadow"] = shadowStyle;
    }

    if (toBeDelayed) {
        // highlighting the border shadow with the desired style after the
        // choosen temporal delay:
        setTimeout(
            setShadowStyle,
            highlighTime
        );
    } else {
        // highlighting the border shadow with the desired style immediately:
        setShadowStyle();
    }

    // resetting the border shadow after the lightning time:
    setTimeout(
        function() { setShadowStyle(true); },
        (toBeDelayed) ? (highlighTime * 2) : highlighTime
    );
}


function setUpInference() {
    status.innerHTML = "... Ready for Inference!";

    ingredientDivIDs.forEach(
        function(id) {
            // removing the previous event listeners (only the one including
            // highlight, collection and display behaviors will be available):
            ingredientIDs2DivsMapping[id].removeEventListener(
                'click',
                ingredientIDs2BindedEventsMapping[id],
                false
            );
    
            // setting ingredient's divs to react to clicks with temporary
            // border shadow highlights:
            ingredientIDs2DivsMapping[id].addEventListener(
                'click',
                function () {
                    highlightCurrentAndPredictedNext(id);
                },
                false
            );
        }
    )
}


function predictNextIngredientClass(lastIngredientClass) {
    // return lastIngredientClass !== 8 ? lastIngredientClass + 1 : 0;
    return model.predict(
        turnClassesIntoOneHotTensors([lastIngredientClass])
    ).argMax(1).dataSync()[0];
}


function setUpModel() {
    // instantiating the model architecture:
    model = buildModelArchitecture();

    // assigning the loss function and optimizer for training and the metrics
    // for both training and inference:
    model.compile({
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
        optimizer: tf.train.sgd(learningRate)
    });

    // letting the model make a useless prediction, so as to be internally
    // ready, serving following inferences with no first-time internal
    // prediction setup delays:
    dryRun();
}


function setUpSampleCollection() {
    status = document.getElementById("status");
    trainButton = document.getElementById("trainButton");

    status.innerHTML = "Setting Up";

    ingredientClasses2IDsMapping = {};
    ingredientIDs2BindedEventsMapping = {};
    ingredientIDs2ClassesMapping = {};
    ingredientIDs2DivsMapping = {}
    labels = [];
    samples = [];

    ingredientDivIDs.forEach(
        function(id, indx) {
            // defining ingredient's classes <-> ids mappings:
            ingredientClasses2IDsMapping[indx] = id;
            ingredientIDs2ClassesMapping[id] = indx;

            // defining ingredient's divs:
            ingredientIDs2DivsMapping[id] = document.getElementById(id);

            // getting the ingredient's picture path:
            picturePath = ingredientIDs2DivsMapping[id]
                .getElementsByTagName('img')[0].src

            // setting ingredient's divs to react to clicks with temporary
            // border shadow highlights:
            ingredientIDs2BindedEventsMapping[id] = highlightAndCollectCurrent
                .bind(null, id, picturePath);
            ingredientIDs2DivsMapping[id].addEventListener(
                'click',
                ingredientIDs2BindedEventsMapping[id],
                false
            )
        }
    );

    trainButton.addEventListener(
        'click',
        function () {
            if (samples.length !== labels.length) {
                console.log(cannotTrainMessage);
            } else {
                trainButton.style.display="none";
                trainModel();
            }
        },
        false
    )

    status.innerHTML = "Collecting Samples...";
}


function trainModel() {
    status.innerHTML = "...Training...";

    const trainingConfigs = {
        batchSize: miniBatchSize,
        epochs: nEpochs,
        callbacks: {onEpochEnd},
    }

    function onEpochEnd(epoch, logs) {
        console.log("Epoch " + epoch + ":", "loss", logs.loss.toFixed(2), "|",
                    "accuracy", logs.acc.toFixed(2));
    }

    function printFinalAccuracy(info) {
        let epochs_accuracies = info.history.acc;
        console.log(
            'Post-Training Accuracy:',
            epochs_accuracies[epochs_accuracies.length -1]
        );
    }

    setUpModel();

    // turning samples and labels from class ids into one-hot encodings
    // respectively to feed the model and be comparable to its outputs:
    samples = turnClassesIntoOneHotTensors(samples);
    labels = turnClassesIntoOneHotTensors(labels);
    
    model.fit(samples, labels, trainingConfigs).then(
        info => {
            printFinalAccuracy(info);
            setUpInference();
        }
    );
}


function turnClassesIntoOneHotTensors(classes) {
    return tf.oneHot(
        tf.tensor(
            classes,
            undefined,
            'int32'
        ),
        numClasses
    )
}


setUpSampleCollection();

// samples = [
//     0,
//     5,
//     8,
//     4,
//     1,
//     6,
//     2,
//     4,
//     3,
//     7,
// ];
// labels = [
//     5,
//     0,
//     4,
//     8,
//     6,
//     1,
//     4,
//     2,
//     7,
//     3,
// ];
