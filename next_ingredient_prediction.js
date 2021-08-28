/*
Model for next ingredient prediction training and deployment for inference.
*/


const greenBorderShadowStyle = "0 0 60px rgb(0, 255, 0)";
const highlighTime = 600;  // [ms]
const lightBlueBorderShadowStyle = "0 0 60px rgb(0, 162, 255)";

let ingredientClasses2IDsMapping;
let ingredientDivIDs;
let ingredientIDs2ClassesMapping
let ingredientIDs2DivsMapping;
let model;


function buildModelArchitecture() {
    // defining the dense layers' hyperparameters:
    let dense_layers_hyperparameters = [
        {
            inputShape: [1],
            units: 3,
            activation:'relu'
        },
        {
            units: 9,
            activation:'softmax'
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


function highlightIngredient(layoutElementID, colorName, toBeDelayed = false) {

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
        ingredientIDs2DivsMapping[layoutElementID].style["boxShadow"] = shadowStyle;
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


function predictNextIngredientClass(lastIngredientClass) {
    return lastIngredientClass !== 8 ? lastIngredientClass + 1 : 0;
}


function setup() {
    ingredientDivIDs = [
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

    ingredientClasses2IDsMapping = {};
    ingredientIDs2ClassesMapping = {};
    ingredientIDs2DivsMapping = {}

    ingredientDivIDs.forEach(
        function(id, indx) {
            // defining ingredient's classes <-> ids mappings:
            ingredientClasses2IDsMapping[indx] = id;
            ingredientIDs2ClassesMapping[id] = indx;

            // defining ingredient's divs:
            ingredientIDs2DivsMapping[id] = document.getElementById(id);

            // setting ingredient's divs to react to clicks with temporary
            // border shadow highlights:
            ingredientIDs2DivsMapping[id].addEventListener(
                'click',
                function () {
                    temp(id);
                },
                false
            )
        }
    );

    model = buildModelArchitecture();
}


function temp(id) {
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

setup();
