/*
Model for next ingredient prediction training and deployment for inference.
*/


const greenBorderShadowStyle = "0 0 60px rgb(0, 255, 0)";
const highlighTime = 600;  // [ms]
const lightBlueBorderShadowStyle = "0 0 60px rgb(0, 162, 255)";

let ingredientDivIDs;
let ingredientIDs2DivsMapping = {};


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

    // setting ingredient's divs to react to clicks with temporary border
    // shadow highlights:
    ingredientDivIDs.forEach(
        function (id){
            ingredientIDs2DivsMapping[id] = document.getElementById(id);
            ingredientIDs2DivsMapping[id].addEventListener(
                'click',
                function () {
                    highlightIngredient(id, 'light blue');
                },
                false
            )
        }
    );
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

setup();
