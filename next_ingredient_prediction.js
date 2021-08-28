/*
Model for next ingredient prediction training and deployment for inference.
*/


let ingredientDivIDs;
let ingredientIDs2DivsMapping = {};

const greenBorderShadowStyle = "0 0 60px rgb(0, 255, 0)";
const lightBlueBorderShadowStyle = "0 0 60px rgb(0, 162, 255)";


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
                    highlightBorderShadow(id, 'light blue')
                },
                false
            )
        }
    );
}


function highlightBorderShadow(layoutElementID, colorName) {
    // highlighting the border shadow with the desired color:
    if (colorName === "green") {
        ingredientIDs2DivsMapping[layoutElementID].style["boxShadow"] = greenBorderShadowStyle;
    } else if (colorName === "light blue") {
        ingredientIDs2DivsMapping[layoutElementID].style["boxShadow"] = lightBlueBorderShadowStyle;
    } else {
        throw("Unknown box shadow style name.");
    }

    // resetting the border shadow after the desired temporal delay:
    setTimeout(
        function() {
            ingredientIDs2DivsMapping[layoutElementID].style["boxShadow"] = "none";
        },
        300
    );
}

setup();
