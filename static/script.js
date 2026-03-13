function openTab(tab){

    let tabs = document.getElementsByClassName("tabcontent");

    for(let i = 0; i < tabs.length; i++){
        tabs[i].style.display = "none";
    }

    document.getElementById(tab).style.display = "block";
}


async function predict(){

    let input = document.getElementById("inputFeatures").value;

    let features = input.split(",").map(Number);

    let response = await fetch("/predict",{

        method:"POST",

        headers:{
            "Content-Type":"application/json"
        },

        body: JSON.stringify({
            features: features
        })

    });

    let data = await response.json();

    // show prediction
    document.getElementById("result").innerHTML =
        "Prediction: " + (data.prediction === 1 ? "Parkinson Detected" : "Healthy")
        + "<br>Probability: " + (data.probability * 100).toFixed(2) + "%";

    // extract SHAP impacts
    let shapValues = data.top_contributions.map(item => item.impact);

    let shapLabels = data.top_contributions.map(item => "Feature " + item.feature_index);

    drawShapChart(shapLabels, shapValues);
}


function drawShapChart(labels, values){

    const canvas = document.getElementById("shapChart");

    const ctx = canvas.getContext("2d");

    // destroy old chart if exists
    if(window.shapChart){
        window.shapChart.destroy();
    }

    window.shapChart = new Chart(ctx,{
        type:'bar',

        data:{
            labels: labels,

            datasets:[{
                label:"Feature Contribution (SHAP)",

                data: values
            }]
        },

        options:{
            responsive:true,

            scales:{
                y:{
                    beginAtZero:true
                }
            }
        }
    });
}