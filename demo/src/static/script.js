console.log("Running script");



var slider = document.getElementById("range");
var timeout = null;
slider.oninput = function() {
    if (timeout !== null) {
        clearTimeout(timeout);
    }

    timeout = setTimeout(function () {
        updateTime()
        getMaps()
    }, 600);
} 


var cardTime = document.getElementById("card-time");

function updateTime(){
    let date = document.getElementById('datepicker').value
    let hour = document.getElementById('range').value
    if(hour < 10){
        hour = "0"+hour;
    }
    cardTime.innerHTML = date + ", " + hour + ":00"

}

updateTime()

function getMaps(){
    updateTime()
    var date = document.getElementById('datepicker').value;
    let hour = slider.value;
    let data = {
        "date": date,
        "hour": hour
    }
    console.log("Get maps for", date);
    //Fetch Forecast
    (async () => {
        let response = await fetch('/forecast',{
            method: "POST",
            headers: new Headers({
                'Content-Type': 'application/json'
            }),
            body: JSON.stringify(data)
        })
        let htmlMap = await response.text();
        document.getElementById("forecast_div").innerHTML = htmlMap;
      })();


    //Fetch Real values
      (async () => {
        let data = {
            "date": date,
            "hour": hour
        }
        let response = await fetch('/true',{
            method: "POST",
            headers: new Headers({
                'Content-Type': 'application/json'
            }),
            body: JSON.stringify(data)
        })
        let htmlMap = await response.text();
        document.getElementById("real_div").innerHTML = htmlMap;
      })();
}


document.getElementById("load_maps").addEventListener("click", getMaps);