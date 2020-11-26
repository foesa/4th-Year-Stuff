const request = require('requests');
const bodyp = require('body-parser');

request(`https://api.openweathermap.org/data/2.5/forecast?q=dublin,IE&units=metric&appid=b593175c14a6ed4bcd98411176673fc5`, (err,response, body) =>{
    if(!err && response == 200){
        parseJson(body)
    }
});


function parseJson(data) {
        data = JSON.parse(data);
        let cityName = data.city.name;
        let country = data.city.country;
        let currentDay = new Date();
        let days = []
        let weatherData = new Object();
        for(let item in data.list){
            let date = new Date.parse(item.dt_txt);
            if (currentDay.toDateString() === date.toDateString()){
                if (date.getHours() === 12){
                    weatherData.weather = item.weather[0].main;
                    weatherData.temp = item.temp;
                    weatherData.feels_like = item.feels_like;
                    if(weatherData.weather === "Rain"){
                        weatherData.needUmbrella = true;
                    }else{
                        weatherData.needUmbrella = false;
                    }
                    days.push(weatherData);
                }
            }else{
                weatherData = new Object();
            }
        }
    
    let CityData = {
        city: cityName,
        country: country,
        days: days
    };
    
    return CityData;
}