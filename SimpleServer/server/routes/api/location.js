const express = require('express');
const mongodb = require('mongodb');
const axios = require('axios');
const router = express.Router();
const API_KEY = 'b593175c14a6ed4bcd98411176673fc5'
// Get Locations
router.get('/', async (req,res) =>{
    const Locations = await loadLocationCollection();
    res.send(await Locations.find({}).toArray());
});

// Add Locations
router.post('/', async (req, res) => {
    const Locations = await loadLocationCollection();
    console.log(req.body);
    const response = await axios.get(`https://api.openweathermap.org/data/2.5/forecast?q=${req.body.City}&units=metric&appid=${API_KEY}`);
    let data = parseJson(response.data);
    console.log(data)
    await Locations.insertOne(data);
    res.sendStatus(201);
})

async function loadLocationCollection() {
    const client = await mongodb.MongoClient.connect(
        'mongodb+srv://foesa:Random12@cluster0.sdvcb.mongodb.net/Locationdb?retryWrites=true&w=majority',
         {useNewUrlParser: true});

    return client.db('Locationdb').collection('Locations')
}

function parseJson(data) {
    let cityName = data.city.name;
    let country = data.city.country;
    let currentDay = new Date();
    let days = []
    let avgTemp = 0;
    let packFor = "";
    let weatherData = new Object();
    let needBrolly = false;
    for(let item of data.list){
        let date = new Date(parseInt(item.dt) * 1000);
        if (currentDay.toDateString() === date.toDateString()){
            if (date.getHours() === 12){
                weatherData.weather = item.weather[0].main;
                weatherData.temp = item.main.temp;
                avgTemp = avgTemp+item.main.temp;
                weatherData.feels_like = item.main.feels_like;
                weatherData.date = date.toDateString();
                if(weatherData.weather === "Rain"){
                    weatherData.needUmbrella = true;
                    needBrolly = true;
                }else{
                    weatherData.needUmbrella = false;
                }
                days.push(weatherData);
            }
        }else{
            weatherData = new Object();
        }
        currentDay = date;
    }
    avgTemp = avgTemp/5;
    if(-10 <= avgTemp < 10){
        packFor = "Cold";
    }else if (10 <= avgTemp < 20){
        packFor = "Warm";
    }else{
        packFor = "hot";
    }

    let CityData = {
        city: cityName,
        country: country,
        days: days,
        needbrolly: needBrolly,
        createdAt: new Date(),
        packFor: packFor
    };

    return CityData;
}

module.exports = router;