const express = require('express');
const router = express.Router();
const AWS = require('aws-sdk');
const axios = require('axios');
const API_KEY = 'b593175c14a6ed4bcd98411176673fc5'
const BUCKET_URL = 'https://csu44000assignment220.s3.eu-west-1.amazonaws.com/moviedata.json'

// Create Tables
// TODO: create AWS DynamoDB details
// TODO: Test DB (Attempt to query it)
router.get('/', async(req, res) =>{
    let resp =  await axios.get(BUCKET_URL);
    let movieData = resp.data;
    console.log(movieData.length)
    var ddb = new AWS.DynamoDB({apiVersion: '2012-08-10'})
    console.log("Access Key:", AWS.config.credentials.accessKeyId);
    var params = {
        TableName : "Movies",
        KeySchema: [       
            { AttributeName: "year", KeyType: "HASH"},  //Partition key
            { AttributeName: "title", KeyType: "RANGE" }  
        ],
        AttributeDefinitions: [       
            { AttributeName: "year", AttributeType: "N" },
            { AttributeName: "title", AttributeType: "S" }
        ],
        ProvisionedThroughput: {       
            ReadCapacityUnits: 1, 
            WriteCapacityUnits: 5
        }
    };

    ddb.createTable(params,function(err,data){
        if(err) console.log("Error, table may already be created", err);
        else {
            console.log("Table Created", data);
            res.sendStatus(201);
        }
    });

    ddb.listTables({Limit: 10}, function(err, data) {
        if (err) {
          console.log("Error, tables not created yet, retry in a moment", err.code);
        } else {
            setTimeout(function(){
                var docClient = new AWS.DynamoDB.DocumentClient();
                console.log("Importing movies in to DynamoDB. Please wait");
                let i = 0;

                movieData.forEach(function(movie){
                var params = {
                    TableName: "Movies",
                    Item: {
                        "year":  movie.year,
                        "title": movie.title,
                        "info":  movie.info
                    }
                };
            
                docClient.put(params, function(err, data) {
                   if (err) {
                       console.error("Unable to add movie", movie.title, ". Error JSON:", JSON.stringify(err, null, 2));
                   } else {
                       console.log(`PutItem${i} succeeded: ${movie.title}`);
                       i++;
                   }
                });
            });
            console.log('Done!');
            }, 3000);
        }
      });
});

// Delete Tables
router.get('/delete/', async(req, res)=>{
    var ddb = new AWS.DynamoDB({apiVersion: '2012-08-10'})
    var params = {
        TableName : "Movies"
    };
    ddb.deleteTable(params, function(err, data) {
        if (err){
            console.error("Unable to delete the table, Error JSON", JSON.stringify(err, null, 2));
            res.sendStatus(400);
        }
        else{
            console.log("Deleted table. Table description JSON:", JSON.stringify(data, null, 2));
            res.sendStatus(200);
        }
    });
});
// Query Databse
module.exports = router;