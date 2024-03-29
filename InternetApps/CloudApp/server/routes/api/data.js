const express = require('express');
const router = express.Router();
const AWS = require('aws-sdk');
const axios = require('axios');
const API_KEY = 'b593175c14a6ed4bcd98411176673fc5'
const BUCKET_URL = 'https://csu44000assignment220.s3.eu-west-1.amazonaws.com/moviedata.json'


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

    //Check if table has been created, if not will retun an error 
    ddb.listTables({Limit: 10}, function(err, data) {
        if (err) {
          console.log("Error, tables not created yet, retry in a moment", err.code);
        } else {
            //Gives it a few seconds for AWS to provision the tables as attempting to add items straight away causes some items not to be added
            setTimeout(async function(){
                var docClient = new AWS.DynamoDB.DocumentClient();
                console.log("Importing movies in to DynamoDB. Please wait");
                let i = 0;

                await movieData.forEach(function(movie){
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
router.post('/', async(req, res)=>{
    console.log(req.body);
    var docClient = new AWS.DynamoDB.DocumentClient();
    let year = Number(req.body.year);
    let title = req.body.title;

    var params = {
        KeyConditionExpression: '#yr = :year and begins_with(title, :title)',
        ExpressionAttributeNames:{
            "#yr": "year"
        },
        ExpressionAttributeValues: {
            ':title': title,
            ':year': year
        },
        TableName: 'Movies'
    };

    await docClient.query(params, function(err, data){
        if (err) console.log("Unable to query. Error: ", JSON.stringify(err, null, 2));
        else {
            console.log("Query succeeded.");
            data.Items.forEach(function(item) {
                console.log(" -", item.year + ": " + item.title
                + " ... " + item.info.genres
                + " ... " + item.info.actors[0]);
            });
            res.send(data.Items);
        }
    });
});
module.exports = router;